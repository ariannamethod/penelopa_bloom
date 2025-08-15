"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run in debug mode example:
$ python train.py --batch_size=32 --other=args

To run DDP on 4 gpus on one node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
import time
import math
import logging
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader


import argparse
import ast
from pathlib import Path
from typing import Any, Dict

import yaml

def _str2bool(v: str) -> bool:
    if v.lower() in {"1", "true", "yes", "y"}:
        return True
    if v.lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("boolean value expected")

def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_eval_node(n) for n in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(n) for n in node.elts)
    if isinstance(node, ast.Dict):
        return {_eval_node(k): _eval_node(v) for k, v in zip(node.keys, node.values)}
    raise ValueError("Unsupported expression in config file")

def _read_config_file(path: Path) -> Dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if path.suffix == ".py":
        with open(path, "r", encoding="utf-8") as f:
            node = ast.parse(f.read(), filename=str(path))
        cfg: Dict[str, Any] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                cfg[stmt.targets[0].id] = _eval_node(stmt.value)
        return cfg
    raise ValueError(f"Unsupported config file: {path}")

def update_config(cfg: Dict[str, Any]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to .py or .yaml config")
    for key, value in list(cfg.items()):
        if key.startswith("_"):
            continue
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=_str2bool)
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int)
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float)
        elif isinstance(value, str):
            parser.add_argument(f"--{key}", type=str)
        elif isinstance(value, tuple) and all(isinstance(x, float) for x in value):
            parser.add_argument(f"--{key}", type=lambda s: tuple(float(x) for x in s.split(',')))
    args = parser.parse_args()

    file_cfg: Dict[str, Any] = {}
    if args.config:
        file_cfg = _read_config_file(Path(args.config))

    for k, v in file_cfg.items():
        if k not in cfg:
            raise ValueError(f"Unknown config key: {k}")
        cfg[k] = v

    for k, v in vars(args).items():
        if k == "config" or v is None:
            continue
        cfg[k] = v


from molly import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-2
betas = (0.9, 0.95)
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# gradient clipping
grad_clip = 1.0
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
update_config(globals())
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
else:
    gpu_id = 0 # gpu_id 0 means this is the (single) master process, basically

if gpu_id == 0:
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
torch.manual_seed(1337 + gpu_id) # note: each worker gets a different seed
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 would require us to change the code to use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
train_config = {
    "dataset": dataset,
    "batch_size": batch_size,
    "block_size": block_size,
    "n_layer": n_layer,
    "n_head": n_head,
    "n_embd": n_embd,
    "dropout": dropout,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "betas": betas,
    "warmup_iters": warmup_iters,
    "lr_decay_iters": lr_decay_iters,
    "min_lr": min_lr,
    "grad_clip": grad_clip,
}
if gpu_id == 0:
    logging.info("Training configuration: %s", train_config)


class BinDataset(Dataset):
    def __init__(self, data_path, block_size):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        return x, y


train_dataset = BinDataset(os.path.join(data_dir, 'train.bin'), block_size)
val_dataset = BinDataset(os.path.join(data_dir, 'val.bin'), block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

train_iter = iter(train_loader)
val_iter = iter(val_loader)


def get_batch(split):
    global train_iter, val_iter
    if split == 'train':
        loader, data_iter = train_loader, train_iter
    else:
        loader, data_iter = val_loader, val_iter
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        if split == 'train':
            train_iter = data_iter
        else:
            val_iter = data_iter
        x, y = next(data_iter)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer = n_layer, n_head = n_head, n_embd = n_embd, block_size = block_size, dropout = dropout)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k, v in model_args.items():
        assert checkpoint_model_args[k] == v, "for now"
        # NOTE: interaction between passed params and checkpoint params requires design review (see issue #1)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off and override the GPT sizing model args from the model config
    model_args['n_layer'] = model.config.n_layer
    model_args['n_head'] = model.config.n_head
    model_args['n_embd'] = model.config.n_embd
# crop down the model block size if desired
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
model.to(device)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, betas)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[gpu_id])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(iter):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) if iter > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and gpu_id == 0:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)
    wandb.config.update(train_config)

# training loop
t0 = time.time()
while True:

    # determine the learning rate for this iteration
    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = learning_rate

    if iter_num % eval_interval == 0 and gpu_id == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            raw_model = model.module if ddp else model
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    X, Y = get_batch('train')
    with ctx:
        logits, loss = model(X, Y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and gpu_id == 0:
        lossf = loss.item() # loss as float. NOTE: CPU-GPU sync; profile to ensure it's not a bottleneck (see issue #2)
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
