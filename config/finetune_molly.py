import time

out_dir = 'out-molly'
eval_interval = 200
wandb_log = False
wandb_project = 'molly'
wandb_run_name = 'ft-' + str(time.time())
compile = False
always_save_checkpoint = True

dataset = 'molly'
init_from = 'gpt2'
batch_size = 1
block_size = 512

learning_rate = 1e-5
max_iters = 1000
decay_lr = False
