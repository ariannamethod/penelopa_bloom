import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from penelopa import PenelopaConfig, Penelopa


class BinDataset(Dataset):
    """Load tokenized data from a .bin file."""

    def __init__(self, data_path: str, block_size: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64))
        return x, y


def estimate_loss(model: Penelopa, loader: DataLoader, device: str) -> float:
    model.eval()
    losses: list[float] = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or finetune Penelopa")
    parser.add_argument("--dataset", default="shakespeare", help="dataset name under data/")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--out_dir", default="out")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join("data", args.dataset)
    train_loader = DataLoader(
        BinDataset(os.path.join(data_dir, "train.bin"), args.block_size),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        BinDataset(os.path.join(data_dir, "val.bin"), args.block_size),
        batch_size=args.batch_size,
    )

    config = PenelopaConfig(
        block_size=args.block_size,
        vocab_size=50257,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.1,
    )
    model = Penelopa(config).to(device)
    optimizer = model.configure_optimizers(1e-2, args.learning_rate, (0.9, 0.95))

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    iter_num = 0
    while iter_num < args.max_iters:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            loss.backward()
            optimizer.step()
            iter_num += 1

            if iter_num % args.eval_interval == 0:
                val_loss = estimate_loss(model, val_loader, device)
                print(f"iter {iter_num}: val loss {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt.pt"))

            if iter_num >= args.max_iters:
                break


if __name__ == "__main__":
    main()
