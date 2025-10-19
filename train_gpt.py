"""
Minimal trainer/generator for SimpleGPT (whitespace-only tokenizer).

This file is a clean replacement for the corrupted script and is kept
intentionally small to make testing reliable.
"""

import json
from collections import Counter
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from gpt import SimpleGPT


DATA_PATH = "data.txt"
MODEL_PATH = "model.pt"
TOK_PATH = "tokenizer.json"


def read_texts(path=DATA_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


class WordDataset(Dataset):
    def __init__(self, examples, block_size, pad_idx=0):
        self.block_size = block_size
        self.pad_idx = pad_idx
        self.data = []
        for ids in examples:
            if len(ids) < 2:
                continue
            for i in range(len(ids)):
                start = max(0, i - block_size + 1)
                chunk = ids[start:i+1]
                if len(chunk) < 2:
                    continue
                if len(chunk) < block_size:
                    chunk = [pad_idx] * (block_size - len(chunk)) + chunk
                self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:], dtype=torch.long)
        return x, y


def build_whitespace_tokenizer(texts, min_freq=1):
    counter = Counter()
    for line in texts:
        counter.update(line.split())
    words = [w for w, c in counter.most_common() if c >= min_freq]
    itos = ['<pad>', '<unk>'] + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def texts_to_ids(texts, stoi):
    examples = []
    for t in texts:
        ids = [stoi.get(w, stoi['<unk>']) for w in t.split()]
        examples.append(ids)
    return examples


def train(epochs=1, block_size=8, batch_size=16, min_freq=1):
    texts = read_texts()
    stoi, itos = build_whitespace_tokenizer(texts, min_freq=min_freq)
    with open(TOK_PATH, 'w', encoding='utf-8') as f:
        json.dump({'itos': itos}, f, ensure_ascii=False, indent=2)

    vocab_size = len(itos)
    print(f"Vocab size: {vocab_size}")

    examples = texts_to_ids(texts, stoi)
    pad_idx = 0
    dataset = WordDataset(examples, block_size, pad_idx=pad_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGPT(vocab_size=vocab_size, n_embd=64, n_head=2, n_layer=1, block_size=block_size).to(device)
    opt = optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), yb.view(B * T))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / max(1, len(loader))
        print(f"Epoch {epoch}/{epochs} avg loss: {avg:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved whitespace tokenizer to {TOK_PATH}")


def generate(prompt, max_new=20):
    with open(TOK_PATH, 'r', encoding='utf-8') as f:
        tok = json.load(f)
    itos = tok['itos']
    stoi = {w: i for i, w in enumerate(itos)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block_size = 8
    model = SimpleGPT(vocab_size=len(itos), n_embd=64, n_head=2, n_layer=1, block_size=block_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    ids = [stoi.get(w, stoi['<unk>']) for w in prompt.split()]
    if len(ids) == 0:
        idx = torch.tensor([[0]], dtype=torch.long, device=device)
    else:
        idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new)
    out_ids = out[0].tolist()
    return ' '.join([itos[i] if i < len(itos) else '<unk>' for i in out_ids])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', action='store_true')
    p.add_argument('--generate', type=str, default=None)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--block_size', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=16)
    args = p.parse_args()
    if args.train:
        train(epochs=args.epochs, block_size=args.block_size, batch_size=args.batch_size)
    elif args.generate is not None:
        print(generate(args.generate))
    else:
        p.print_help()


if __name__ == '__main__':
    main()