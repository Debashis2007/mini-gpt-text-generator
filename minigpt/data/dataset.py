"""Data loading and processing utilities."""

import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Dataset for training language models on text data."""
    
    def __init__(
        self, 
        examples: List[List[int]], 
        block_size: int, 
        pad_idx: int = 0
    ):
        """
        Args:
            examples: List of token ID sequences
            block_size: Size of context window
            pad_idx: Token ID to use for padding (default: 0)
        """
        self.block_size = block_size
        self.pad_idx = pad_idx
        self.data = []
        
        # Process examples into training pairs
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class WhitespaceTokenizer:
    """Simple whitespace-based tokenizer with frequency filtering."""
    
    def __init__(
        self, 
        texts: Optional[List[str]] = None, 
        min_freq: int = 1
    ):
        """
        Args:
            texts: Optional list of texts to build vocabulary from
            min_freq: Minimum frequency threshold for tokens
        """
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        
        if texts:
            self.build_vocab(texts, min_freq)
    
    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """Build vocabulary from texts."""
        counter = Counter()
        for text in texts:
            counter.update(text.split())
            
        # Add special tokens first
        self.itos = [self.pad_token, self.unk_token]
        
        # Add frequent tokens
        words = [w for w, c in counter.most_common() if c >= min_freq]
        self.itos.extend(words)
        
        # Create lookup
        self.stoi = {w: i for i, w in enumerate(self.itos)}
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return [self.stoi.get(w, self.stoi[self.unk_token]) 
                for w in text.split()]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return ' '.join(self.itos[i] if i < len(self.itos) else self.unk_token 
                       for i in ids)
    
    def save(self, path: Path) -> None:
        """Save tokenizer vocabulary to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'itos': self.itos}, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'WhitespaceTokenizer':
        """Load tokenizer vocabulary from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.itos = data['itos']
        tokenizer.stoi = {w: i for i, w in enumerate(tokenizer.itos)}
        return tokenizer