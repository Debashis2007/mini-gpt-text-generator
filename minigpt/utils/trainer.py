"""Utilities for training and generation."""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from minigpt.models.gpt import SimpleGPT
from minigpt.data.dataset import TextDataset, WhitespaceTokenizer


logger = logging.getLogger(__name__)


class Trainer:
    """Training helper for SimpleGPT models."""
    
    def __init__(
        self,
        model: SimpleGPT,
        optimizer: optim.Optimizer,
        device: torch.device,
        save_dir: Union[str, Path],
    ):
        """
        Args:
            model: The GPT model to train
            optimizer: Optimizer for training
            device: Device to train on
            save_dir: Directory to save checkpoints and tokenizer
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(
        self, 
        dataloader: DataLoader,
        epoch: int,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            _, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                logger.info(f"Epoch {epoch} batch {i}: loss {loss.item():.4f}")
                
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save a training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        path = self.save_dir / f'checkpoint-{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


class Generator:
    """Text generation helper for SimpleGPT models."""
    
    def __init__(
        self,
        model: SimpleGPT,
        tokenizer: WhitespaceTokenizer,
        device: torch.device,
    ):
        """
        Args:
            model: Trained GPT model
            tokenizer: Tokenizer used during training
            device: Device to generate on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> str:
        """Generate text continuation from prompt."""
        self.model.eval()
        with torch.no_grad():
            # Encode prompt
            ids = self.tokenizer.encode(prompt)
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            
            # Generate
            y = self.model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            
            # Decode
            generated_ids = y[0].tolist()
            return self.tokenizer.decode(generated_ids)