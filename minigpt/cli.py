"""Command-line interface for training and generation."""

import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from minigpt.models.gpt import SimpleGPT
from minigpt.data.dataset import TextDataset, WhitespaceTokenizer
from minigpt.utils.trainer import Trainer, Generator


logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def read_texts(path: Path) -> list[str]:
    """Read training texts from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def train(args):
    """Train a new model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and process data
    texts = read_texts(args.data)
    tokenizer = WhitespaceTokenizer(texts, min_freq=args.min_freq)
    tokenizer.save(args.save_dir / 'tokenizer.json')
    
    examples = [tokenizer.encode(text) for text in texts]
    dataset = TextDataset(examples, args.block_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Create model
    model = SimpleGPT(
        vocab_size=len(tokenizer.itos),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout
    ).to(device)
    
    # Set up training
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    trainer = Trainer(model, optimizer, device, args.save_dir)
    
    # Train
    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        loss = trainer.train_epoch(dataloader, epoch)
        logger.info(f"Epoch {epoch}/{args.epochs} average loss: {loss:.4f}")
        
        if epoch % args.save_every == 0:
            trainer.save_checkpoint(epoch, loss)
    
    # Save final model
    torch.save(model.state_dict(), args.save_dir / 'model.pt')
    logger.info("Training complete!")


def generate(args):
    """Generate text from a trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer and model
    tokenizer = WhitespaceTokenizer.load(args.save_dir / 'tokenizer.json')
    model = SimpleGPT(
        vocab_size=len(tokenizer.itos),
        block_size=args.block_size
    ).to(device)
    model.load_state_dict(
        torch.load(args.save_dir / 'model.pt', map_location=device)
    )
    
    # Generate
    generator = Generator(model, tokenizer, device)
    output = generator.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print(output)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train and run a minimal GPT model'
    )
    parser.add_argument(
        '--data', 
        type=Path,
        default=Path('data.txt'),
        help='Training data file'
    )
    parser.add_argument(
        '--save-dir',
        type=Path,
        default=Path('outputs'),
        help='Directory to save model and tokenizer'
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Training arguments
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--epochs', type=int, default=1)
    train_parser.add_argument('--batch-size', type=int, default=16)
    train_parser.add_argument('--block-size', type=int, default=8)
    train_parser.add_argument('--min-freq', type=int, default=1)
    train_parser.add_argument('--n-layer', type=int, default=1)
    train_parser.add_argument('--n-head', type=int, default=2)
    train_parser.add_argument('--n-embd', type=int, default=64)
    train_parser.add_argument('--dropout', type=float, default=0.0)
    train_parser.add_argument('--learning-rate', type=float, default=5e-4)
    train_parser.add_argument('--weight-decay', type=float, default=0.01)
    train_parser.add_argument('--save-every', type=int, default=1)
    
    # Generation arguments
    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('prompt', type=str)
    generate_parser.add_argument('--max-tokens', type=int, default=20)
    generate_parser.add_argument('--temperature', type=float, default=1.0)
    generate_parser.add_argument('--top-k', type=int)
    generate_parser.add_argument('--block-size', type=int, default=8)
    
    args = parser.parse_args()
    setup_logging()
    
    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.command == 'train':
        train(args)
    else:
        generate(args)


if __name__ == '__main__':
    main()