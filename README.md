# Mini GPT Text Generator

This repository contains two approaches to next-word prediction / small text generation:

1. Baseline bigram model (pure Python, no dependencies) — quick to run and demonstrates next-word prediction.
2. Minimal PyTorch-based mini-GPT (educational) — can be trained on `data.txt` to learn next-word prediction with a Transformer.

## Files
- `data.txt` — training data (simple example sentences). Replace with your own data.
- `baseline.py` — trains a bigram frequency model and generates text. No dependencies.
- `gpt.py` — tiny Transformer-style model implemented with PyTorch.
- `train_gpt.py` — training and generation script for the mini-GPT.
- `requirements.txt` — Python packages for training the mini-GPT.

## Quick start — baseline (no installs)

Run the baseline bigram model which trains on `data.txt` and generates example outputs:

```powershell
python baseline.py
```

You should see the model saved as `bigram_model.json` and several generated samples.

## Train the mini-GPT (requires PyTorch)

1. Create a virtual environment and install requirements (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train the model:

```powershell
python train_gpt.py --train --epochs 1
```

This will save:
- `model.pt` - trained PyTorch model weights
- `tokenizer.json` - simple whitespace vocabulary

Training results (demo run):
- Vocabulary size: 36 tokens
- Training time: ~1-2 seconds on CPU
- Final loss: ~3.43 (1 epoch)

## Generate with the trained model

```powershell
python train_gpt.py --generate "hello"
```

The model will complete the prompt with generated text based on its training.

## Command line options

```
--train              Run model training
--generate TEXT      Generate text from a prompt
--epochs N          Number of training epochs (default: 1)
--block_size N      Context window size (default: 8)
--batch_size N      Training batch size (default: 16)
```

## Implementation Details

Current version features:
- Minimal but complete Transformer architecture (see `gpt.py`)
- Simple whitespace tokenization with frequency filtering
- Efficient PyTorch Dataset/DataLoader for training
- Clean command-line interface
- Demo-scale hyperparameters for quick testing

## Next Steps

Planned improvements:
1. Replace whitespace tokenizer with BPE/subword tokenization
2. Add model checkpointing and training resumption
3. Implement proper temperature control for generation
4. Add validation set and perplexity metrics
5. Scale up model size and training parameters

For better results:
- Use more training data
- Increase model size (`n_embd`, `n_head`, `n_layer`)
- Train for more epochs
- Implement learning rate scheduling
- Add dropout for regularization

## Notes
- The provided GPT is intentionally small for demonstration
- Current tokenizer is word-level (whitespace) for simplicity
- Security note: Mind PyTorch model loading warning in production use