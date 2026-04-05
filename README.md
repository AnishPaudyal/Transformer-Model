# Transformer Model

A GPT-style autoregressive language model implemented from scratch in PyTorch, following the architecture from *Attention Is All You Need*.

## Architecture

- 6 transformer blocks with multi-head self-attention (4 heads)
- 32-dimensional token and positional embeddings
- Pre-layer normalization for training stability
- Feedforward network with ReLU activation (4× expansion)
- Causal masking to prevent attention to future tokens
- Dropout (0.2) for regularization

## Training

- Optimizer: AdamW with weight decay
- Iterations: 5,000
- Loss: 4.37 → 1.97 (cross-entropy)
- Character-level tokenization on input text corpus
- 90/10 train/test split

## Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

## Run
```bash
pip install torch
python transformer.py
```
