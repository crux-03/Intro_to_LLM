# Week 3: Attention Mechanisms

This is the most important week in the course. Attention is the breakthrough that made transformers - and GPT - possible.

## What You'll Learn

- **Basic Attention**: The Query-Key-Value framework
- **Self-Attention**: Learnable projections for Q, K, V
- **Causal Masking**: How GPT prevents "peeking" at future tokens
- **Multi-Head Attention**: Running multiple attention patterns in parallel

## Lessons

Run each lesson in order:

```bash
python 01_attention.py       # ~30 min - The attention mechanism
python 02_self_attention.py  # ~30 min - Learnable projections
python 03_causal_masking.py  # ~30 min - Masking for GPT
python 04_multihead.py       # ~45 min - Multi-head attention
```

## This Week's Project

Build and test the complete attention module:

```bash
python project_attention.py
```

You'll create the exact attention mechanism used in GPT.

## Key Concepts

### The Attention Formula

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

- **Q (Query)**: What am I looking for?
- **K (Key)**: What do I have to offer?
- **V (Value)**: What information do I return?

### Why Attention Matters

Before attention, models processed text sequentially:
```
The → cat → sat → on → the → mat
```

With attention, every word can directly "look at" every other word:
```
"it" can directly attend to "cat" regardless of distance
```

### Causal Masking

GPT generates text left-to-right. During training, we prevent the model from seeing future tokens:

```
Position 0: Can see [0]
Position 1: Can see [0, 1]
Position 2: Can see [0, 1, 2]
Position 3: Can see [0, 1, 2, 3]
```

### Multi-Head Attention

Instead of one attention pattern, we run multiple in parallel:
- Head 1 might learn syntactic relationships
- Head 2 might learn semantic similarity
- Head 3 might learn positional patterns

## Quick Reference

```python
# Basic attention
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
weights = F.softmax(scores, dim=-1)
output = weights @ V

# Causal mask
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float('-inf'))

# Multi-head reshape
# (batch, seq, d_model) -> (batch, heads, seq, head_dim)
x = x.view(batch, seq, num_heads, head_dim).transpose(1, 2)
```

## GPT-2 Attention Specs

| Model | d_model | num_heads | head_dim |
|-------|---------|-----------|----------|
| GPT-2 Small | 768 | 12 | 64 |
| GPT-2 Medium | 1024 | 16 | 64 |
| GPT-2 Large | 1280 | 20 | 64 |

Notice: head_dim is always 64. Larger models add more heads.

## Next Week Preview

In Week 4, we'll add Feed-Forward Networks and Layer Normalization to build complete Transformer blocks, then stack them to create the full GPT architecture.
