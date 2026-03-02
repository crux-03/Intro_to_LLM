# Week 4: Building GPT

This is the culmination of everything you've learned. We assemble all the pieces into a complete GPT model and train it to generate text.

## What You'll Learn

- **Transformer Blocks**: Layer norm, feed-forward networks, residual connections
- **Complete GPT**: Token embeddings, position embeddings, stacked blocks
- **Training**: Loss computation, optimization, generation

## Lessons

Run each lesson in order:

```bash
python 01_transformer_block.py  #  The building block
python 02_complete_gpt.py       #  Full GPT architecture
python 03_training.py           # Training on Shakespeare
```

## This Week's Project

Build and train a complete GPT model:

```bash
python project_gpt.py
```

By the end, your model will generate Shakespeare-like text!

## The Transformer Block

Each block contains:
```
Input
  │
  ├──────────────┐
  ▼              │ (residual)
LayerNorm        │
  │              │
  ▼              │
Attention        │
  │              │
  ▼              │
+ ◄──────────────┘
  │
  ├──────────────┐
  ▼              │ (residual)
LayerNorm        │
  │              │
  ▼              │
Feed-Forward     │
  │              │
  ▼              │
+ ◄──────────────┘
  │
Output
```

## Complete GPT Architecture

```
Token IDs
    │
    ▼
Token Embedding + Position Embedding
    │
    ▼
┌─────────────────┐
│ Transformer     │ ─┐
│ Block 1         │  │
└─────────────────┘  │
        │            │  × num_layers
        ▼            │
┌─────────────────┐  │
│ Block 2...N     │  │
└─────────────────┘  │
        │        ────┘
        ▼
Final LayerNorm
    │
    ▼
Linear → Vocabulary
    │
    ▼
Logits (next token probabilities)
```

## GPT-2 Configurations

| Model | d_model | heads | layers | Parameters |
|-------|---------|-------|--------|------------|
| GPT-2 Small | 768 | 12 | 12 | 124M |
| GPT-2 Medium | 1024 | 16 | 24 | 350M |
| GPT-2 Large | 1280 | 20 | 36 | 774M |
| GPT-2 XL | 1600 | 25 | 48 | 1.5B |

## Quick Reference

```python
# Transformer block
class TransformerBlock(nn.Module):
    def forward(self, x):
        x = x + self.attention(self.ln1(x))  # Attention + residual
        x = x + self.ffn(self.ln2(x))        # FFN + residual
        return x

# Training step
logits, loss = model(input_ids, targets)
loss.backward()
optimizer.step()

# Generation
for _ in range(num_tokens):
    logits, _ = model(input_ids)
    next_token = sample(logits[:, -1, :])
    input_ids = torch.cat([input_ids, next_token], dim=1)
```

## Next Week Preview

In Week 5, we'll cover evaluation metrics (perplexity), decoding strategies (top-k, top-p), and saving/loading models.
