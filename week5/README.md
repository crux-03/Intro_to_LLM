# Week 5: Evaluation, Decoding, and Model Management

Now that you can build and train GPT, you need to know how to evaluate it, generate good text from it, and save/load your work.

## What You'll Learn

- **Evaluation**: Loss, perplexity, train/val/test splits, overfitting detection
- **Decoding**: Temperature, top-k, top-p sampling strategies
- **Model Management**: Saving, loading, checkpoints, pretrained weights

## Lessons

Run each lesson in order:

```bash
python 01_evaluation.py      # ~25 min - Loss and perplexity
python 02_decoding.py        # ~30 min - Sampling strategies
python 03_saving_loading.py  # ~25 min - Model persistence
```

## This Week's Project

A complete evaluation and generation system:

```bash
python project_complete.py
```

## Perplexity

The standard metric for language models:

```
Perplexity = exp(average cross-entropy loss)
```

Intuition: "How many tokens is the model choosing between?"
- Perplexity of 10 = as confused as choosing from 10 equally likely tokens
- Lower is better!

| Model | Typical Perplexity |
|-------|-------------------|
| Random (50K vocab) | 50,000 |
| N-gram | 200-500 |
| LSTM | 50-100 |
| GPT-2 | 15-30 |

## Decoding Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Greedy | Always pick highest prob | Deterministic output |
| Temperature | Scale logits by T | Control randomness |
| Top-k | Sample from top k only | Avoid unlikely tokens |
| Top-p | Sample from smallest set summing to p | Adaptive filtering |

**ChatGPT defaults**: Temperature ~0.7, Top-p ~0.9

## Quick Reference

```python
# Perplexity
perplexity = math.exp(loss)

# Temperature sampling
logits = logits / temperature
probs = F.softmax(logits, dim=-1)
token = torch.multinomial(probs, 1)

# Top-k sampling
top_k_logits, top_k_idx = torch.topk(logits, k)
# Sample from top_k_logits only

# Top-p sampling
sorted_probs = sort(probs, descending=True)
cumsum = torch.cumsum(sorted_probs, dim=-1)
# Keep tokens until cumsum > p

# Save model
torch.save(model.state_dict(), 'model.pt')

# Load model
model.load_state_dict(torch.load('model.pt'))
```

## Next Week Preview

In Week 6, we'll learn fine-tuning: adapting pretrained models for classification tasks like sentiment analysis and spam detection.
