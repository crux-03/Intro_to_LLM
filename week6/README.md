# Week 6: Fine-Tuning for Classification

You've built GPT from scratch and understand pretraining. Now we'll learn fine-tuning: adapting a pretrained model for specific tasks.

## What You'll Learn

- **Fine-tuning basics**: Why transfer learning works, classification heads
- **Sentiment analysis**: Binary classification (positive/negative)
- **Spam detection**: Real-world binary classification
- **Topic classification**: Multi-class classification

## Lessons

Run each lesson in order:

```bash
python 01_finetuning_intro.py    # ~20 min - Fine-tuning fundamentals
python 02_sentiment.py           # ~25 min - Sentiment classifier
python 03_spam_multiclass.py     # ~30 min - Spam + topic classification
```

## This Week's Project

Complete classification system:

```bash
python project_classifiers.py
```

## Why Fine-Tuning Works

```
PRETRAINING = General education
    - Grammar, syntax, world knowledge
    - Takes months, costs millions
    
FINE-TUNING = Specialized training  
    - Already knows language
    - Just learns the task
    - Takes hours, costs little
```

## Classification Architecture

```
GPT outputs: (batch, seq_length, d_model)
                    |
                    v
Take last token: (batch, d_model)
                    |
                    v
Classification head: Linear(d_model, num_classes)
                    |
                    v
Logits: (batch, num_classes)
```

## Freezing Strategies

| Strategy | What's Trainable | Use When |
|----------|-----------------|----------|
| Head only | Classification head | Little data, fast results |
| Top layers | Head + top N blocks | Moderate data |
| Full | Everything | Lots of data, best accuracy |

## Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correct |
| Precision | TP/(TP+FP) | "When we say positive, how often right?" |
| Recall | TP/(TP+FN) | "How many positives did we catch?" |
| F1 | 2*P*R/(P+R) | Balance of precision and recall |

## Quick Reference

```python
# Classification head
class GPTClassifier(nn.Module):
    def __init__(self, gpt, num_classes):
        self.gpt = gpt
        d_model = gpt.token_embedding.weight.shape[1]
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        hidden = self.get_hidden_states(x)
        last = hidden[:, -1, :]  # Last token
        return self.classifier(last)

# Freeze GPT weights
for param in model.gpt.parameters():
    param.requires_grad = False

# Only classifier is trainable now
```

## Next Week Preview

In Week 7, we'll learn instruction fine-tuning: making the model follow commands like ChatGPT!
