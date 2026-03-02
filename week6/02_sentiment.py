"""
Lesson 2: Sentiment Analysis
=============================

Now let's do a COMPLETE fine-tuning example!

We'll train a sentiment classifier:
    Input: Movie review text
    Output: Positive or Negative

This is one of the most common NLP tasks and a great
way to understand fine-tuning in practice.

We'll cover:
    - Preparing a sentiment dataset
    - Fine-tuning our GPT classifier
    - Evaluating with accuracy, precision, recall, F1
    - Making predictions on new text

Usage: python 02_sentiment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import re
from collections import Counter
import random


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 2: Sentiment Analysis")
print("  Fine-Tuning for Classification")
print("=" * 60)

print("""
Let's build a complete sentiment classifier!

Task: Given a movie review, predict if it's positive or negative.

    "This movie was amazing!" -> Positive
    "Terrible waste of time." -> Negative

We'll:
    1. Create a sentiment dataset
    2. Build our tokenizer
    3. Fine-tune GPT for classification
    4. Evaluate accuracy, precision, recall, F1
    5. Make predictions on new reviews
""")

pause()


# ---------------------------------------------------------------------------
# The Dataset
# ---------------------------------------------------------------------------

print("THE SENTIMENT DATASET")
print("-" * 40)

POSITIVE_REVIEWS = [
    "This movie was absolutely amazing! Great acting and story.",
    "I loved every minute of this film. Highly recommended!",
    "Brilliant performances and stunning visuals. A masterpiece.",
    "One of the best movies I've ever seen. Truly inspiring.",
    "Fantastic film with great characters and plot.",
    "A wonderful movie experience. I was moved to tears.",
    "Excellent direction and superb acting throughout.",
    "This is a must-see film. Entertaining from start to finish.",
    "Beautiful cinematography and a touching story.",
    "I can't recommend this movie enough. Simply outstanding.",
    "A delightful film that exceeded all my expectations.",
    "The acting was superb and the story was captivating.",
    "An incredible journey that left me speechless.",
    "This movie restored my faith in cinema. Bravo!",
    "A perfect blend of humor, drama, and action.",
    "I was thoroughly entertained. Great movie!",
    "The best film of the year without a doubt.",
    "A heartwarming story with memorable characters.",
    "Absolutely loved it. Will watch again!",
    "A triumphant achievement in filmmaking.",
]

NEGATIVE_REVIEWS = [
    "This movie was terrible. Complete waste of time.",
    "I hated every minute of this boring film.",
    "Awful acting and a nonsensical plot. Avoid this.",
    "One of the worst movies I've ever seen. Disappointing.",
    "A complete disaster from start to finish.",
    "I want my two hours back. Absolutely dreadful.",
    "Poor direction and wooden performances throughout.",
    "This is a skip. Not worth your time or money.",
    "Boring, predictable, and poorly executed.",
    "I couldn't wait for this movie to end. Terrible.",
    "A disappointing mess of a film.",
    "The acting was atrocious and the plot made no sense.",
    "An embarrassing attempt at filmmaking.",
    "This movie insulted my intelligence. Awful!",
    "A painful experience. Do not watch.",
    "I was thoroughly bored. Bad movie!",
    "The worst film of the year by far.",
    "A forgettable and tedious experience.",
    "Absolutely hated it. Never again!",
    "A failed attempt at entertainment.",
]

print(f"Positive reviews: {len(POSITIVE_REVIEWS)}")
print(f"Negative reviews: {len(NEGATIVE_REVIEWS)}")
print()
print("Sample positive review:")
print(f"  '{POSITIVE_REVIEWS[0]}'")
print()
print("Sample negative review:")
print(f"  '{NEGATIVE_REVIEWS[0]}'")

pause()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

print("BUILDING THE TOKENIZER")
print("-" * 40)


class SimpleTokenizer:
    def __init__(self, texts, max_vocab=1000):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        
        # Add most common words
        for word, _ in word_counts.most_common(max_vocab - 2):
            self.word_to_idx[word] = len(self.word_to_idx)
        
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
    
    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()
    
    def encode(self, text, max_length=None):
        words = self._tokenize(text)
        ids = [self.word_to_idx.get(w, 1) for w in words]  # 1 is <UNK>
        
        if max_length:
            if len(ids) < max_length:
                ids = ids + [0] * (max_length - len(ids))  # Pad
            else:
                ids = ids[:max_length]  # Truncate
        
        return ids
    
    def decode(self, ids):
        return ' '.join(self.idx_to_word.get(i, '<UNK>') for i in ids if i != 0)


# Build tokenizer
all_texts = POSITIVE_REVIEWS + NEGATIVE_REVIEWS
tokenizer = SimpleTokenizer(all_texts)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print()

# Test encoding
text = "This movie was amazing!"
encoded = tokenizer.encode(text, max_length=10)
decoded = tokenizer.decode(encoded)

print(f"Original: '{text}'")
print(f"Encoded:  {encoded}")
print(f"Decoded:  '{decoded}'")

pause()


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

print("CREATING THE DATASET")
print("-" * 40)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=32):
        self.encodings = [tokenizer.encode(t, max_length) for t in texts]
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx]),
            torch.tensor(self.labels[idx])
        )


# Prepare data
texts = POSITIVE_REVIEWS + NEGATIVE_REVIEWS
labels = [1] * len(POSITIVE_REVIEWS) + [0] * len(NEGATIVE_REVIEWS)

# Shuffle
random.seed(42)
combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)

# Split into train/test
split = int(0.8 * len(texts))
train_texts, test_texts = texts[:split], texts[split:]
train_labels, test_labels = labels[:split], labels[split:]

# Create datasets
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

x, y = train_dataset[0]
print(f"\nSample input shape: {x.shape}")
print(f"Sample label: {y.item()} ({'Positive' if y.item() == 1 else 'Negative'})")

pause()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

print("BUILDING THE CLASSIFIER")
print("-" * 40)


# Model components
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_length=512, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_length, max_seq_length)))
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        return self.out_dropout(self.out_proj((weights @ V).transpose(1, 2).reshape(B, T, C)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, num_heads, max_seq_length, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPTClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes, 
                 max_seq_length=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, None, max_seq_length, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_final(x)
        
        # Use last token for classification
        last_hidden = x[:, -1, :]
        logits = self.classifier(last_hidden)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return logits, loss


# Create classifier
model = GPTClassifier(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    num_heads=4,
    num_layers=2,
    num_classes=2,
    max_seq_length=32,
    dropout=0.1
)

params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,}")

pause()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

print("TRAINING THE CLASSIFIER")
print("-" * 40)


def train_classifier(model, train_loader, test_loader, epochs=20, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                logits, _ = model(x)
                test_correct += (logits.argmax(1) == y).sum().item()
                test_total += y.size(0)
        
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")
    
    return model


print("Training...")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

model = train_classifier(model, train_loader, test_loader, epochs=20, lr=1e-3)
print("\nTraining complete!")

pause()


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

print("DETAILED EVALUATION METRICS")
print("-" * 40)

print("""
For classification, we care about more than just accuracy:

ACCURACY: Overall correct predictions
    (TP + TN) / (TP + TN + FP + FN)

PRECISION: Of predicted positives, how many are correct?
    TP / (TP + FP)
    "When we say positive, how often are we right?"

RECALL: Of actual positives, how many did we catch?
    TP / (TP + FN)
    "How many positives did we miss?"

F1 SCORE: Harmonic mean of precision and recall
    2 * (precision * recall) / (precision + recall)
    "Balance between precision and recall"
""")

pause()


def compute_metrics(model, data_loader):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            logits, _ = model(x)
            preds = logits.argmax(1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
    
    # Confusion matrix values
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    }


metrics = compute_metrics(model, test_loader)

print("Test Set Metrics:")
print("-" * 30)
print(f"Accuracy:  {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall:    {metrics['recall']:.2%}")
print(f"F1 Score:  {metrics['f1']:.2%}")
print()
print("Confusion Matrix:")
print(f"  TP={metrics['confusion']['TP']} (correctly predicted positive)")
print(f"  TN={metrics['confusion']['TN']} (correctly predicted negative)")
print(f"  FP={metrics['confusion']['FP']} (false positive)")
print(f"  FN={metrics['confusion']['FN']} (false negative)")

pause()


# ---------------------------------------------------------------------------
# Making Predictions
# ---------------------------------------------------------------------------

print("MAKING PREDICTIONS ON NEW TEXT")
print("-" * 40)


def predict_sentiment(model, tokenizer, text):
    model.eval()
    
    input_ids = torch.tensor([tokenizer.encode(text, max_length=32)])
    
    with torch.no_grad():
        logits, _ = model(input_ids)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(1).item()
    
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = probs[0, pred].item()
    
    return sentiment, confidence


new_reviews = [
    "I absolutely loved this movie! Best film ever!",
    "What a terrible waste of my time. Awful!",
    "It was okay, nothing special.",
    "A masterpiece of modern cinema!",
    "I fell asleep halfway through. Boring.",
]

print("Predictions on new reviews:")
print("-" * 50)
for review in new_reviews:
    sentiment, confidence = predict_sentiment(model, tokenizer, review)
    print(f"\n'{review[:45]}...'")
    print(f"  -> {sentiment} ({confidence:.1%} confident)")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
You've built a complete sentiment classifier!

WHAT WE DID:

1. PREPARED DATA
   - Created positive/negative review dataset
   - Built tokenizer from training data
   - Split into train/test sets

2. BUILT CLASSIFIER
   - GPT backbone with transformer blocks
   - Classification head on last token
   - Cross-entropy loss for training

3. TRAINED MODEL
   - AdamW optimizer
   - Gradient clipping
   - Monitored train/test accuracy

4. EVALUATED
   - Accuracy, Precision, Recall, F1
   - Confusion matrix
   - Understand model mistakes

5. MADE PREDICTIONS
   - Tokenize new text
   - Forward pass through model
   - Get sentiment + confidence

Next: Spam detection and multi-class classification!
""")

print("=" * 60)
print("  End of Lesson 2")
print("=" * 60)
