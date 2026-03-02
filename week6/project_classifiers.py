"""
Week 6 Project: Complete Classification System
================================================

This project brings together everything from Week 6:
    - Sentiment Analysis
    - Spam Detection  
    - Topic Classification
    - All evaluation metrics

You'll build a unified classification system that can
handle multiple tasks.

Usage: python project_classifiers.py
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
print("  Week 6 Project: Complete Classification System")
print("=" * 60)


# ===========================================================================
# COMPONENTS
# ===========================================================================

class Tokenizer:
    """Simple word-level tokenizer."""
    
    def __init__(self, texts, max_vocab=1000):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        for word, _ in word_counts.most_common(max_vocab - 2):
            self.word_to_idx[word] = len(self.word_to_idx)
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
    
    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()
    
    def encode(self, text, max_length=None):
        words = self._tokenize(text)
        ids = [self.word_to_idx.get(w, 1) for w in words]
        if max_length:
            if len(ids) < max_length:
                ids = ids + [0] * (max_length - len(ids))
            else:
                ids = ids[:max_length]
        return ids


class TextDataset(Dataset):
    """Dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=32):
        self.encodings = [tokenizer.encode(t, max_length) for t in texts]
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx]), torch.tensor(self.labels[idx])


class TransformerClassifier(nn.Module):
    """Transformer-based text classifier."""
    
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, 
                 num_classes=2, max_length=32, dropout=0.1):
        super().__init__()
        self.max_length = max_length
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            self._make_block(d_model, num_heads, max_length, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Causal mask
        self.register_buffer('mask', torch.tril(torch.ones(max_length, max_length)))
    
    def _make_block(self, d_model, num_heads, max_length, dropout):
        return nn.ModuleDict({
            'ln1': nn.LayerNorm(d_model),
            'ln2': nn.LayerNorm(d_model),
            'attn_qkv': nn.Linear(d_model, 3 * d_model),
            'attn_out': nn.Linear(d_model, d_model),
            'ffn': nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout)
            ),
            'num_heads': nn.Parameter(torch.tensor(num_heads), requires_grad=False),
            'dropout': nn.Dropout(dropout)
        })
    
    def _attention(self, x, block):
        B, T, C = x.shape
        num_heads = int(block['num_heads'].item())
        head_dim = C // num_heads
        
        qkv = block['attn_qkv'](x).view(B, T, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = block['dropout'](F.softmax(scores, dim=-1))
        
        out = (weights @ V).transpose(1, 2).reshape(B, T, C)
        return block['attn_out'](out)
    
    def forward(self, x, labels=None):
        B, T = x.shape
        
        # Embeddings
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = self.dropout(tok + pos)
        
        # Transformer blocks
        for block in self.blocks:
            h = h + self._attention(block['ln1'](h), block)
            h = h + block['ffn'](block['ln2'](h))
        
        # Classify using last token
        h = self.ln(h)
        logits = self.classifier(h[:, -1, :])
        
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return logits, loss
    
    def predict(self, x):
        """Get class predictions."""
        self.eval()
        with torch.no_grad():
            logits, _ = self(x)
            return logits.argmax(1)
    
    def predict_proba(self, x):
        """Get class probabilities."""
        self.eval()
        with torch.no_grad():
            logits, _ = self(x)
            return F.softmax(logits, dim=-1)


# ===========================================================================
# TRAINING AND EVALUATION
# ===========================================================================

def train(model, train_data, val_data, epochs=20, lr=1e-3, patience=5):
    """Train with early stopping."""
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_acc = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for x, y in train_loader:
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model.predict(x)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:2d} | Accuracy: {acc:.2%}")
        
        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_acc


def evaluate(model, data_loader, num_classes=2):
    """Comprehensive evaluation."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in data_loader:
            preds = model.predict(x)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
    
    # Per-class metrics
    class_metrics = {}
    for c in range(num_classes):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != c and l == c)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        class_metrics[c] = {'precision': prec, 'recall': rec, 'f1': f1}
    
    # Overall
    accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
    macro_f1 = sum(m['f1'] for m in class_metrics.values()) / num_classes
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class': class_metrics
    }


# ===========================================================================
# DATASETS
# ===========================================================================

# Sentiment data
SENTIMENT_POS = [
    "This movie was amazing!", "I loved every minute!",
    "Brilliant film, highly recommended!", "Outstanding performance!",
    "A masterpiece of cinema!", "Best movie I've seen!",
    "Wonderful story and acting!", "Absolutely fantastic!",
    "I was moved to tears!", "Incredible experience!",
]

SENTIMENT_NEG = [
    "Terrible movie, waste of time.", "I hated this film.",
    "Awful acting and plot.", "Worst movie ever made.",
    "Complete disaster.", "Boring and predictable.",
    "I want my money back.", "Dreadful from start to end.",
    "Painful to watch.", "Absolutely terrible!",
]

# Spam data
SPAM = [
    "You won $1000000! Click now!", "FREE iPhone! Claim prize!",
    "URGENT: Verify account immediately!", "Make money fast!",
    "Exclusive deal 90% off!", "Winner of lottery!",
    "Bank alert: Click to verify!", "Double your bitcoin!",
    "Hot singles near you!", "Claim inheritance now!",
]

HAM = [
    "Meeting at 3pm tomorrow.", "Thanks for your help!",
    "Happy birthday!", "See you for lunch?",
    "Great presentation today.", "Don't forget the meeting.",
    "How's the project going?", "Coffee tomorrow?",
    "Nice work on the report.", "Call me when you can.",
]

# Topic data
TOPICS = {
    'sports': ["Team wins championship", "Player breaks record", "Coach resigns", "Finals start next week"],
    'tech': ["New smartphone released", "AI startup funded", "Security bug fixed", "App launches feature"],
    'politics': ["President signs bill", "Senator debates reform", "Election results contested", "Summit scheduled"],
    'entertainment': ["Movie breaks records", "Star announces tour", "Show wins award", "Album released"],
}


# ===========================================================================
# MAIN PROJECT
# ===========================================================================

print("\n1. SENTIMENT CLASSIFIER")
print("-" * 40)

# Prepare data
sent_texts = SENTIMENT_POS + SENTIMENT_NEG
sent_labels = [1] * len(SENTIMENT_POS) + [0] * len(SENTIMENT_NEG)
random.seed(42)
combined = list(zip(sent_texts, sent_labels))
random.shuffle(combined)
sent_texts, sent_labels = zip(*combined)

split = int(0.8 * len(sent_texts))
sent_tokenizer = Tokenizer(sent_texts[:split])
sent_train = TextDataset(sent_texts[:split], sent_labels[:split], sent_tokenizer)
sent_test = TextDataset(sent_texts[split:], sent_labels[split:], sent_tokenizer)

sent_model = TransformerClassifier(
    vocab_size=sent_tokenizer.vocab_size,
    num_classes=2,
    max_length=20
)

print(f"  Vocabulary: {sent_tokenizer.vocab_size}")
print(f"  Train/Test: {len(sent_train)}/{len(sent_test)}")
print()

sent_model, _ = train(sent_model, sent_train, sent_test, epochs=30)
sent_metrics = evaluate(sent_model, DataLoader(sent_test, batch_size=8), num_classes=2)
print(f"\n  Accuracy: {sent_metrics['accuracy']:.2%}, F1: {sent_metrics['macro_f1']:.2%}")

pause()


print("2. SPAM CLASSIFIER")
print("-" * 40)

spam_texts = SPAM + HAM
spam_labels = [1] * len(SPAM) + [0] * len(HAM)
random.seed(43)
combined = list(zip(spam_texts, spam_labels))
random.shuffle(combined)
spam_texts, spam_labels = zip(*combined)

split = int(0.8 * len(spam_texts))
spam_tokenizer = Tokenizer(spam_texts[:split])
spam_train = TextDataset(spam_texts[:split], spam_labels[:split], spam_tokenizer)
spam_test = TextDataset(spam_texts[split:], spam_labels[split:], spam_tokenizer)

spam_model = TransformerClassifier(
    vocab_size=spam_tokenizer.vocab_size,
    num_classes=2,
    max_length=16
)

print(f"  Vocabulary: {spam_tokenizer.vocab_size}")
print(f"  Train/Test: {len(spam_train)}/{len(spam_test)}")
print()

spam_model, _ = train(spam_model, spam_train, spam_test, epochs=30)
spam_metrics = evaluate(spam_model, DataLoader(spam_test, batch_size=8), num_classes=2)
print(f"\n  Accuracy: {spam_metrics['accuracy']:.2%}, F1: {spam_metrics['macro_f1']:.2%}")

pause()


print("3. TOPIC CLASSIFIER (4 classes)")
print("-" * 40)

topic_texts = []
topic_labels = []
for idx, (topic, headlines) in enumerate(TOPICS.items()):
    topic_texts.extend(headlines)
    topic_labels.extend([idx] * len(headlines))

random.seed(44)
combined = list(zip(topic_texts, topic_labels))
random.shuffle(combined)
topic_texts, topic_labels = zip(*combined)

split = int(0.75 * len(topic_texts))
topic_tokenizer = Tokenizer(topic_texts[:split])
topic_train = TextDataset(topic_texts[:split], topic_labels[:split], topic_tokenizer, max_length=12)
topic_test = TextDataset(topic_texts[split:], topic_labels[split:], topic_tokenizer, max_length=12)

topic_model = TransformerClassifier(
    vocab_size=topic_tokenizer.vocab_size,
    num_classes=4,
    max_length=12
)

print(f"  Vocabulary: {topic_tokenizer.vocab_size}")
print(f"  Train/Test: {len(topic_train)}/{len(topic_test)}")
print(f"  Classes: Sports, Tech, Politics, Entertainment")
print()

topic_model, _ = train(topic_model, topic_train, topic_test, epochs=40)
topic_metrics = evaluate(topic_model, DataLoader(topic_test, batch_size=4), num_classes=4)
print(f"\n  Accuracy: {topic_metrics['accuracy']:.2%}, Macro F1: {topic_metrics['macro_f1']:.2%}")

pause()


print("4. UNIFIED PREDICTION INTERFACE")
print("-" * 40)


class UnifiedClassifier:
    """Unified interface for all classifiers."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.class_names = {}
    
    def add_model(self, name, model, tokenizer, class_names, max_length=32):
        self.models[name] = {'model': model, 'max_length': max_length}
        self.tokenizers[name] = tokenizer
        self.class_names[name] = class_names
    
    def predict(self, name, text):
        if name not in self.models:
            return None
        
        tokenizer = self.tokenizers[name]
        model = self.models[name]['model']
        max_len = self.models[name]['max_length']
        
        ids = torch.tensor([tokenizer.encode(text, max_len)])
        probs = model.predict_proba(ids)[0]
        pred = probs.argmax().item()
        
        return {
            'class': self.class_names[name][pred],
            'confidence': probs[pred].item(),
            'all_probs': {n: p.item() for n, p in zip(self.class_names[name], probs)}
        }


# Create unified classifier
classifier = UnifiedClassifier()
classifier.add_model('sentiment', sent_model, sent_tokenizer, ['Negative', 'Positive'], 20)
classifier.add_model('spam', spam_model, spam_tokenizer, ['Ham', 'Spam'], 16)
classifier.add_model('topic', topic_model, topic_tokenizer, 
                     ['Sports', 'Tech', 'Politics', 'Entertainment'], 12)

# Test predictions
test_texts = [
    ("sentiment", "This is absolutely wonderful!"),
    ("sentiment", "Terrible and boring."),
    ("spam", "You won a million dollars!"),
    ("spam", "Meeting tomorrow at 3pm."),
    ("topic", "Team wins the finals"),
    ("topic", "New software update released"),
]

print("\nPredictions:")
print("-" * 50)
for task, text in test_texts:
    result = classifier.predict(task, text)
    print(f"  [{task}] '{text[:30]}...'")
    print(f"    -> {result['class']} ({result['confidence']:.1%})")
    print()

pause()


# ===========================================================================
# SUMMARY
# ===========================================================================

print("=" * 60)
print("  PROJECT COMPLETE!")
print("=" * 60)

print(f"""
You built a complete classification system!

MODELS TRAINED:

1. Sentiment Classifier
   Accuracy: {sent_metrics['accuracy']:.2%}
   F1 Score: {sent_metrics['macro_f1']:.2%}

2. Spam Detector  
   Accuracy: {spam_metrics['accuracy']:.2%}
   F1 Score: {spam_metrics['macro_f1']:.2%}

3. Topic Classifier (4 classes)
   Accuracy: {topic_metrics['accuracy']:.2%}
   Macro F1: {topic_metrics['macro_f1']:.2%}

FEATURES:
  - Unified prediction interface
  - Confidence scores
  - Per-class metrics
  - Early stopping

This is production-ready classification code!

Next week: INSTRUCTION FINE-TUNING
We'll make the model follow commands like ChatGPT!
""")

print("=" * 60)
print("  End of Week 6 Project")
print("=" * 60)
