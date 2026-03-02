"""
Lesson 3: Spam Detection and Multi-class Classification
=========================================================

We'll build two classifiers:
    1. Spam Detector (binary classification)
    2. Topic Classifier (multi-class classification)

You'll apply everything you've learned about fine-tuning
and see how the same approach works for different tasks.

By the end, you'll have practical experience with:
    - Binary classification (spam/not spam)
    - Multi-class classification (multiple categories)
    - Evaluation for different classification types

Usage: python 03_spam_multiclass.py
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
print("  Lesson 3: Spam Detection & Multi-class Classification")
print("=" * 60)

print("""
You'll build two classifiers:

PART A: SPAM DETECTION (Binary)
    Input: Email/message text
    Output: Spam or Not Spam (Ham)
    
PART B: TOPIC CLASSIFICATION (Multi-class)
    Input: News headline
    Output: Sports, Tech, Politics, or Entertainment

These are extremely practical tasks used in production systems!
""")

pause()


# ---------------------------------------------------------------------------
# Common Components
# ---------------------------------------------------------------------------

class SimpleTokenizer:
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
    def __init__(self, texts, labels, tokenizer, max_length=32):
        self.encodings = [tokenizer.encode(t, max_length) for t in texts]
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx]), torch.tensor(self.labels[idx])


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_length=64, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_length, max_seq_length)))
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return self.out((weights @ V).transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_length=64, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads, max_seq_length, dropout)
        self.ffn = FeedForward(d_model, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class Classifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes, 
                 max_seq_length=64, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, max_seq_length, dropout) 
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
    
    def forward(self, x, labels=None):
        B, T = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = self.dropout(tok + pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln(h)
        logits = self.head(h[:, -1, :])
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return logits, loss


def train_model(model, train_data, test_data, epochs=15, lr=1e-3, verbose=True):
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if verbose and (epoch + 1) % 5 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    logits, _ = model(x)
                    correct += (logits.argmax(1) == y).sum().item()
                    total += y.size(0)
            print(f"    Epoch {epoch+1:2d} | Test Accuracy: {correct/total:.2%}")
    
    return model


# ===========================================================================
# PART A: SPAM DETECTION
# ===========================================================================

print("=" * 60)
print("  PART A: SPAM DETECTION")
print("=" * 60)

print("""
Task: Classify messages as Spam or Ham (not spam).

Spam characteristics:
    - Urgent calls to action
    - Money/prize offers
    - Suspicious links/requests

Ham characteristics:
    - Normal conversations
    - Legitimate requests
    - No suspicious patterns
""")

pause()


SPAM_MESSAGES = [
    "Congratulations! You've won $1000000! Click here to claim now!",
    "URGENT: Your account will be suspended. Verify immediately!",
    "FREE iPhone! Just complete this survey to claim your prize!",
    "You have been selected for a cash prize of $5000!",
    "Act now! Limited time offer - 90% discount!",
    "Your package is waiting. Pay $1.99 shipping to receive.",
    "Hot singles in your area want to meet you!",
    "Make $500 daily working from home! No experience needed!",
    "WINNER! You've been chosen for our exclusive lottery!",
    "Bank alert: Suspicious activity. Click to verify your account.",
    "Claim your free gift card worth $500 now!",
    "Congratulations! You qualify for a $50000 loan!",
    "Double your bitcoin in 24 hours! Guaranteed returns!",
    "Your computer has a virus! Call now for support!",
    "Exclusive deal just for you! 80% off everything!",
    "You have 24 hours to claim your inheritance of $2M!",
    "Free vacation getaway! Just pay taxes and fees!",
    "Your email won our daily prize draw! Claim now!",
    "Alert: Your Netflix payment failed. Update billing info.",
    "Make money fast with this one weird trick!",
]

HAM_MESSAGES = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Can you send me the report when you get a chance?",
    "Thanks for your help with the project yesterday.",
    "The meeting has been rescheduled to 3pm.",
    "Happy birthday! Hope you have a great day!",
    "Just checking in - how's the new job going?",
    "Don't forget to pick up milk on your way home.",
    "The kids have soccer practice at 5pm today.",
    "Great presentation today! Really well done.",
    "Can we reschedule our call to next week?",
    "I'll be working from home tomorrow.",
    "Thanks for dinner last night! We had a great time.",
    "The document you requested is attached.",
    "Let me know if you need any help with the move.",
    "Coffee on Thursday sounds good. See you then!",
    "Running a bit late, be there in 10 minutes.",
    "How did your interview go yesterday?",
    "The weather looks nice for the weekend.",
    "Did you see the game last night? Amazing finish!",
    "Your order has shipped and will arrive Monday.",
]


# Prepare data
spam_texts = SPAM_MESSAGES + HAM_MESSAGES
spam_labels = [1] * len(SPAM_MESSAGES) + [0] * len(HAM_MESSAGES)

random.seed(42)
combined = list(zip(spam_texts, spam_labels))
random.shuffle(combined)
spam_texts, spam_labels = zip(*combined)

split = int(0.8 * len(spam_texts))
train_texts, test_texts = spam_texts[:split], spam_texts[split:]
train_labels, test_labels = spam_labels[:split], spam_labels[split:]

print(f"Spam Detection Dataset:")
print(f"  Training: {len(train_texts)} samples")
print(f"  Test: {len(test_texts)} samples")
print(f"  Classes: 0=Ham, 1=Spam")

pause()


print("TRAINING SPAM DETECTOR")
print("-" * 40)

spam_tokenizer = SimpleTokenizer(train_texts, max_vocab=500)
spam_train = TextDataset(train_texts, train_labels, spam_tokenizer, max_length=32)
spam_test = TextDataset(test_texts, test_labels, spam_tokenizer, max_length=32)

spam_model = Classifier(
    vocab_size=spam_tokenizer.vocab_size,
    d_model=64,
    num_heads=4,
    num_layers=2,
    num_classes=2,
    max_seq_length=32
)

print("  Training spam detector...")
spam_model = train_model(spam_model, spam_train, spam_test)

pause()


print("SPAM DETECTION RESULTS")
print("-" * 40)


def evaluate_binary(model, data_loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            logits, _ = model(x)
            preds.extend(logits.argmax(1).tolist())
            labels.extend(y.tolist())
    
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}


spam_test_loader = DataLoader(spam_test, batch_size=8)
metrics = evaluate_binary(spam_model, spam_test_loader)

print("Spam Detection Metrics:")
print(f"  Accuracy:  {metrics['acc']:.2%}")
print(f"  Precision: {metrics['prec']:.2%}")
print(f"  Recall:    {metrics['rec']:.2%}")
print(f"  F1 Score:  {metrics['f1']:.2%}")

pause()


print("TESTING SPAM DETECTOR")
print("-" * 40)


def predict_spam(model, tokenizer, text):
    model.eval()
    ids = torch.tensor([tokenizer.encode(text, max_length=32)])
    with torch.no_grad():
        logits, _ = model(ids)
        pred = logits.argmax(1).item()
    return "SPAM" if pred == 1 else "Ham"


test_messages = [
    "FREE MONEY! Claim your $10000 prize now!",
    "Hey, want to grab coffee tomorrow?",
    "Your account has been compromised! Click here!",
    "Meeting moved to 2pm. See you there.",
]

print("\nTesting on new messages:")
for msg in test_messages:
    result = predict_spam(spam_model, spam_tokenizer, msg)
    print(f"  '{msg[:40]}...' -> {result}")

pause()


# ===========================================================================
# PART B: MULTI-CLASS TOPIC CLASSIFICATION
# ===========================================================================

print("=" * 60)
print("  PART B: MULTI-CLASS TOPIC CLASSIFICATION")
print("=" * 60)

print("""
Task: Classify news headlines into 4 categories.

Categories:
    0 = Sports
    1 = Technology
    2 = Politics
    3 = Entertainment
""")

pause()


TOPICS_DATA = {
    'sports': [
        "Lakers win championship in overtime thriller",
        "World Cup final breaks viewership records",
        "Tennis star announces retirement after injury",
        "Olympic swimmer breaks world record",
        "Football team signs star quarterback",
        "Baseball playoffs begin next week",
        "Golf tournament postponed due to weather",
        "Basketball coach wins coach of the year",
        "Soccer team advances to finals",
        "Hockey player traded to rival team",
    ],
    'tech': [
        "New smartphone features revolutionary camera",
        "AI startup raises billion dollar funding",
        "Tech giant announces virtual reality headset",
        "Software update fixes critical security bug",
        "Electric car company unveils new model",
        "Social media platform changes privacy policy",
        "Quantum computer achieves new milestone",
        "Streaming service launches new features",
        "Robot learns to perform surgery",
        "Cryptocurrency reaches all time high",
    ],
    'politics': [
        "President signs new infrastructure bill",
        "Senate debates healthcare reform",
        "Governor announces reelection campaign",
        "International summit addresses climate change",
        "Congress passes budget resolution",
        "Mayor unveils new transportation plan",
        "Election results contested in court",
        "Diplomats meet to discuss trade deal",
        "New legislation targets tax reform",
        "Political party announces new leadership",
    ],
    'entertainment': [
        "Blockbuster movie breaks box office records",
        "Pop star announces world tour dates",
        "Award show reveals this year nominees",
        "Netflix releases highly anticipated series",
        "Celebrity couple announces engagement",
        "Music festival lineup announced",
        "Director wins lifetime achievement award",
        "Broadway show extends run due to demand",
        "Video game sequel announced for next year",
        "Famous actor joins superhero franchise",
    ],
}


# Prepare data
topic_texts = []
topic_labels = []
for idx, (topic, headlines) in enumerate(TOPICS_DATA.items()):
    topic_texts.extend(headlines)
    topic_labels.extend([idx] * len(headlines))

random.seed(123)
combined = list(zip(topic_texts, topic_labels))
random.shuffle(combined)
topic_texts, topic_labels = zip(*combined)

split = int(0.8 * len(topic_texts))
train_t, test_t = topic_texts[:split], topic_texts[split:]
train_l, test_l = topic_labels[:split], topic_labels[split:]

print(f"Topic Classification Dataset:")
print(f"  Training: {len(train_t)} samples")
print(f"  Test: {len(test_t)} samples")
print(f"  Classes: Sports(0), Tech(1), Politics(2), Entertainment(3)")

pause()


print("TRAINING TOPIC CLASSIFIER")
print("-" * 40)

topic_tokenizer = SimpleTokenizer(train_t, max_vocab=500)
topic_train = TextDataset(train_t, train_l, topic_tokenizer, max_length=24)
topic_test = TextDataset(test_t, test_l, topic_tokenizer, max_length=24)

topic_model = Classifier(
    vocab_size=topic_tokenizer.vocab_size,
    d_model=64,
    num_heads=4,
    num_layers=2,
    num_classes=4,  # 4 topics!
    max_seq_length=24
)

print("  Training topic classifier...")
topic_model = train_model(topic_model, topic_train, topic_test, epochs=20)

pause()


print("MULTI-CLASS EVALUATION")
print("-" * 40)

print("""
For multi-class, we compute metrics PER CLASS:
    - Precision, Recall, F1 for each class
    - Macro average (average across classes)
    - Overall accuracy
""")

pause()


def evaluate_multiclass(model, data_loader, num_classes):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            logits, _ = model(x)
            preds.extend(logits.argmax(1).tolist())
            labels.extend(y.tolist())
    
    # Per-class metrics
    results = {}
    for c in range(num_classes):
        tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        results[c] = {'prec': prec, 'rec': rec, 'f1': f1}
    
    # Overall accuracy
    acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
    
    # Macro averages
    macro_prec = sum(r['prec'] for r in results.values()) / num_classes
    macro_rec = sum(r['rec'] for r in results.values()) / num_classes
    macro_f1 = sum(r['f1'] for r in results.values()) / num_classes
    
    return results, acc, macro_prec, macro_rec, macro_f1


topic_test_loader = DataLoader(topic_test, batch_size=8)
class_results, acc, m_prec, m_rec, m_f1 = evaluate_multiclass(topic_model, topic_test_loader, 4)

class_names = ['Sports', 'Tech', 'Politics', 'Entertainment']

print("Per-class metrics:")
print("-" * 50)
print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 50)
for idx, name in enumerate(class_names):
    r = class_results[idx]
    print(f"{name:<15} {r['prec']:>10.2%} {r['rec']:>10.2%} {r['f1']:>10.2%}")
print("-" * 50)
print(f"{'Macro Avg':<15} {m_prec:>10.2%} {m_rec:>10.2%} {m_f1:>10.2%}")
print(f"\nOverall Accuracy: {acc:.2%}")

pause()


print("TESTING TOPIC CLASSIFIER")
print("-" * 40)


def predict_topic(model, tokenizer, text):
    model.eval()
    ids = torch.tensor([tokenizer.encode(text, max_length=24)])
    with torch.no_grad():
        logits, _ = model(ids)
        pred = logits.argmax(1).item()
    return class_names[pred]


test_headlines = [
    "Team wins championship game",
    "New app launches with AI features",
    "Senator proposes new bill",
    "Album tops music charts",
]

print("\nTesting on new headlines:")
for headline in test_headlines:
    result = predict_topic(topic_model, topic_tokenizer, headline)
    print(f"  '{headline}' -> {result}")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  WEEK 6 COMPLETE!")
print("=" * 60)

print("""
Congratulations! You've mastered fine-tuning for classification!

WHAT YOU LEARNED:

1. FINE-TUNING BASICS
   - Why transfer learning works
   - Adding classification heads
   - Freezing strategies

2. BINARY CLASSIFICATION
   - Sentiment analysis (positive/negative)
   - Spam detection (spam/ham)
   - Precision, recall, F1 metrics

3. MULTI-CLASS CLASSIFICATION
   - Topic classification (4 categories)
   - Per-class metrics
   - Macro averaging

4. PRACTICAL SKILLS
   - Dataset preparation
   - Training and evaluation
   - Making predictions

NEXT WEEK: Instruction Fine-Tuning!
We'll make the model follow commands like ChatGPT!
""")

print("=" * 60)
print("  End of Week 6")
print("=" * 60)
