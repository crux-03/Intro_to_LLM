"""
Lesson 1: Introduction to Fine-Tuning
======================================

You've built GPT from scratch and understand pretraining.
Now we'll learn FINE-TUNING: adapting a pretrained model
for a specific task.

Fine-tuning is how most real-world LLM applications work:
    1. Start with pretrained model (GPT-2, LLaMA, etc.)
    2. Add task-specific layers if needed
    3. Train on your labeled data
    4. Deploy!

This week we'll fine-tune for CLASSIFICATION tasks:
    - Sentiment analysis (positive/negative)
    - Spam detection
    - Topic classification

Usage: python 01_finetuning_intro.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 1: Introduction to Fine-Tuning")
print("  Adapting Pretrained Models for Your Tasks")
print("=" * 60)

print("""
PRETRAINING vs FINE-TUNING

Pretraining taught the model general language understanding:
    - Grammar and syntax
    - Word meanings and relationships
    - World knowledge
    - Reasoning patterns

Fine-tuning teaches the model YOUR specific task:
    - "Is this review positive or negative?"
    - "Is this email spam?"
    - "What topic is this article about?"

The magic: Fine-tuning is FAST and needs LITTLE DATA!
Why? The model already understands language.
It just needs to learn to apply that understanding to your task.
""")

pause()


# ---------------------------------------------------------------------------
# Why Fine-Tuning Works
# ---------------------------------------------------------------------------

print("WHY FINE-TUNING WORKS")
print("-" * 40)

print("""
Think of it like education:

PRETRAINING = General education (K-12)
    - Learn to read and write
    - Understand the world
    - General reasoning skills
    - Takes YEARS, costs a LOT

FINE-TUNING = Specialized training
    - Already know how to read
    - Just learn the specific task
    - "When you see X, do Y"
    - Takes HOURS, costs LITTLE

A doctor doesn't learn medicine from scratch.
They build on years of general education!

Same with LLMs:
    - GPT-2 was pretrained for months on millions of documents
    - Fine-tuning takes hours on thousands of examples
    - The pretrained knowledge transfers to the new task
""")

pause()


print("TRANSFER LEARNING ILLUSTRATED")
print("-" * 40)

print("""
What the model learns at each stage:

PRETRAINED GPT (next token prediction):
    +-------------------------------------------+
    | Layer 12: High-level language features    |
    | Layer 11: Semantic understanding          |
    | Layer 10: Context integration             |
    |    ...                                    |
    | Layer 2:  Basic syntax patterns           |
    | Layer 1:  Character/word patterns         |
    | Embeddings: Token representations         |
    +-------------------------------------------+
    
FINE-TUNED FOR SENTIMENT:
    +-------------------------------------------+
    | Classification Head: positive/negative    |  <- NEW!
    | Layer 12: Sentiment-relevant features     |  <- Adjusted
    | Layer 11: Opinion/emotion detection       |  <- Adjusted
    | Layer 10: Context integration             |  <- Slightly adjusted
    |    ...                                    |
    | Layer 1:  Basic patterns                  |  <- Mostly unchanged
    | Embeddings: Token representations         |  <- Mostly unchanged
    +-------------------------------------------+

Lower layers stay similar (basic language).
Higher layers adapt to the new task!
""")

pause()


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

print("ADDING A CLASSIFICATION HEAD")
print("-" * 40)

print("""
For classification, we add a new layer on top of GPT:

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

Why use the LAST token?
    - GPT processes left-to-right
    - Last token has seen ALL previous tokens
    - Contains a "summary" of the whole sequence
    
(This is like [CLS] token in BERT, but simpler)
""")

pause()


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------

# Building blocks (from previous weeks)
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


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, None, max_seq_length, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# Classification wrapper
class GPTClassifier(nn.Module):
    """GPT with a classification head."""
    
    def __init__(self, gpt_model, num_classes, dropout=0.1):
        super().__init__()
        self.gpt = gpt_model
        d_model = gpt_model.token_embedding.weight.shape[1]
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, input_ids, labels=None):
        # Get hidden states (before lm_head)
        B, T = input_ids.shape
        tok_emb = self.gpt.token_embedding(input_ids)
        pos_emb = self.gpt.position_embedding(torch.arange(T, device=input_ids.device))
        x = self.gpt.dropout(tok_emb + pos_emb)
        for block in self.gpt.blocks:
            x = block(x)
        hidden_states = self.gpt.ln_final(x)
        
        # Use last token for classification
        last_hidden = hidden_states[:, -1, :]
        logits = self.classifier(last_hidden)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return logits, loss


print("BUILDING THE CLASSIFIER")
print("-" * 40)

# Create base GPT
gpt = GPT(
    vocab_size=1000,
    d_model=64,
    num_heads=4,
    num_layers=4,
    max_seq_length=128
)

# Wrap with classification head for sentiment (2 classes)
classifier = GPTClassifier(gpt, num_classes=2)

# Count parameters
total = sum(p.numel() for p in classifier.parameters())
gpt_params = sum(p.numel() for p in classifier.gpt.parameters())
head_params = sum(p.numel() for p in classifier.classifier.parameters())

print(f"GPT parameters: {gpt_params:,}")
print(f"Classifier head: {head_params:,}")
print(f"Total: {total:,}")
print()
print("The classifier head is tiny compared to GPT!")
print("Most parameters come from the pretrained backbone.")

pause()


# ---------------------------------------------------------------------------
# Fine-Tuning Strategies
# ---------------------------------------------------------------------------

print("FINE-TUNING STRATEGIES")
print("-" * 40)

print("""
How much of the model should we train?

STRATEGY 1: Train only classification head (Feature Extraction)
    - Freeze ALL GPT weights
    - Only train the new classifier
    - Fastest, needs least data
    - Good when: Little data, want fast results
    
STRATEGY 2: Train head + top layers
    - Freeze lower GPT layers
    - Fine-tune top few layers + classifier
    - Middle ground
    - Good when: Moderate data, better accuracy needed

STRATEGY 3: Train everything (Full Fine-Tuning)
    - Update ALL weights
    - Slowest, needs most data
    - Best accuracy (usually)
    - Good when: Lots of data, maximum performance needed
""")

pause()


def freeze_layers(model, strategy='head_only', num_unfrozen_layers=2):
    """Freeze model parameters based on strategy."""
    
    if strategy == 'head_only':
        # Freeze everything in GPT
        for param in model.gpt.parameters():
            param.requires_grad = False
    
    elif strategy == 'top_layers':
        # Freeze embeddings
        for param in model.gpt.token_embedding.parameters():
            param.requires_grad = False
        for param in model.gpt.position_embedding.parameters():
            param.requires_grad = False
        
        # Freeze bottom layers
        num_layers = len(model.gpt.blocks)
        for i, block in enumerate(model.gpt.blocks):
            if i < num_layers - num_unfrozen_layers:
                for param in block.parameters():
                    param.requires_grad = False
    
    elif strategy == 'full':
        # Everything trainable
        for param in model.parameters():
            param.requires_grad = True
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


print("COMPARING FREEZING STRATEGIES")
print("-" * 40)
print()

for strategy in ['head_only', 'top_layers', 'full']:
    # Reset model
    gpt = GPT(vocab_size=1000, d_model=64, num_heads=4, num_layers=4, max_seq_length=128)
    classifier = GPTClassifier(gpt, num_classes=2)
    
    # Apply strategy
    trainable, total = freeze_layers(classifier, strategy)
    
    print(f"{strategy.upper()}:")
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print()

pause()


# ---------------------------------------------------------------------------
# Testing the Classifier
# ---------------------------------------------------------------------------

print("TESTING THE CLASSIFIER")
print("-" * 40)

# Reset for testing
gpt = GPT(vocab_size=1000, d_model=64, num_heads=4, num_layers=4, max_seq_length=128)
classifier = GPTClassifier(gpt, num_classes=2)

# Test forward pass
batch_size = 4
seq_length = 20

input_ids = torch.randint(0, 1000, (batch_size, seq_length))
labels = torch.randint(0, 2, (batch_size,))

classifier.eval()
logits, loss = classifier(input_ids, labels)

print(f"Input shape: {input_ids.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss.item():.4f}")
print()

predictions = logits.argmax(dim=-1)
print(f"Predictions: {predictions.tolist()}")
print(f"True labels: {labels.tolist()}")
accuracy = (predictions == labels).float().mean()
print(f"Accuracy: {accuracy.item():.2%} (random, untrained)")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
1. WHY FINE-TUNING WORKS
   - Pretrained model already understands language
   - Just needs to learn the new task
   - Fast training, little data needed

2. CLASSIFICATION HEAD
   - Add Linear layer on top of GPT
   - Use last token's hidden state
   - Output: logits for each class

3. FREEZING STRATEGIES
   - Head only: Fastest, needs least data (~0.1% trainable)
   - Top layers: Balance speed and accuracy (~25% trainable)
   - Full: Best accuracy, needs most data (100% trainable)

4. TRANSFER LEARNING
   - Lower layers: General language features (keep frozen)
   - Higher layers: Task-specific features (fine-tune)
   - Gradual unfreezing can help

Next: Complete sentiment analysis example!
""")

print("=" * 60)
print("  End of Lesson 1")
print("=" * 60)
