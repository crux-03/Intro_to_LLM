"""
Lesson 2: Self-Attention
=========================

In basic attention, we used Q = K = V = embeddings directly.
Real transformers add LEARNABLE PROJECTIONS - matrices that
transform the embeddings into specialized Q, K, V representations.

This gives the model much more flexibility to learn what to
look for (Q), what to expose (K), and what to return (V).

Usage: python 02_self_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 2: Self-Attention")
print("  Learnable Query, Key, Value Projections")
print("=" * 60)


# ---------------------------------------------------------------------------
# Why Learnable Projections?
# ---------------------------------------------------------------------------

print("""
WHY LEARNABLE PROJECTIONS?

In basic attention: Q = K = V = embeddings

This is limited. Each word's query IS its embedding, and its
key IS its embedding. No flexibility!

Self-attention adds learnable projections:
    Q = embeddings @ W_q
    K = embeddings @ W_k
    V = embeddings @ W_v

Now the model can LEARN:
    - W_q: What features to look for (queries)
    - W_k: What features to expose for matching (keys)
    - W_v: What features to return (values)

Each projection specializes for its role.
""")

pause()


print("PROJECTIONS AS FEATURE EXTRACTORS")
print("-" * 40)

print("""
Think of projections as feature extractors:

    Original embedding: [general word representation]
                               |
            +------------------+------------------+
            |                  |                  |
            v                  v                  v
        W_q extracts      W_k extracts       W_v extracts
        "what to seek"    "how to match"     "what to return"
            |                  |                  |
            v                  v                  v
            Q                  K                  V

Example for the word "cat":
    - Q might encode: "I'm looking for verbs or adjectives"
    - K might encode: "I'm a noun, I'm an animal, I'm the subject"
    - V might encode: "Here's my semantic content"

Different projections let each role specialize.
""")

pause()


# ---------------------------------------------------------------------------
# Implementing Projections
# ---------------------------------------------------------------------------

print("IMPLEMENTING PROJECTIONS")
print("-" * 40)

d_model = 64

# Create projection layers
W_q = nn.Linear(d_model, d_model)
W_k = nn.Linear(d_model, d_model)
W_v = nn.Linear(d_model, d_model)

print(f"Projection layers created for d_model={d_model}:")
print(f"  W_q: {W_q.weight.shape} + bias {W_q.bias.shape}")
print(f"  W_k: {W_k.weight.shape} + bias {W_k.bias.shape}")
print(f"  W_v: {W_v.weight.shape} + bias {W_v.bias.shape}")
print()

# Count parameters
total_params = sum(p.numel() for layer in [W_q, W_k, W_v] for p in layer.parameters())
print(f"Total projection parameters: {total_params:,}")

pause()


print("APPLYING PROJECTIONS")
print("-" * 40)

# Sample input
batch_size = 2
seq_length = 4
x = torch.randn(batch_size, seq_length, d_model)

print(f"Input shape: {x.shape}")
print()

# Project to Q, K, V
Q = W_q(x)
K = W_k(x)
V = W_v(x)

print("After projection:")
print(f"  Q shape: {Q.shape}")
print(f"  K shape: {K.shape}")
print(f"  V shape: {V.shape}")

pause()


# ---------------------------------------------------------------------------
# Self-Attention Module
# ---------------------------------------------------------------------------

print("COMPLETE SELF-ATTENTION MODULE")
print("-" * 40)


class SelfAttention(nn.Module):
    """Self-attention with learnable Q, K, V projections."""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Learnable projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_weights=False):
        """
        Args:
            x: (batch_size, seq_length, d_model)
            return_weights: If True, also return attention weights
        
        Returns:
            output: (batch_size, seq_length, d_model)
            weights: (optional) attention weights
        """
        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        
        # Softmax to get weights
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Apply attention to values
        attended = weights @ V
        
        # Output projection
        output = self.W_o(attended)
        
        if return_weights:
            return output, weights
        return output


# Create and test
attention = SelfAttention(d_model=64)

print("SelfAttention module created")
print()
print("Components:")
for name, param in attention.named_parameters():
    print(f"  {name}: {param.shape}")

pause()


print("TESTING SELF-ATTENTION")
print("-" * 40)

# Test forward pass
batch_size = 2
seq_length = 6
d_model = 64

x = torch.randn(batch_size, seq_length, d_model)
attention.eval()

output, weights = attention(x, return_weights=True)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Weights shape: {weights.shape}")
print()
print("Input and output have the same shape!")
print("This is important - attention can be stacked in layers.")

pause()


# ---------------------------------------------------------------------------
# The Output Projection
# ---------------------------------------------------------------------------

print("THE OUTPUT PROJECTION (W_o)")
print("-" * 40)

print("""
You might wonder: why do we need W_o at the end?

After attention, we have a weighted combination of values.
The output projection serves several purposes:

1. MIXING INFORMATION
   Combines information from the attended values
   
2. DIMENSIONALITY
   Can change output dimension (though usually same as input)
   
3. ADDITIONAL CAPACITY
   More learnable parameters = more expressive model

4. SYMMETRY
   Input projection (W_q, W_k, W_v) → Attention → Output projection (W_o)

For multi-head attention (next lesson), W_o becomes crucial
for combining information from different heads.
""")

pause()


# ---------------------------------------------------------------------------
# Visualizing Self-Attention
# ---------------------------------------------------------------------------

print("VISUALIZING SELF-ATTENTION")
print("-" * 40)

# Create fresh attention and input
attention_vis = SelfAttention(d_model=32)
attention_vis.eval()

words = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(words)
x = torch.randn(1, seq_len, 32)

# Get attention weights
with torch.no_grad():
    _, weights = attention_vis(x, return_weights=True)
weights = weights[0]  # Remove batch dimension

print("Attention weights (randomly initialized):")
print("-" * 50)
print(f"{'':8}", end="")
for w in words:
    print(f"{w:7}", end="")
print()
print("-" * 50)

for i, word in enumerate(words):
    print(f"{word:8}", end="")
    for j in range(seq_len):
        print(f"{weights[i, j].item():.3f}  ", end="")
    print()

print()
print("These weights are random since the model is untrained.")
print("After training, meaningful patterns emerge!")

pause()


# ---------------------------------------------------------------------------
# Dropout in Attention
# ---------------------------------------------------------------------------

print("DROPOUT IN ATTENTION")
print("-" * 40)

print("""
We apply dropout to attention weights during training.
This helps prevent overfitting by:
    - Randomly zeroing some attention connections
    - Forcing the model to not rely too heavily on any one connection

During evaluation (model.eval()), dropout is disabled.
""")

# Demonstrate
attention_drop = SelfAttention(d_model=64, dropout=0.3)
x = torch.randn(1, 4, 64)

# Training mode
attention_drop.train()
out1 = attention_drop(x)

# Same input again (different due to dropout)
out2 = attention_drop(x)

# Eval mode (no dropout)
attention_drop.eval()
out3 = attention_drop(x)
out4 = attention_drop(x)

print(f"Training mode - same input, same output? {torch.equal(out1, out2)}")
print(f"Eval mode - same input, same output? {torch.equal(out3, out4)}")

pause()


# ---------------------------------------------------------------------------
# Parameter Count
# ---------------------------------------------------------------------------

print("PARAMETER COUNT")
print("-" * 40)

def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

# Different sizes
for d in [64, 256, 512, 768]:
    attn = SelfAttention(d_model=d)
    params = count_parameters(attn)
    print(f"d_model={d:4d}: {params:,} parameters")

print()
print("Parameters scale as O(d_model²)")
print("4 projection matrices, each d_model × d_model")

pause()


# ---------------------------------------------------------------------------
# Comparison: Basic vs Self-Attention
# ---------------------------------------------------------------------------

print("BASIC ATTENTION vs SELF-ATTENTION")
print("-" * 40)

print("""
BASIC ATTENTION (Lesson 1):
    Q = embeddings
    K = embeddings
    V = embeddings
    
    - No learnable parameters in attention itself
    - Q, K, V are identical
    - Limited flexibility

SELF-ATTENTION (This Lesson):
    Q = embeddings @ W_q
    K = embeddings @ W_k
    V = embeddings @ W_v
    output = attended @ W_o
    
    - 4 learnable projection matrices
    - Q, K, V can specialize for their roles
    - Much more expressive

This is what real transformers use!
""")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
1. LEARNABLE PROJECTIONS give each role its own transformation:
   - Q = x @ W_q  (what to look for)
   - K = x @ W_k  (what to expose for matching)
   - V = x @ W_v  (what information to return)

2. OUTPUT PROJECTION (W_o) combines attended information

3. SELF-ATTENTION MODULE:
   - Input: (batch, seq_len, d_model)
   - Output: (batch, seq_len, d_model)
   - Same shape in and out!

4. DROPOUT on attention weights prevents overfitting

5. PARAMETER COUNT: 4 × d_model × d_model
   (plus biases)

Next: Causal masking for GPT-style models!
""")

print("=" * 60)
print("  End of Lesson 2")
print("=" * 60)
