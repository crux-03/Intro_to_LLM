"""
Lesson 4: Multi-Head Attention
===============================

The final piece of the attention puzzle!

Instead of one attention pattern, we run multiple "heads" in parallel.
Each head can learn to focus on different things:
    - One head might learn syntax
    - Another might learn semantics
    - Another might learn position relationships

This is exactly what GPT uses.

Usage: python 04_multihead.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 4: Multi-Head Attention")
print("  The Complete Attention Mechanism")
print("=" * 60)


# ---------------------------------------------------------------------------
# Why Multiple Heads?
# ---------------------------------------------------------------------------

print("""
WHY MULTIPLE HEADS?

Consider understanding a sentence:
    "The cat sat on the mat because it was tired."

Multiple relationships matter simultaneously:
    - SYNTACTIC: "sat" relates to "cat" (subject-verb)
    - REFERENTIAL: "it" refers to "cat" (pronoun resolution)
    - SEMANTIC: "tired" relates to "sat" (cause-effect)

A single attention head might capture ONE of these.
Multiple heads can capture ALL of them at once!

Each head has its own Q, K, V projections.
Each can specialize for different patterns.
""")

pause()


# ---------------------------------------------------------------------------
# The Multi-Head Architecture
# ---------------------------------------------------------------------------

print("THE MULTI-HEAD ARCHITECTURE")
print("-" * 40)

print("""
How multi-head attention works:

    Input: (batch, seq_len, d_model)  e.g., (32, 128, 512)
    
    1. PROJECT and split into heads:
       d_model splits into num_heads pieces
       d_model=512, num_heads=8 → head_dim=64
       
    2. ATTENTION per head:
       Each head does attention independently
       8 heads running in parallel
       
    3. CONCATENATE heads:
       Combine all head outputs
       Back to (batch, seq_len, d_model)
       
    4. OUTPUT projection:
       Final linear layer

Same total computation, but multiple perspectives!
""")

pause()


# ---------------------------------------------------------------------------
# The Reshape Trick
# ---------------------------------------------------------------------------

print("THE RESHAPE TRICK")
print("-" * 40)

print("""
We could create separate projections for each head...
But there's a more efficient way!

Use ONE big projection, then RESHAPE into heads:
    1. Project: (batch, seq, d_model) → (batch, seq, d_model)
    2. Reshape: (batch, seq, d_model) → (batch, seq, num_heads, head_dim)
    3. Transpose: → (batch, num_heads, seq, head_dim)
    
Now each head can do attention independently!
""")

pause()

# Demonstrate
batch_size = 2
seq_length = 4
d_model = 64
num_heads = 4
head_dim = d_model // num_heads

print(f"d_model: {d_model}")
print(f"num_heads: {num_heads}")
print(f"head_dim: {head_dim}")
print()

# Start with projected Q
Q = torch.randn(batch_size, seq_length, d_model)
print(f"After projection, Q shape: {Q.shape}")

# Reshape to separate heads
Q = Q.view(batch_size, seq_length, num_heads, head_dim)
print(f"After reshape: {Q.shape}")

# Transpose to (batch, heads, seq, head_dim)
Q = Q.transpose(1, 2)
print(f"After transpose: {Q.shape}")

print()
print("Now each head can do attention independently!")

pause()


# ---------------------------------------------------------------------------
# Multi-Head Attention Module
# ---------------------------------------------------------------------------

print("MULTI-HEAD CAUSAL ATTENTION MODULE")
print("-" * 40)


class MultiHeadCausalAttention(nn.Module):
    """Multi-head causal self-attention for GPT."""
    
    def __init__(self, d_model, num_heads, max_seq_length=512, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Combined Q, K, V projection (more efficient than separate)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Causal mask
        mask = torch.tril(torch.ones(max_seq_length, max_seq_length))
        self.register_buffer('mask', mask)
    
    def forward(self, x, return_weights=False):
        batch_size, seq_length, _ = x.shape
        
        # Project Q, K, V in one shot
        qkv = self.qkv_proj(x)  # (batch, seq, 3 * d_model)
        
        # Reshape and split into Q, K, V
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        
        # Causal mask
        mask = self.mask[:seq_length, :seq_length]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        
        # Apply attention
        attended = weights @ V  # (batch, heads, seq, head_dim)
        
        # Concatenate heads
        attended = attended.transpose(1, 2)  # (batch, seq, heads, head_dim)
        attended = attended.reshape(batch_size, seq_length, self.d_model)
        
        # Output projection
        output = self.out_proj(attended)
        output = self.output_dropout(output)
        
        if return_weights:
            return output, weights
        return output


print("MultiHeadCausalAttention module defined")

pause()


# ---------------------------------------------------------------------------
# Testing the Module
# ---------------------------------------------------------------------------

print("TESTING MULTI-HEAD ATTENTION")
print("-" * 40)

# Create module
mha = MultiHeadCausalAttention(
    d_model=64,
    num_heads=4,
    max_seq_length=128
)
mha.eval()

# Test input
x = torch.randn(2, 8, 64)

output, weights = mha(x, return_weights=True)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Weights shape: {weights.shape}")
print()
print(f"Note: weights has shape (batch, num_heads, seq, seq)")
print(f"Each head has its own attention pattern!")

pause()


# ---------------------------------------------------------------------------
# Visualizing Multiple Heads
# ---------------------------------------------------------------------------

print("VISUALIZING ATTENTION PER HEAD")
print("-" * 40)

# Create module with 4 heads
mha_vis = MultiHeadCausalAttention(d_model=32, num_heads=4, max_seq_length=32)
mha_vis.eval()

words = ["The", "cat", "sat", "down"]
x = torch.randn(1, len(words), 32)

with torch.no_grad():
    _, weights = mha_vis(x, return_weights=True)
weights = weights[0]  # Remove batch dim

print(f"Each head learns different patterns:")
print()

for head_idx in range(4):
    print(f"--- Head {head_idx} ---")
    head_weights = weights[head_idx]
    
    print(f"{'':8}", end="")
    for w in words:
        print(f"{w:8}", end="")
    print()
    
    for i, word in enumerate(words):
        print(f"{word:8}", end="")
        for j in range(len(words)):
            if j <= i:  # Causal mask
                print(f"{head_weights[i,j].item():.3f}   ", end="")
            else:
                print(f"{'---':8}", end="")
        print()
    print()

print("These patterns are random (untrained).")
print("After training, different heads specialize for different tasks!")

pause()


# ---------------------------------------------------------------------------
# GPT-2 Configuration
# ---------------------------------------------------------------------------

print("GPT-2 ATTENTION CONFIGURATION")
print("-" * 40)

print("""
GPT-2 attention specs:

Model         d_model    num_heads    head_dim
---------     -------    ---------    --------
GPT-2 Small      768          12          64
GPT-2 Medium    1024          16          64
GPT-2 Large     1280          20          64
GPT-2 XL        1600          25          64

Notice: head_dim is always 64!
Larger models add more heads, not bigger heads.
""")

pause()

# Create GPT-2 Small attention
gpt2_attn = MultiHeadCausalAttention(
    d_model=768,
    num_heads=12,
    max_seq_length=1024
)

params = sum(p.numel() for p in gpt2_attn.parameters())

print("GPT-2 Small Attention Layer:")
print(f"  d_model: 768")
print(f"  num_heads: 12")
print(f"  head_dim: {768 // 12}")
print(f"  Parameters: {params:,}")

pause()


# ---------------------------------------------------------------------------
# The Combined QKV Projection
# ---------------------------------------------------------------------------

print("THE COMBINED QKV PROJECTION")
print("-" * 40)

print("""
We used: self.qkv_proj = nn.Linear(d_model, 3 * d_model)

Instead of separate W_q, W_k, W_v projections.

Why? Efficiency!

Separate projections:
    Q = W_q(x)  # one matmul
    K = W_k(x)  # another matmul
    V = W_v(x)  # another matmul
    Total: 3 matrix multiplications

Combined projection:
    QKV = W_qkv(x)  # one big matmul
    Q, K, V = split(QKV)  # just reshaping
    Total: 1 matrix multiplication

Same result, but faster!
GPUs prefer fewer, larger operations.
""")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  WEEK 3 COMPLETE!")
print("=" * 60)

print("""
Congratulations! You've mastered attention!

LESSON 1 - Basic Attention:
    Attention(Q,K,V) = softmax(QK^T/√d) × V
    Soft lookup mechanism

LESSON 2 - Self-Attention:
    Learnable Q, K, V projections
    Output projection W_o

LESSON 3 - Causal Attention:
    Lower triangular mask
    Prevents seeing future tokens

LESSON 4 - Multi-Head Attention:
    Multiple attention heads in parallel
    Each head learns different patterns
    Efficient combined QKV projection

THIS IS THE HEART OF TRANSFORMERS!

What you built is exactly what runs in GPT, BERT,
and every modern language model.

Next week: We'll add Feed-Forward Networks and Layer
Normalization to build the complete GPT architecture!
""")

print("=" * 60)
print("  End of Week 3")
print("=" * 60)
