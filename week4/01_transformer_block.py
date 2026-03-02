"""
Lesson 1: The Transformer Block
================================

The Transformer block is the repeating unit that makes up GPT.
GPT-2 Small stacks 12 of these blocks. GPT-3 stacks 96!

Each block contains:
    1. Multi-head attention (from Week 3)
    2. Feed-forward network
    3. Layer normalization
    4. Residual connections

This lesson builds each component and combines them.

Usage: python 01_transformer_block.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 1: The Transformer Block")
print("  The Building Block of GPT")
print("=" * 60)


# ---------------------------------------------------------------------------
# Multi-Head Attention (from Week 3)
# ---------------------------------------------------------------------------

class MultiHeadCausalAttention(nn.Module):
    """Multi-head causal attention from Week 3."""
    
    def __init__(self, d_model, num_heads, max_seq_length=512, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        mask = torch.tril(torch.ones(max_seq_length, max_seq_length))
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        
        out = (weights @ V).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out_proj(out))


# ---------------------------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------------------------

print("""
LAYER NORMALIZATION

LayerNorm stabilizes training by normalizing activations.

For each sample, across all features:
    1. Compute mean and variance
    2. Normalize to mean=0, variance=1
    3. Scale and shift with learned parameters (gamma, beta)

Why do we need this?
    - Deep networks have unstable gradients
    - Activations can explode or vanish
    - LayerNorm keeps values in a reasonable range
""")

pause()

# Demonstrate LayerNorm
d_model = 64
layer_norm = nn.LayerNorm(d_model)

# Create input with unusual statistics
x = torch.randn(2, 4, d_model) * 10 + 5  # Mean ~5, Std ~10

print("LayerNorm in action:")
print(f"  Before - Mean: {x.mean().item():.2f}, Std: {x.std().item():.2f}")

normalized = layer_norm(x)
print(f"  After  - Mean: {normalized.mean().item():.4f}, Std: {normalized.std().item():.2f}")
print()
print("LayerNorm has learnable parameters:")
print(f"  Weight (gamma): {layer_norm.weight.shape} - scale factor")
print(f"  Bias (beta): {layer_norm.bias.shape} - shift factor")
print(f"  Total: {sum(p.numel() for p in layer_norm.parameters())} parameters")

pause()


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

print("FEED-FORWARD NETWORK (FFN)")
print("-" * 40)

print("""
The FFN processes each position independently:

    Linear(d_model → 4*d_model)  # Expand
    GELU activation              # Non-linearity
    Linear(4*d_model → d_model)  # Contract back

Why expand then contract?
    - Expansion creates capacity for complex transformations
    - GELU enables non-linear processing
    - Contraction returns to original dimension

Think of it this way:
    - Attention mixes information BETWEEN positions
    - FFN processes EACH position deeply
""")

pause()


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model  # Standard expansion factor
        
        self.fc1 = nn.Linear(d_model, d_ff)    # Expand
        self.fc2 = nn.Linear(d_ff, d_model)    # Contract
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)           # Smooth activation
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Test FFN
ffn = FeedForward(d_model=64)
ffn.eval()

print("Testing Feed-Forward Network:")
print(f"  fc1: 64 → 256 ({64 * 256 + 256:,} parameters)")
print(f"  fc2: 256 → 64 ({256 * 64 + 64:,} parameters)")
print(f"  Total: {sum(p.numel() for p in ffn.parameters()):,} parameters")
print()

x = torch.randn(2, 10, 64)
output = ffn(x)
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {output.shape}")

pause()


print("WHY GELU INSTEAD OF RELU?")
print("-" * 40)

print("""
ReLU: max(0, x)
    - Sharp cutoff at 0
    - Derivative is 0 for x < 0 ("dead neurons")

GELU: x * Φ(x) where Φ is the Gaussian CDF
    - Smooth, continuous everywhere
    - Small negative values pass through (attenuated)
    - Better gradient flow

GELU is slightly more expensive but works better for transformers.
""")

# Show the difference
x = torch.linspace(-3, 3, 7)
print("Comparison at key points:")
print(f"  x:        {[f'{v:.1f}' for v in x.tolist()]}")
print(f"  ReLU(x):  {[f'{v:.2f}' for v in F.relu(x).tolist()]}")
print(f"  GELU(x):  {[f'{v:.2f}' for v in F.gelu(x).tolist()]}")

pause()


# ---------------------------------------------------------------------------
# Residual Connections
# ---------------------------------------------------------------------------

print("RESIDUAL CONNECTIONS")
print("-" * 40)

print("""
Residual connections add the input to the output:

    output = x + sublayer(x)

Why is this so important?

1. GRADIENT FLOW
   Without residual: gradients must flow through every layer
   With residual: gradients have a "shortcut" path
   
2. LEARNING IDENTITY
   If sublayer outputs zeros, output = x (identity function)
   The network can easily learn to "skip" unhelpful layers
   
3. DEEP NETWORKS
   ResNets proved this enables training 100+ layer networks
   Transformers use residuals everywhere

Without residual connections, training deep transformers would fail!
""")

pause()


# ---------------------------------------------------------------------------
# The Transformer Block
# ---------------------------------------------------------------------------

print("THE COMPLETE TRANSFORMER BLOCK")
print("-" * 40)

print("""
GPT uses "Pre-LayerNorm" - LayerNorm BEFORE each sublayer:

    x ─────────────────┐
    │                  │
    ▼                  │ (residual)
    LayerNorm          │
    │                  │
    ▼                  │
    Attention          │
    │                  │
    ▼                  │
    + ◄────────────────┘
    │
    ├─────────────────┐
    │                 │
    ▼                 │ (residual)
    LayerNorm         │
    │                 │
    ▼                 │
    FFN               │
    │                 │
    ▼                 │
    + ◄───────────────┘
    │
    output

Let's implement it!
""")

pause()


class TransformerBlock(nn.Module):
    """
    A single transformer block for GPT.
    
    Uses Pre-LayerNorm: normalize BEFORE each sublayer.
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, max_seq_length=512, dropout=0.1):
        super().__init__()
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Attention
        self.attention = MultiHeadCausalAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Feed-forward
        self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x):
        # Attention block with residual
        x = x + self.attention(self.ln1(x))
        
        # FFN block with residual
        x = x + self.ffn(self.ln2(x))
        
        return x


# Test the block
block = TransformerBlock(
    d_model=64,
    num_heads=4,
    d_ff=256,
    max_seq_length=128,
    dropout=0.1
)
block.eval()

print("Transformer Block created!")
print()

# Count parameters
total = sum(p.numel() for p in block.parameters())
attn = sum(p.numel() for p in block.attention.parameters())
ffn = sum(p.numel() for p in block.ffn.parameters())
ln = sum(p.numel() for p in block.ln1.parameters()) + sum(p.numel() for p in block.ln2.parameters())

print(f"Parameter breakdown:")
print(f"  Attention:   {attn:,}")
print(f"  FFN:         {ffn:,}")
print(f"  LayerNorms:  {ln:,}")
print(f"  Total:       {total:,}")
print()

x = torch.randn(2, 10, 64)
output = block(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print()
print("Input and output have the same shape - blocks can be stacked!")

pause()


# ---------------------------------------------------------------------------
# Stacking Blocks
# ---------------------------------------------------------------------------

print("STACKING MULTIPLE BLOCKS")
print("-" * 40)

print("""
GPT stacks multiple identical transformer blocks:

    GPT-2 Small:  12 blocks
    GPT-2 Medium: 24 blocks
    GPT-2 Large:  36 blocks
    GPT-2 XL:     48 blocks
    GPT-3:        96 blocks

Each block refines the representations further.
""")

pause()


class StackedBlocks(nn.Module):
    """Stack of transformer blocks."""
    
    def __init__(self, num_blocks, d_model, num_heads, d_ff=None, 
                 max_seq_length=512, dropout=0.1):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_length, dropout)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# Create 6 stacked blocks
stacked = StackedBlocks(
    num_blocks=6,
    d_model=64,
    num_heads=4,
    d_ff=256,
    max_seq_length=128,
    dropout=0.1
)
stacked.eval()

total_params = sum(p.numel() for p in stacked.parameters())

print(f"6 Stacked Transformer Blocks:")
print(f"  Total parameters: {total_params:,}")
print(f"  Per block:        {total_params // 6:,}")
print()

x = torch.randn(2, 10, 64)
output = stacked(x)
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {output.shape}")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
1. LAYER NORMALIZATION
   - Normalizes activations for stable training
   - Learnable scale (gamma) and shift (beta)
   - Applied BEFORE sublayers (Pre-LayerNorm)

2. FEED-FORWARD NETWORK
   - Expand: d_model → 4*d_model
   - GELU activation
   - Contract: 4*d_model → d_model
   - Processes each position independently

3. RESIDUAL CONNECTIONS
   - output = x + sublayer(x)
   - Enables gradient flow
   - Makes deep networks trainable

4. TRANSFORMER BLOCK
   - LayerNorm → Attention → Residual
   - LayerNorm → FFN → Residual
   - Input and output have same shape

5. STACKING
   - GPT stacks 12-96 identical blocks
   - Each block refines representations

Next: Complete GPT model with embeddings and output layer!
""")

print("=" * 60)
print("  End of Lesson 1")
print("=" * 60)
