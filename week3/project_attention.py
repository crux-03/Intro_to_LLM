"""
Week 3 Project: Building Complete Attention
============================================

Put together everything you learned this week to build
the complete multi-head causal attention module used in GPT.

This is a real, production-quality implementation!

Usage: python project_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Week 3 Project: Complete Attention Module")
print("  Building the Heart of GPT")
print("=" * 60)


# ---------------------------------------------------------------------------
# The Complete Implementation
# ---------------------------------------------------------------------------

print("""
THE PROJECT

Build a complete multi-head causal attention module:
    - Efficient combined QKV projection
    - Proper head splitting and combining
    - Causal masking
    - Dropout for regularization

This is exactly what GPT uses!
""")

pause()


class MultiHeadCausalAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    
    This is the complete attention mechanism used in GPT models.
    Each head can learn different attention patterns, and the
    causal mask ensures no "peeking" at future tokens.
    
    Args:
        d_model: Model dimension (embedding size)
        num_heads: Number of attention heads
        max_seq_length: Maximum sequence length for mask
        dropout: Dropout probability
    """
    
    def __init__(self, d_model, num_heads, max_seq_length=512, dropout=0.1):
        super().__init__()
        
        # Validate dimensions
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Combined projection for Q, K, V (more efficient)
        # Single matrix: (d_model) -> (3 * d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Create causal mask and register as buffer
        # Buffer = part of module state, but not a learnable parameter
        mask = torch.tril(torch.ones(max_seq_length, max_seq_length))
        self.register_buffer('mask', mask)
    
    def forward(self, x, return_weights=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_length, d_model)
            return_weights: If True, also return attention weights
        
        Returns:
            output: (batch_size, seq_length, d_model)
            weights: (optional) (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size, seq_length, _ = x.shape
        
        # Step 1: Project to Q, K, V in one operation
        qkv = self.qkv_proj(x)  # (batch, seq, 3 * d_model)
        
        # Step 2: Reshape to separate Q, K, V and heads
        # (batch, seq, 3*d_model) -> (batch, seq, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        
        # Reorder to (3, batch, num_heads, seq, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Step 3: Compute attention scores
        # Q @ K^T: (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq)
        #        = (batch, heads, seq, seq)
        scores = Q @ K.transpose(-2, -1)
        
        # Scale by sqrt(head_dim)
        scores = scores / math.sqrt(self.head_dim)
        
        # Step 4: Apply causal mask
        mask = self.mask[:seq_length, :seq_length]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 5: Softmax to get attention weights
        weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights (during training)
        weights = self.attn_dropout(weights)
        
        # Step 6: Apply attention to values
        # weights @ V: (batch, heads, seq, seq) @ (batch, heads, seq, head_dim)
        #            = (batch, heads, seq, head_dim)
        attended = weights @ V
        
        # Step 7: Concatenate heads
        # Transpose: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
        attended = attended.transpose(1, 2)
        # Reshape: (batch, seq, heads, head_dim) -> (batch, seq, d_model)
        attended = attended.reshape(batch_size, seq_length, self.d_model)
        
        # Step 8: Output projection
        output = self.out_proj(attended)
        output = self.output_dropout(output)
        
        if return_weights:
            return output, weights
        return output


print("MultiHeadCausalAttention class defined!")
print()
print("Key components:")
print("  - qkv_proj: Combined Q, K, V projection")
print("  - out_proj: Output projection")
print("  - mask: Causal attention mask (buffer)")
print("  - dropout: Regularization")

pause()


# ---------------------------------------------------------------------------
# Testing the Implementation
# ---------------------------------------------------------------------------

print("TESTING THE IMPLEMENTATION")
print("-" * 40)

# Create attention module
attn = MultiHeadCausalAttention(
    d_model=64,
    num_heads=4,
    max_seq_length=128,
    dropout=0.1
)

# Print parameter count
total_params = sum(p.numel() for p in attn.parameters())
print(f"Total parameters: {total_params:,}")
print()

# Test shapes
print("Shape tests:")
for batch, seq in [(1, 10), (4, 32), (8, 64)]:
    x = torch.randn(batch, seq, 64)
    attn.eval()
    output = attn(x)
    print(f"  Input: {x.shape} -> Output: {output.shape}")

pause()


# ---------------------------------------------------------------------------
# Testing Causal Masking
# ---------------------------------------------------------------------------

print("VERIFYING CAUSAL MASKING")
print("-" * 40)

attn.eval()
x = torch.randn(1, 6, 64)
output, weights = attn(x, return_weights=True)

# Check that future positions have zero weight
weights_np = weights[0, 0].detach()  # First batch, first head

print("Attention weights for head 0:")
print("(should be zero above diagonal)")
print()

for i in range(6):
    row = weights_np[i].tolist()
    formatted = []
    for j, w in enumerate(row):
        if j <= i:
            formatted.append(f"{w:.3f}")
        else:
            # Should be zero (or very close due to numerical precision)
            assert w < 1e-6, f"Mask failed! Position [{i},{j}] = {w}"
            formatted.append("  0  ")
    print(f"  Position {i}: {formatted}")

print()
print("Causal mask verified! No attention to future positions.")

pause()


# ---------------------------------------------------------------------------
# GPT-2 Configuration Test
# ---------------------------------------------------------------------------

print("GPT-2 CONFIGURATION TEST")
print("-" * 40)

# GPT-2 Small dimensions
gpt2_small = MultiHeadCausalAttention(
    d_model=768,
    num_heads=12,
    max_seq_length=1024,
    dropout=0.1
)

params = sum(p.numel() for p in gpt2_small.parameters())

print("GPT-2 Small Attention Layer:")
print(f"  d_model: 768")
print(f"  num_heads: 12")
print(f"  head_dim: {768 // 12}")
print(f"  Parameters: {params:,}")
print()

# Test with typical input
x = torch.randn(2, 128, 768)
gpt2_small.eval()
output = gpt2_small(x)
print(f"  Input: {x.shape}")
print(f"  Output: {output.shape}")

pause()


# ---------------------------------------------------------------------------
# Visualizing Attention Patterns
# ---------------------------------------------------------------------------

print("VISUALIZING ATTENTION PATTERNS")
print("-" * 40)

# Small example for visualization
words = ["The", "cat", "sat", "down"]
vis_attn = MultiHeadCausalAttention(d_model=32, num_heads=4, max_seq_length=32)
vis_attn.eval()

x = torch.randn(1, len(words), 32)
_, weights = vis_attn(x, return_weights=True)
weights = weights[0].detach()  # Remove batch dim

print(f"Attention patterns for {len(words)} heads:")
print()

for head in range(4):
    print(f"Head {head}:")
    head_weights = weights[head]
    
    # Header
    print(f"  {'':6}", end="")
    for w in words:
        print(f"{w:>8}", end="")
    print()
    
    # Rows
    for i, word in enumerate(words):
        print(f"  {word:6}", end="")
        for j in range(len(words)):
            w = head_weights[i, j].item()
            if j <= i:
                print(f"{w:8.3f}", end="")
            else:
                print(f"{'---':>8}", end="")
        print()
    print()

print("Different heads show different (random) patterns.")
print("After training, they would specialize for different tasks!")

pause()


# ---------------------------------------------------------------------------
# Performance Check
# ---------------------------------------------------------------------------

print("PERFORMANCE CHECK")
print("-" * 40)

import time

# Benchmark
d_model = 512
num_heads = 8
seq_length = 256
batch_size = 16

perf_attn = MultiHeadCausalAttention(d_model=d_model, num_heads=num_heads)
perf_attn.eval()

x = torch.randn(batch_size, seq_length, d_model)

# Warmup
for _ in range(5):
    _ = perf_attn(x)

# Time it
num_runs = 20
start = time.time()
for _ in range(num_runs):
    _ = perf_attn(x)
end = time.time()

avg_time = (end - start) / num_runs * 1000  # ms

print(f"Configuration:")
print(f"  batch_size: {batch_size}")
print(f"  seq_length: {seq_length}")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print()
print(f"Average forward pass time: {avg_time:.2f} ms")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  PROJECT COMPLETE!")
print("=" * 60)

print("""
You've built a complete, production-quality attention module!

WHAT YOU IMPLEMENTED:

1. EFFICIENT QKV PROJECTION
   - Single matrix multiply for Q, K, V
   - Proper reshaping into heads

2. MULTI-HEAD ATTENTION
   - Parallel attention across multiple heads
   - Each head has dimension d_model // num_heads

3. CAUSAL MASKING
   - Lower triangular mask
   - Prevents attending to future positions

4. PROPER DROPOUT
   - On attention weights
   - On output

5. OUTPUT PROJECTION
   - Combines multi-head outputs

THIS IS EXACTLY WHAT GPT USES!

Next week: We'll add Feed-Forward Networks and Layer
Normalization, then stack attention blocks to build
the complete GPT architecture!
""")

print("=" * 60)
print("  End of Week 3 Project")
print("=" * 60)
