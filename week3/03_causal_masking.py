"""
Lesson 3: Causal Masking
=========================

GPT generates text one token at a time, left to right.
During training, we need to prevent the model from "peeking"
at future tokens - that would be cheating!

Causal masking ensures each position can only attend to
itself and previous positions. This is what makes GPT
a "causal" language model.

Usage: python 03_causal_masking.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 3: Causal Masking")
print("  Making Attention Work for GPT")
print("=" * 60)


# ---------------------------------------------------------------------------
# Why Causal Masking?
# ---------------------------------------------------------------------------

print("""
WHY CAUSAL MASKING?

GPT generates text left-to-right, one token at a time:
    "The" → "The cat" → "The cat sat" → ...

When predicting "sat", GPT should ONLY see "The cat".
It should NOT see what comes after.

Problem with regular attention:
    Every position can see every other position.
    When predicting at position 2, the model sees ALL tokens.
    It could just copy the answer - that's cheating!

Solution with causal masking:
    Position i can only see positions 0, 1, ..., i
    Future positions are "masked out"
""")

pause()


# ---------------------------------------------------------------------------
# The Causal Mask
# ---------------------------------------------------------------------------

print("THE CAUSAL MASK")
print("-" * 40)

print("""
A causal mask is a LOWER TRIANGULAR matrix:

                Position attending TO
                 0    1    2    3
Position    0 [  1    0    0    0  ]  ← Can only see itself
FROM        1 [  1    1    0    0  ]  ← Can see 0, 1
            2 [  1    1    1    0  ]  ← Can see 0, 1, 2  
            3 [  1    1    1    1  ]  ← Can see all (0, 1, 2, 3)

Reading: mask[i,j] = 1 means position i CAN attend to position j
         mask[i,j] = 0 means position i CANNOT attend to position j
""")

pause()

# Create the mask
seq_length = 4
causal_mask = torch.tril(torch.ones(seq_length, seq_length))

print("Creating a causal mask with torch.tril():")
print()
print(causal_mask)
print()
print("1 = can attend, 0 = cannot attend")

pause()


# ---------------------------------------------------------------------------
# Applying the Mask
# ---------------------------------------------------------------------------

print("APPLYING THE MASK")
print("-" * 40)

print("""
We apply the mask BEFORE softmax.

The trick: Set masked positions to -infinity!
    softmax(-inf) = 0

So masked positions get ZERO attention weight.
""")

pause()

# Create some attention scores
torch.manual_seed(42)
scores = torch.randn(seq_length, seq_length)

print("Original attention scores:")
for i, row in enumerate(scores):
    print(f"  Position {i}: {[f'{x:.2f}' for x in row.tolist()]}")

pause()

# Apply the mask
masked_scores = scores.masked_fill(causal_mask == 0, float('-inf'))

print("After masking (zeros become -inf):")
for i, row in enumerate(masked_scores):
    vals = []
    for x in row.tolist():
        if x == float('-inf'):
            vals.append('-inf')
        else:
            vals.append(f'{x:.2f}')
    print(f"  Position {i}: {vals}")

pause()

# Apply softmax
weights = F.softmax(masked_scores, dim=-1)

print("After softmax:")
for i, row in enumerate(weights):
    print(f"  Position {i}: {[f'{x:.3f}' for x in row.tolist()]}")

print()
print("Notice:")
print("  - Row 0 has weight only on position 0")
print("  - Row 1 has weights on positions 0, 1 only")
print("  - Each row sums to 1.0")
print("  - All -inf positions became 0!")

pause()


# ---------------------------------------------------------------------------
# Causal Self-Attention Module
# ---------------------------------------------------------------------------

print("CAUSAL SELF-ATTENTION MODULE")
print("-" * 40)


class CausalSelfAttention(nn.Module):
    """Self-attention with causal masking for GPT."""
    
    def __init__(self, d_model, max_seq_length=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Causal mask - registered as buffer (not a parameter)
        mask = torch.tril(torch.ones(max_seq_length, max_seq_length))
        self.register_buffer('mask', mask)
    
    def forward(self, x, return_weights=False):
        """
        Args:
            x: (batch_size, seq_length, d_model)
        
        Returns:
            output: (batch_size, seq_length, d_model)
        """
        batch_size, seq_length, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        
        # Apply causal mask
        mask = self.mask[:seq_length, :seq_length]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        
        # Apply attention to values
        attended = weights @ V
        
        # Output projection
        output = self.W_o(attended)
        output = self.output_dropout(output)
        
        if return_weights:
            return output, weights
        return output


# Test it
causal_attn = CausalSelfAttention(d_model=64, max_seq_length=128)
causal_attn.eval()

x = torch.randn(2, 6, 64)
output, weights = causal_attn(x, return_weights=True)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Weights shape: {weights.shape}")

pause()


# ---------------------------------------------------------------------------
# Verifying the Mask Works
# ---------------------------------------------------------------------------

print("VERIFYING THE CAUSAL MASK")
print("-" * 40)

# Look at attention weights for first sequence
w = weights[0].detach()

print("Attention weights for first sequence:")
print("(showing that future positions get zero weight)")
print()

for i in range(6):
    row = w[i].tolist()
    visible = [f'{row[j]:.2f}' if j <= i else '0.00' for j in range(6)]
    print(f"  Position {i}: {visible}")

print()
print("Each position only has non-zero weights for current and past positions!")

pause()


# ---------------------------------------------------------------------------
# Understanding register_buffer
# ---------------------------------------------------------------------------

print("UNDERSTANDING register_buffer")
print("-" * 40)

print("""
We used self.register_buffer('mask', mask)

Why not just self.mask = mask?

register_buffer tells PyTorch:
    - This tensor is part of the module's state
    - Save it when saving the model
    - Move it to GPU with the model
    - But DON'T treat it as a learnable parameter

The mask is fixed, not learned. But we still want it to:
    - Move to GPU when we call model.cuda()
    - Be saved/loaded with model checkpoints
""")

pause()

# Show the difference
print("Parameters (learned):")
for name, param in causal_attn.named_parameters():
    print(f"  {name}: {param.shape}")

print()
print("Buffers (not learned):")
for name, buf in causal_attn.named_buffers():
    print(f"  {name}: {buf.shape}")

pause()


# ---------------------------------------------------------------------------
# Causal vs Bidirectional
# ---------------------------------------------------------------------------

print("CAUSAL vs BIDIRECTIONAL ATTENTION")
print("-" * 40)

print("""
Different models use different masking:

GPT (Causal/Autoregressive):
    - Each position sees only past positions
    - Used for text GENERATION
    - "Given these words, predict the next word"
    
    Mask:
    [1 0 0 0]
    [1 1 0 0]
    [1 1 1 0]
    [1 1 1 1]

BERT (Bidirectional):
    - Each position sees ALL positions
    - Used for text UNDERSTANDING
    - "Given all context, understand the meaning"
    
    Mask: (no mask, or all 1s)
    [1 1 1 1]
    [1 1 1 1]
    [1 1 1 1]
    [1 1 1 1]

We're building GPT, so we use CAUSAL attention!
""")

pause()


# ---------------------------------------------------------------------------
# Complete Example
# ---------------------------------------------------------------------------

print("COMPLETE EXAMPLE")
print("-" * 40)

words = ["The", "cat", "sat", "on"]
seq_len = len(words)
d_model = 32

# Create attention
attn = CausalSelfAttention(d_model=d_model, max_seq_length=32)
attn.eval()

# Process
x = torch.randn(1, seq_len, d_model)
output, weights = attn(x, return_weights=True)
weights = weights[0].detach()

print(f"Processing: {words}")
print()
print("Attention pattern:")
print("-" * 45)
print(f"{'':10}", end="")
for w in words:
    print(f"{w:10}", end="")
print()
print("-" * 45)

for i, word in enumerate(words):
    print(f"{word:10}", end="")
    for j in range(seq_len):
        if j <= i:
            print(f"{weights[i, j].item():.3f}     ", end="")
        else:
            print(f"{'---':10}", end="")
    print()

print("-" * 45)
print()
print("'---' means masked (cannot attend to future)")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
1. WHY CAUSAL MASKING?
   - GPT generates left-to-right
   - Can't let model "peek" at future tokens
   - Each position sees only current and past

2. THE CAUSAL MASK
   - Lower triangular matrix: torch.tril(torch.ones(n, n))
   - mask[i,j] = 1 if j <= i, else 0

3. APPLYING THE MASK
   - Before softmax: scores.masked_fill(mask == 0, float('-inf'))
   - After softmax: masked positions become 0

4. IMPLEMENTATION
   - Use register_buffer for the mask (not a parameter)
   - Create mask once, slice for actual sequence length

5. CAUSAL vs BIDIRECTIONAL
   - GPT: Causal (sees past only)
   - BERT: Bidirectional (sees all)

Next: Multi-head attention - the final piece!
""")

print("=" * 60)
print("  End of Lesson 3")
print("=" * 60)
