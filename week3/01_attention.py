"""
Lesson 1: Understanding Attention
==================================

Attention is what makes transformers so powerful. It's the mechanism
that allows every word to "look at" every other word and decide
which ones are relevant.

This lesson builds the intuition for how attention works.

Usage: python 01_attention.py
"""

import torch
import torch.nn.functional as F
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 1: Understanding Attention")
print("  The Heart of Transformers")
print("=" * 60)


# ---------------------------------------------------------------------------
# The Problem Attention Solves
# ---------------------------------------------------------------------------

print("""
THE PROBLEM ATTENTION SOLVES

Consider this sentence:
    "The cat sat on the mat because it was tired."

What does "it" refer to? The cat, obviously.

For us, this is easy. For a model, it's hard:
    - "it" is 7 words away from "cat"
    - There's another noun in between ("mat")
    - The model needs to somehow connect distant words

Old approach (RNNs):
    - Process words one at a time: The → cat → sat → ...
    - Pass information forward through hidden states
    - Problem: Information gets "diluted" over distance

New approach (Attention):
    - Let every word directly look at every other word
    - Compute: "How relevant is each word to my current prediction?"
    - Distance doesn't matter!
""")

pause()


# ---------------------------------------------------------------------------
# The Query-Key-Value Framework
# ---------------------------------------------------------------------------

print("THE QUERY-KEY-VALUE FRAMEWORK")
print("-" * 40)

print("""
Attention uses three concepts: Query (Q), Key (K), and Value (V).

Think of it like searching a library:
    - QUERY: What you're looking for ("books about cats")
    - KEY: The label on each book ("animals", "cooking", "history")
    - VALUE: The actual content of each book

The process:
    1. Compare your QUERY to all KEYS
    2. Get relevance scores (how well does each key match?)
    3. Use scores to weight the VALUES
    4. Return weighted combination of values

For "The cat sat on the mat because it was tired":
    Query from "it": "What noun am I referring to?"
    Keys from other words: What each word represents
    Values from other words: Their meaning/content
    
    Result: "it" attends strongly to "cat", gets cat's information
""")

pause()


# ---------------------------------------------------------------------------
# The Attention Formula
# ---------------------------------------------------------------------------

print("THE ATTENTION FORMULA")
print("-" * 40)

print("""
    Attention(Q, K, V) = softmax(Q × K^T / √d) × V

Let's break this down:

    1. Q × K^T
       Dot product between query and all keys
       Measures similarity: higher = more similar
    
    2. / √d
       Scale by square root of dimension
       Prevents scores from getting too large
    
    3. softmax(...)
       Convert scores to probabilities (sum to 1)
       These are the "attention weights"
    
    4. × V
       Weight values by attention weights
       Return weighted combination

Let me show you step by step...
""")

pause()


# ---------------------------------------------------------------------------
# Step-by-Step Implementation
# ---------------------------------------------------------------------------

print("STEP 1: Setup")
print("-" * 40)

# Create sample data
seq_length = 4
d_model = 8

torch.manual_seed(42)
embeddings = torch.randn(seq_length, d_model)

print(f"Imagine we have 4 words, each with 8-dimensional embeddings.")
print(f"Embedding shape: {embeddings.shape}")
print()
print("For basic attention, Q = K = V = embeddings")

# For basic attention, Q = K = V
Q = embeddings
K = embeddings
V = embeddings

pause()


print("STEP 2: Compute Attention Scores")
print("-" * 40)

# Q × K^T
# (4, 8) × (8, 4) = (4, 4)
scores = Q @ K.T

print("Attention scores = Q × K^T")
print(f"Shape: {scores.shape} (4×4 matrix)")
print()
print("Each entry [i,j] = how much word i attends to word j")
print()
print("Scores:")
for i, row in enumerate(scores):
    print(f"  Word {i}: {[f'{x:.2f}' for x in row.tolist()]}")

pause()


print("STEP 3: Scale the Scores")
print("-" * 40)

print("""
We divide by √d to prevent extreme values.

Large scores → softmax becomes very "peaky" (one value dominates)
Scaling keeps values in a reasonable range.
""")

d_k = d_model
scaled_scores = scores / math.sqrt(d_k)

print(f"Scaling factor: √{d_k} = {math.sqrt(d_k):.2f}")
print()
print("Scaled scores:")
for i, row in enumerate(scaled_scores):
    print(f"  Word {i}: {[f'{x:.2f}' for x in row.tolist()]}")

pause()


print("STEP 4: Apply Softmax")
print("-" * 40)

print("""
Softmax converts scores to probabilities.
Each row sums to 1.0 - these are the attention weights.
""")

weights = F.softmax(scaled_scores, dim=-1)

print("Attention weights (each row sums to 1):")
for i, row in enumerate(weights):
    row_sum = sum(row.tolist())
    print(f"  Word {i}: {[f'{x:.3f}' for x in row.tolist()]} (sum={row_sum:.3f})")

pause()


print("STEP 5: Weight the Values")
print("-" * 40)

print("""
Finally, we compute a weighted combination of values.
Each word's output is the weighted sum of all values.
""")

output = weights @ V

print(f"Output shape: {output.shape} (same as input!)")
print()
print("Each output position is a weighted combination of all value vectors,")
print("where the weights tell us how much to attend to each position.")

pause()


# ---------------------------------------------------------------------------
# Complete Attention Function
# ---------------------------------------------------------------------------

print("COMPLETE ATTENTION FUNCTION")
print("-" * 40)


def attention(Q, K, V):
    """
    Basic attention mechanism.
    
    Args:
        Q: Queries (seq_len, d_model)
        K: Keys (seq_len, d_model)
        V: Values (seq_len, d_model)
    
    Returns:
        output: Attended values (seq_len, d_model)
        weights: Attention weights (seq_len, seq_len)
    """
    d_k = K.shape[-1]
    
    # Compute scaled attention scores
    scores = Q @ K.T / math.sqrt(d_k)
    
    # Convert to probabilities
    weights = F.softmax(scores, dim=-1)
    
    # Weighted combination of values
    output = weights @ V
    
    return output, weights


# Test it
output, weights = attention(embeddings, embeddings, embeddings)

print("Testing our attention function:")
print(f"  Input shape: {embeddings.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Weights shape: {weights.shape}")

pause()


# ---------------------------------------------------------------------------
# Visualizing Attention Patterns
# ---------------------------------------------------------------------------

print("VISUALIZING ATTENTION PATTERNS")
print("-" * 40)

print("""
Let's pretend our words are: ["The", "cat", "is", "sleeping"]

The attention weights show how much each word attends to others:
""")

words = ["The", "cat", "is", "sleeping"]

print("Attention weights:")
print("-" * 45)
print(f"{'':12}", end="")
for w in words:
    print(f"{w:10}", end="")
print()
print("-" * 45)

for i, word in enumerate(words):
    print(f"{word:12}", end="")
    for j in range(len(words)):
        print(f"{weights[i, j].item():10.3f}", end="")
    print()

print("-" * 45)
print()
print("Reading: Row shows what that word attends to.")
print(f"Example: '{words[1]}' attends to '{words[3]}' with weight {weights[1, 3].item():.3f}")

pause()


# ---------------------------------------------------------------------------
# Batched Attention
# ---------------------------------------------------------------------------

print("HANDLING BATCHES")
print("-" * 40)

print("""
In practice, we process multiple sequences at once.
The shapes become: (batch_size, seq_length, d_model)

The attention formula is the same, just with an extra dimension.
""")


def batched_attention(Q, K, V):
    """
    Attention with batched inputs.
    
    Args:
        Q, K, V: (batch_size, seq_len, d_model)
    
    Returns:
        output: (batch_size, seq_len, d_model)
        weights: (batch_size, seq_len, seq_len)
    """
    d_k = K.shape[-1]
    
    # Transpose last two dims of K: (batch, seq, d) -> (batch, d, seq)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    
    weights = F.softmax(scores, dim=-1)
    output = weights @ V
    
    return output, weights


# Test with batch
batch_size = 2
batch_embeddings = torch.randn(batch_size, seq_length, d_model)
output, weights = batched_attention(batch_embeddings, batch_embeddings, batch_embeddings)

print(f"Batch input shape: {batch_embeddings.shape}")
print(f"Output shape: {output.shape}")
print(f"Weights shape: {weights.shape}")

pause()


# ---------------------------------------------------------------------------
# Why Attention is Powerful
# ---------------------------------------------------------------------------

print("WHY ATTENTION IS SO POWERFUL")
print("-" * 40)

print("""
1. LONG-RANGE DEPENDENCIES
   Every word can directly attend to every other word.
   Word 1 can easily attend to word 100 - distance doesn't matter.

2. PARALLELIZATION
   All attention computations happen simultaneously.
   GPUs love this - much faster than sequential RNNs.

3. INTERPRETABILITY
   Attention weights show what the model "looks at."
   We can visualize which words relate to which.

4. FLEXIBILITY
   Same mechanism works for any sequence length.
   Can be adapted for many different tasks.

This is why the paper was called "Attention Is All You Need"!
""")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
1. ATTENTION lets every position look at every other position

2. QUERY-KEY-VALUE framework:
   - Query: What am I looking for?
   - Key: What does each position offer?
   - Value: What information to return?

3. THE FORMULA:
   Attention(Q, K, V) = softmax(Q × K^T / √d) × V

4. ATTENTION WEIGHTS show how much each position attends to others
   - Each row sums to 1
   - After training, they become meaningful patterns

5. BATCHED ATTENTION processes multiple sequences at once

Next: Self-attention with learnable projections!
""")

print("=" * 60)
print("  End of Lesson 1")
print("=" * 60)
