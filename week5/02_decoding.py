"""
Lesson 2: Decoding Strategies
==============================

When generating text, we have choices about HOW to select
the next token from the model's probability distribution.

Different strategies produce very different outputs:
    - Greedy: Always pick most likely (boring, repetitive)
    - Temperature: Control randomness
    - Top-k: Sample from top k tokens only
    - Top-p: Sample from smallest set summing to p

This lesson covers all major decoding strategies used in
production LLMs like ChatGPT!

Usage: python 02_decoding.py
"""

import torch
import torch.nn.functional as F
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 2: Decoding Strategies")
print("  Temperature, Top-k, and Top-p Sampling")
print("=" * 60)

print("""
THE PROBLEM

The model outputs logits (scores) for each possible next token.
We convert these to probabilities with softmax.
But then... how do we actually pick the next token?

Options:
    1. GREEDY: Always pick the highest probability
    2. TEMPERATURE: Adjust the "sharpness" of distribution
    3. TOP-K: Only consider the k most likely tokens
    4. TOP-P: Consider smallest set with cumulative prob > p

Each has tradeoffs between coherence and creativity.
""")

pause()


# Shared vocabulary for examples
vocab = ["the", "a", "cat", "dog", "sat", "ran"]
logits = torch.tensor([2.0, 0.5, 1.5, 1.2, 0.8, 0.3])


# ---------------------------------------------------------------------------
# Greedy Decoding
# ---------------------------------------------------------------------------

print("GREEDY DECODING")
print("-" * 40)

print("""
Greedy: Always pick the token with highest probability.

    next_token = argmax(probabilities)

Pros:
    - Deterministic (same input = same output)
    - Coherent (picks "safe" choices)

Cons:
    - Boring and repetitive
    - Gets stuck in loops
    - Never explores creative options
""")

pause()


def greedy_decode(logits):
    """Pick the most likely token."""
    return logits.argmax(dim=-1)


probs = F.softmax(logits, dim=-1)
chosen = greedy_decode(logits)

print(f"Vocabulary: {vocab}")
print(f"Logits:     {logits.tolist()}")
print(f"Probabilities: {[f'{p:.2f}' for p in probs.tolist()]}")
print()
print(f"Greedy choice: '{vocab[chosen]}' (probability {probs[chosen]:.2f})")
print()
print("Problem: Greedy ALWAYS picks 'the' - no variety!")
print("In text generation, this leads to boring, repetitive output.")

pause()


# ---------------------------------------------------------------------------
# Temperature Sampling
# ---------------------------------------------------------------------------

print("TEMPERATURE SAMPLING")
print("-" * 40)

print("""
Temperature controls the "sharpness" of the probability distribution.

    adjusted_logits = logits / temperature
    probs = softmax(adjusted_logits)

Temperature effects:
    T < 1.0: Sharper (more confident, less random)
    T = 1.0: Original distribution
    T > 1.0: Flatter (less confident, more random)
    T → 0:   Becomes greedy
    T → ∞:   Becomes uniform random

Lower temperature = more focused, predictable
Higher temperature = more creative, chaotic
""")

pause()


def temperature_sample(logits, temperature=1.0):
    """Sample with temperature scaling."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


print(f"Original logits: {logits.tolist()}")
print()
print("Temperature effects on probabilities:")
print("-" * 55)

for temp in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    scaled = logits / temp
    probs = F.softmax(scaled, dim=-1)
    probs_str = [f"{p:.2f}" for p in probs.tolist()]
    print(f"T={temp:.1f}: {probs_str}")

print()
print("Lower T → probability concentrated on top token ('the')")
print("Higher T → probability spread more evenly")

pause()


print("TEMPERATURE SAMPLING IN ACTION")
print("-" * 40)

print()
print("Sampling 1000 times at different temperatures:")
print(f"Tokens: {vocab}")
print("-" * 55)


def sample_many(logits, temperature, num_samples=1000):
    """Sample many times and count results."""
    counts = torch.zeros(len(logits))
    for _ in range(num_samples):
        idx = temperature_sample(logits, temperature)
        counts[idx] += 1
    return counts / num_samples


for temp in [0.3, 0.5, 0.7, 1.0, 1.5]:
    freqs = sample_many(logits, temp, 1000)
    freq_str = [f"{f:.2f}" for f in freqs.tolist()]
    print(f"T={temp:.1f}: {freq_str}")

print()
print("T=0.3: 'the' almost always")
print("T=1.5: Much more variety")

pause()


# ---------------------------------------------------------------------------
# Top-K Sampling
# ---------------------------------------------------------------------------

print("TOP-K SAMPLING")
print("-" * 40)

print("""
Top-k: Only consider the k most likely tokens, ignore the rest.

    1. Sort tokens by probability
    2. Keep only top k tokens
    3. Re-normalize to sum to 1
    4. Sample from this reduced set

Why top-k?
    - Prevents sampling very unlikely tokens
    - "The cat sat on the [refrigerator]" ← unlikely but possible!
    - Top-k avoids such surprises

Typical values: k = 40-100
""")

pause()


def top_k_sample(logits, k, temperature=1.0):
    """Sample from top-k tokens only."""
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Find top k values and indices
    top_k_logits, top_k_indices = torch.topk(scaled_logits, k)
    
    # Convert to probabilities
    probs = F.softmax(top_k_logits, dim=-1)
    
    # Sample from top k
    idx = torch.multinomial(probs, num_samples=1)
    
    # Map back to original vocabulary index
    return top_k_indices[idx]


print(f"Vocabulary: {vocab}")
probs = F.softmax(logits, dim=-1)
print(f"Full probabilities: {[f'{p:.3f}' for p in probs.tolist()]}")
print()

# Show top-k filtering
for k in [2, 3, 4]:
    top_k_logits, top_k_idx = torch.topk(logits, k)
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    
    tokens_kept = [vocab[i] for i in top_k_idx.tolist()]
    probs_kept = [f"{p:.3f}" for p in top_k_probs.tolist()]
    
    print(f"Top-{k}: tokens={tokens_kept}, probs={probs_kept}")

print()
print("Top-k filters out unlikely tokens before sampling.")

pause()


# ---------------------------------------------------------------------------
# Top-P (Nucleus) Sampling
# ---------------------------------------------------------------------------

print("TOP-P (NUCLEUS) SAMPLING")
print("-" * 40)

print("""
Top-p: Keep the smallest set of tokens whose cumulative probability >= p.

    1. Sort tokens by probability (descending)
    2. Compute cumulative sum
    3. Keep tokens until cumsum exceeds p
    4. Sample from this dynamic set

Why top-p over top-k?
    - ADAPTS to the distribution!
    - If model is confident: only 1-2 tokens included
    - If model is uncertain: many tokens included
    - More flexible than fixed k

Also called "nucleus sampling" - samples from probability "nucleus"

Typical values: p = 0.9-0.95
""")

pause()


def top_p_sample(logits, p, temperature=1.0):
    """Sample using nucleus (top-p) sampling."""
    # Apply temperature
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sort by probability
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff
    mask = cumsum <= p
    # Always include at least one token
    mask = torch.cat([torch.tensor([True]), mask[:-1]])
    
    # Zero out tokens outside nucleus
    filtered_probs = probs.clone()
    indices_to_remove = sorted_indices[~mask]
    filtered_probs[indices_to_remove] = 0
    
    # Re-normalize
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    return torch.multinomial(filtered_probs, num_samples=1), filtered_probs


probs = F.softmax(logits, dim=-1)
print(f"Vocabulary: {vocab}")
print(f"Probabilities: {[f'{p:.3f}' for p in probs.tolist()]}")
print()

# Show cumulative sums
sorted_probs, sorted_idx = torch.sort(probs, descending=True)
cumsum = torch.cumsum(sorted_probs, dim=-1)

print("Sorted by probability:")
print(f"  Tokens:     {[vocab[i] for i in sorted_idx.tolist()]}")
print(f"  Probs:      {[f'{p:.3f}' for p in sorted_probs.tolist()]}")
print(f"  Cumulative: {[f'{c:.3f}' for c in cumsum.tolist()]}")
print()

for p_val in [0.5, 0.8, 0.95]:
    _, filtered = top_p_sample(logits, p=p_val)
    filtered_str = [f"{p:.3f}" for p in filtered.tolist()]
    num_kept = (filtered > 0).sum().item()
    print(f"Top-p={p_val}: {num_kept} tokens kept -> {filtered_str}")

pause()


# ---------------------------------------------------------------------------
# Combining Strategies
# ---------------------------------------------------------------------------

print("COMBINING STRATEGIES")
print("-" * 40)

print("""
In practice, we often COMBINE strategies:

    Temperature + Top-p (most common):
        1. Apply temperature scaling
        2. Apply top-p filtering
        3. Sample

    Temperature + Top-k:
        1. Apply temperature scaling
        2. Keep only top-k tokens
        3. Sample

ChatGPT uses temperature + top-p by default!
""")

pause()


def sample(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Flexible sampling with temperature, top-k, and/or top-p.
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-k
    if top_k is not None:
        top_k_logits, top_k_idx = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(0, top_k_idx, top_k_logits)
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Apply top-p
    if top_p is not None:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = torch.cat([torch.tensor([True]), cumsum[:-1] <= top_p])
        indices_to_remove = sorted_idx[~mask]
        probs[indices_to_remove] = 0
        probs = probs / probs.sum()
    
    return torch.multinomial(probs, num_samples=1)


print("Testing combined strategies (100 samples each):")
print(f"Vocabulary: {vocab}")
print()

configs = [
    {"temperature": 1.0},
    {"temperature": 0.7},
    {"temperature": 0.7, "top_k": 3},
    {"temperature": 0.7, "top_p": 0.8},
    {"temperature": 0.7, "top_k": 4, "top_p": 0.9},
]

for params in configs:
    counts = torch.zeros(6)
    for _ in range(100):
        idx = sample(logits.clone(), **params)
        counts[idx] += 1
    
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    count_str = [f"{int(c):>2}" for c in counts.tolist()]
    print(f"  {param_str:40} -> {count_str}")

pause()


# ---------------------------------------------------------------------------
# Practical Recommendations
# ---------------------------------------------------------------------------

print("PRACTICAL RECOMMENDATIONS")
print("-" * 40)

print("""
When to use what?

GREEDY (temperature=0):
    - Factual questions needing consistent answers
    - Code generation (deterministic)
    - When reproducibility matters

LOW TEMPERATURE (0.3-0.7):
    - Professional writing
    - Summarization
    - Translation
    
MEDIUM TEMPERATURE (0.7-1.0) + TOP-P (0.9):
    - Creative writing
    - Conversational AI
    - This is ChatGPT's default!

HIGH TEMPERATURE (1.0-1.5):
    - Brainstorming
    - Poetry
    - When you want maximum creativity

TOP-K (40-100):
    - General purpose
    - Faster than top-p
    - Good default choice

TOP-P (0.9-0.95):
    - Adapts to distribution shape
    - Most flexible
    - Industry standard

Common combinations:
    - ChatGPT: temperature=0.7, top_p=0.9
    - Creative: temperature=1.0, top_p=0.95
    - Factual: temperature=0 (greedy)
""")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
1. GREEDY
   - Always pick argmax(probs)
   - Deterministic but boring
   - Use for factual/code tasks

2. TEMPERATURE
   - logits / T before softmax
   - Low T = focused, High T = random
   - T=0 is greedy, T→∞ is uniform

3. TOP-K
   - Keep only top k tokens
   - Fixed k regardless of distribution
   - Simple and fast

4. TOP-P (NUCLEUS)
   - Keep smallest set where cumsum >= p
   - Adapts to distribution shape
   - Industry standard

5. COMBINING
   - Temperature + Top-p is most common
   - ChatGPT: T=0.7, top_p=0.9

Next: Saving and loading model weights!
""")

print("=" * 60)
print("  End of Lesson 2")
print("=" * 60)
