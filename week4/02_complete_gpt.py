"""
Lesson 2: The Complete GPT Model
=================================

This is the moment you've been working towards!

We assemble all the pieces into a complete GPT:
    - Token embeddings (Week 2)
    - Position embeddings (Week 2)
    - Multi-head causal attention (Week 3)
    - Transformer blocks (Lesson 1)
    - Output projection to vocabulary

By the end, you'll have a working GPT architecture!

Usage: python 02_complete_gpt.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 2: The Complete GPT Model")
print("  Building GPT from Scratch")
print("=" * 60)

print("""
THE COMPLETE GPT ARCHITECTURE

    Token IDs
        │
        ▼
    Token Embedding + Position Embedding
        │
        ▼
    ┌─────────────────┐
    │ Transformer     │ ──┐
    │ Block 1         │   │
    └─────────────────┘   │
            │             │  × num_layers
            ▼             │  (12 for GPT-2 Small)
    ┌─────────────────┐   │
    │ Transformer     │   │
    │ Block 2...N     │   │
    └─────────────────┘   │
            │         ────┘
            ▼
    Final LayerNorm
        │
        ▼
    Linear → Vocabulary Size
        │
        ▼
    Logits (scores for each token)
""")

pause()


# ---------------------------------------------------------------------------
# Building Blocks (from previous lessons)
# ---------------------------------------------------------------------------

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


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, num_heads, max_seq_length, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# The Complete GPT Model
# ---------------------------------------------------------------------------

print("THE COMPLETE GPT MODEL")
print("-" * 40)

print("""
Now let's put it all together!
""")

pause()


class GPT(nn.Module):
    """
    GPT Language Model.
    
    This is the complete architecture used in GPT-2/GPT-3.
    """
    
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        max_seq_length=512,
        d_ff=None,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token embeddings: vocab_size -> d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embeddings: max_seq_length -> d_model
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Dropout after embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_length, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(d_model)
        
        # Output projection (language model head)
        # Projects from d_model back to vocabulary size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # WEIGHT TYING: Share weights between token embedding and output
        # This is a key trick that improves performance and reduces parameters
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        """
        Forward pass.
        
        Args:
            input_ids: Token indices (batch_size, seq_length)
            targets: Target token indices for training (batch_size, seq_length)
        
        Returns:
            logits: Predictions (batch_size, seq_length, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        batch_size, seq_length = input_ids.shape
        
        # Get token embeddings
        token_emb = self.token_embedding(input_ids)  # (B, T, d_model)
        
        # Get position embeddings
        positions = torch.arange(seq_length, device=input_ids.device)
        pos_emb = self.position_embedding(positions)  # (T, d_model)
        
        # Combine and apply dropout
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        return logits, loss


print("GPT class defined!")
print()
print("Key components:")
print("  - token_embedding: Maps token IDs to vectors")
print("  - position_embedding: Adds position information")
print("  - blocks: Stack of transformer blocks")
print("  - ln_final: Final layer normalization")
print("  - lm_head: Projects to vocabulary (weight-tied)")

pause()


# ---------------------------------------------------------------------------
# Testing the Model
# ---------------------------------------------------------------------------

print("TESTING THE GPT MODEL")
print("-" * 40)

# Create a small GPT
model = GPT(
    vocab_size=1000,
    d_model=64,
    num_heads=4,
    num_layers=4,
    max_seq_length=128,
    dropout=0.1
)
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())

print(f"Model configuration:")
print(f"  vocab_size: 1000")
print(f"  d_model: 64")
print(f"  num_heads: 4")
print(f"  num_layers: 4")
print(f"  Total parameters: {total_params:,}")
print()

# Test forward pass
batch_size = 2
seq_length = 10
input_ids = torch.randint(0, 1000, (batch_size, seq_length))

logits, _ = model(input_ids)

print(f"Forward pass:")
print(f"  Input shape: {input_ids.shape}")
print(f"  Output shape: {logits.shape}")
print(f"  (batch_size, seq_length, vocab_size)")

pause()


# ---------------------------------------------------------------------------
# Weight Tying
# ---------------------------------------------------------------------------

print("WEIGHT TYING")
print("-" * 40)

print("""
Notice this line in our GPT:
    self.lm_head.weight = self.token_embedding.weight

This is "weight tying" - the input embedding and output projection
share the SAME weight matrix!

Why does this make sense?
    - Input: token_id → embedding vector
    - Output: embedding vector → token_id (reverse!)
    
Using the same matrix for both directions:
    - Reduces parameters significantly
    - Actually improves performance
    - Makes intuitive sense

Let's verify they're the same:
""")

pause()

print(f"Token embedding shape: {model.token_embedding.weight.shape}")
print(f"LM head weight shape: {model.lm_head.weight.shape}")
print(f"Same object? {model.token_embedding.weight is model.lm_head.weight}")
print()

# Calculate savings
without_tying = 1000 * 64 * 2  # Two separate matrices
with_tying = 1000 * 64         # One shared matrix
print(f"Parameters saved: {without_tying - with_tying:,}")

pause()


# ---------------------------------------------------------------------------
# Training with Loss
# ---------------------------------------------------------------------------

print("TRAINING WITH LOSS")
print("-" * 40)

print("""
GPT is trained to predict the next token at each position:

    Input:  [The] [cat] [sat] [on] [the]
    Target: [cat] [sat] [on] [the] [mat]

The model outputs logits for each position, and we compute
cross-entropy loss between predictions and targets.
""")

pause()

# Create input and target (target is input shifted by 1)
input_ids = torch.randint(0, 1000, (2, 10))
targets = torch.randint(0, 1000, (2, 10))

logits, loss = model(input_ids, targets)

print(f"Input shape: {input_ids.shape}")
print(f"Target shape: {targets.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss.item():.4f}")
print()
print("The loss tells us how wrong our predictions are.")
print("During training, we minimize this loss!")

pause()


# ---------------------------------------------------------------------------
# GPT-2 Configurations
# ---------------------------------------------------------------------------

print("GPT-2 CONFIGURATIONS")
print("-" * 40)

print("""
Let's see what real GPT-2 models look like:
""")

configs = {
    "GPT-2 Small":  {"vocab_size": 50257, "d_model": 768,  "num_heads": 12, "num_layers": 12},
    "GPT-2 Medium": {"vocab_size": 50257, "d_model": 1024, "num_heads": 16, "num_layers": 24},
    "GPT-2 Large":  {"vocab_size": 50257, "d_model": 1280, "num_heads": 20, "num_layers": 36},
    "GPT-2 XL":     {"vocab_size": 50257, "d_model": 1600, "num_heads": 25, "num_layers": 48},
}

print(f"{'Model':<15} {'d_model':>8} {'heads':>6} {'layers':>7} {'Parameters':>15}")
print("-" * 55)

for name, config in configs.items():
    gpt = GPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_length=1024
    )
    params = sum(p.numel() for p in gpt.parameters())
    print(f"{name:<15} {config['d_model']:>8} {config['num_heads']:>6} {config['num_layers']:>7} {params:>15,}")
    del gpt  # Free memory

print()
print("These match the original GPT-2 parameter counts!")

pause()


# ---------------------------------------------------------------------------
# Text Generation
# ---------------------------------------------------------------------------

print("TEXT GENERATION")
print("-" * 40)

print("""
The model outputs logits for each position. To generate text:

    1. Get logits for the last position
    2. Apply softmax to get probabilities
    3. Sample from the distribution
    4. Append sampled token to sequence
    5. Repeat!
""")

pause()


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature=1.0):
    """
    Generate text autoregressively.
    
    Args:
        model: GPT model
        input_ids: Starting tokens (batch_size, seq_length)
        max_new_tokens: Number of tokens to generate
        temperature: Higher = more random, lower = more deterministic
    
    Returns:
        Generated tokens (batch_size, seq_length + max_new_tokens)
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        # Crop to max sequence length if needed
        idx_cond = input_ids[:, -model.max_seq_length:]
        
        # Get predictions
        logits, _ = model(idx_cond)
        
        # Take logits at last position only
        logits = logits[:, -1, :] / temperature
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return input_ids


# Test generation (model is untrained, so output will be random)
model = GPT(vocab_size=100, d_model=64, num_heads=4, num_layers=2, max_seq_length=64)
model.eval()

start_tokens = torch.tensor([[1]])  # Start with token 1
generated = generate(model, start_tokens, max_new_tokens=10, temperature=1.0)

print("Testing generation (untrained model):")
print(f"  Start tokens: {start_tokens.tolist()}")
print(f"  Generated: {generated.tolist()}")
print()
print("Output is random since model is untrained.")
print("After training, it would generate coherent text!")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  CONGRATULATIONS!")
print("  You've built GPT from scratch!")
print("=" * 60)

print("""
THE COMPLETE GPT ARCHITECTURE:

1. INPUT PROCESSING
   - Token embedding: token_id → vector
   - Position embedding: position → vector
   - Combined: token_emb + pos_emb

2. TRANSFORMER BLOCKS (× N)
   - LayerNorm → Attention → Residual
   - LayerNorm → FFN → Residual

3. OUTPUT PROCESSING
   - Final LayerNorm
   - Linear projection to vocabulary (weight-tied)
   - Logits for each token

4. TRAINING
   - Input: tokens [0, 1, 2, ..., n-1]
   - Target: tokens [1, 2, 3, ..., n] (shifted)
   - Loss: Cross-entropy

5. GENERATION
   - Get logits for last position
   - Sample next token
   - Append and repeat

THIS IS THE EXACT ARCHITECTURE OF GPT-2 AND GPT-3!
(Just scaled up with more layers and bigger dimensions)

Next: Training the model on real text!
""")

print("=" * 60)
print("  End of Lesson 2")
print("=" * 60)
