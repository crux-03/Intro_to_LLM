"""
Week 4 Project: Build and Train Your Own GPT
=============================================

This project brings together everything from Week 4.
You'll build a complete GPT model and train it on Shakespeare.

By the end, you'll have a working text generator!

Usage: python project_gpt.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Week 4 Project: Build and Train GPT")
print("=" * 60)


# ===========================================================================
# COMPLETE GPT IMPLEMENTATION
# ===========================================================================

class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class MultiHeadCausalAttention(nn.Module):
    """Multi-head causal self-attention."""
    
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
        
        # Combined QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        # Attention weights
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        
        # Apply attention
        out = (weights @ V).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-LayerNorm."""
    
    def __init__(self, d_model, num_heads, d_ff=None, max_seq_length=512, dropout=0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, num_heads, max_seq_length, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Complete GPT Language Model.
    
    Architecture:
        - Token + Position embeddings
        - N transformer blocks
        - Final LayerNorm
        - Output projection (weight-tied with embeddings)
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
        
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_length, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens autoregressively."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max length
            idx_cond = idx[:, -self.max_seq_length:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


# ===========================================================================
# TOKENIZER AND DATASET
# ===========================================================================

class CharTokenizer:
    """Character-level tokenizer."""
    
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char.get(i, '?') for i in indices])


class TextDataset(Dataset):
    """Dataset for language modeling."""
    
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return x, y


# ===========================================================================
# TRAINING DATA
# ===========================================================================

SHAKESPEARE = """
ROMEO: But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.
Be not her maid, since she is envious.
Her vestal livery is but sick and green
And none but fools do wear it; cast it off.
It is my lady, O, it is my love!
O, that she knew she were!
She speaks yet she says nothing. What of that?
Her eye discourses; I will answer it.
I am too bold, 'tis not to me she speaks.
Two of the fairest stars in all the heaven,
Having some business, do entreat her eyes
To twinkle in their spheres till they return.
What if her eyes were there, they in her head?
The brightness of her cheek would shame those stars,
As daylight doth a lamp; her eyes in heaven
Would through the airy region stream so bright
That birds would sing and think it were not night.
See, how she leans her cheek upon her hand!
O, that I were a glove upon that hand,
That I might touch that cheek!

JULIET: Ay me!

ROMEO: She speaks!
O, speak again, bright angel! for thou art
As glorious to this night, being o'er my head
As is a winged messenger of heaven.

JULIET: O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

ROMEO: Shall I hear more, or shall I speak at this?

JULIET: 'Tis but thy name that is my enemy;
Thou art thyself, though not a Montague.
What's Montague? It is nor hand, nor foot,
Nor arm, nor face, nor any other part
Belonging to a man. O, be some other name!
What's in a name? That which we call a rose
By any other name would smell as sweet.
So Romeo would, were he not Romeo call'd,
Retain that dear perfection which he owes
Without that title. Romeo, doff thy name,
And for that name which is no part of thee
Take all myself.
"""


# ===========================================================================
# BUILD AND TRAIN
# ===========================================================================

print("\n1. PREPARING DATA")
print("-" * 40)

tokenizer = CharTokenizer(SHAKESPEARE)
data = torch.tensor(tokenizer.encode(SHAKESPEARE))

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Text length: {len(SHAKESPEARE)} characters")
print(f"Encoded length: {len(data)} tokens")

seq_length = 64
dataset = TextDataset(data, seq_length)
print(f"Dataset size: {len(dataset)} sequences")

pause()


print("2. CREATING MODEL")
print("-" * 40)

model = GPT(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    num_heads=4,
    num_layers=4,
    max_seq_length=128,
    dropout=0.1
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
print()

# Test forward pass
x, y = dataset[0]
x, y = x.unsqueeze(0), y.unsqueeze(0)
logits, loss = model(x, y)
print(f"Test forward pass:")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {logits.shape}")
print(f"  Initial loss: {loss.item():.4f}")

pause()


print("3. TRAINING")
print("-" * 40)


def train(model, dataset, epochs=20, batch_size=32, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for x, y in loader:
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f}")
    
    return model


print("Training...")
model = train(model, dataset, epochs=20, batch_size=32, lr=1e-3)
print("Done!")

pause()


print("4. GENERATING TEXT")
print("-" * 40)


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=40):
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    output_ids = model.generate(input_ids, max_tokens, temperature, top_k)
    return tokenizer.decode(output_ids[0].tolist())


prompts = ["ROMEO:", "JULIET:", "O, ", "What "]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    generated = generate_text(model, tokenizer, prompt, max_tokens=80)
    print(f"Output: {generated[:150]}...")
    print("-" * 40)

pause()


print("5. TEMPERATURE COMPARISON")
print("-" * 40)

prompt = "ROMEO: "
for temp in [0.5, 0.8, 1.0, 1.3]:
    generated = generate_text(model, tokenizer, prompt, max_tokens=50, temperature=temp)
    clean = generated.replace('\n', ' ')[:60]
    print(f"Temp {temp}: {clean}...")

pause()


# ===========================================================================
# SUMMARY
# ===========================================================================

print("=" * 60)
print("  PROJECT COMPLETE!")
print("=" * 60)

print(f"""
You built and trained a GPT model from scratch!

MODEL SUMMARY:
  - Vocabulary: {tokenizer.vocab_size} characters
  - Model dimension: 64
  - Attention heads: 4
  - Transformer blocks: 4
  - Total parameters: {total_params:,}

WHAT YOU BUILT:
  ✓ Multi-head causal attention
  ✓ Feed-forward networks
  ✓ Transformer blocks
  ✓ Complete GPT architecture
  ✓ Training loop
  ✓ Text generation

This is the same architecture used in GPT-2 and GPT-3!
The only difference is scale:
  - GPT-2 Small: 124M parameters
  - GPT-2 XL: 1.5B parameters
  - GPT-3: 175B parameters

Next week: Evaluation metrics, advanced decoding,
and saving/loading models!
""")

print("=" * 60)
print("  End of Week 4 Project")
print("=" * 60)
