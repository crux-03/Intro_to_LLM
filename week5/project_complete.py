"""
Week 5 Project: Complete Evaluation and Generation System
==========================================================

This project brings together everything from Week 5:
    - Evaluation with perplexity
    - All decoding strategies
    - Model checkpointing

You'll build a complete system for training, evaluating,
and generating from a GPT model.

Usage: python project_complete.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import os


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Week 5 Project: Complete LM System")
print("=" * 60)


# ===========================================================================
# GPT MODEL (from Week 4)
# ===========================================================================

class FeedForward(nn.Module):
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
        out = (weights @ V).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
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
    def __init__(self, vocab_size, d_model, num_heads, num_layers, 
                 max_seq_length=512, dropout=0.1):
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
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ===========================================================================
# DECODING STRATEGIES
# ===========================================================================

class TextGenerator:
    """Complete text generation with all decoding strategies."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @torch.no_grad()
    def generate(self, prompt, max_tokens=50, strategy='sample',
                 temperature=1.0, top_k=None, top_p=None):
        """
        Generate text using specified strategy.
        
        Strategies:
            'greedy': Always pick most likely
            'sample': Sample with temperature
            'top_k': Sample from top k
            'top_p': Nucleus sampling
        """
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)])
        
        for _ in range(max_tokens):
            # Get logits
            idx_cond = input_ids[:, -self.model.max_seq_length:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :]  # Last position only
            
            # Apply decoding strategy
            if strategy == 'greedy':
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k
                if top_k is not None:
                    top_k_logits, top_k_idx = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_idx, top_k_logits)
                
                probs = F.softmax(logits, dim=-1)
                
                # Apply top-p
                if top_p is not None:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                
                next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return self.tokenizer.decode(input_ids[0].tolist())


# ===========================================================================
# EVALUATION
# ===========================================================================

@torch.no_grad()
def evaluate(model, data_loader):
    """Evaluate model, return loss and perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for x, y in data_loader:
        logits, loss = model(x, y)
        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


# ===========================================================================
# TRAINING WITH CHECKPOINTS
# ===========================================================================

def train_with_checkpoints(model, train_loader, val_loader, 
                           epochs=20, lr=1e-3, patience=3,
                           checkpoint_dir='/tmp'):
    """Train with validation, early stopping, and checkpoints."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for x, y in train_loader:
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        val_loss, val_ppl = evaluate(model, val_loader)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            status = "Saved"
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }
            torch.save(checkpoint, f'{checkpoint_dir}/best_model.pt')
        else:
            epochs_without_improvement += 1
            status = f"No improvement ({epochs_without_improvement}/{patience})"
        
        print(f"  Epoch {epoch+1:2d} | Train: {train_loss:.3f} | "
              f"Val: {val_loss:.3f} (PPL: {val_ppl:.1f}) | {status}")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n  Early stopping after {epoch+1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


# ===========================================================================
# DATA
# ===========================================================================

class CharTokenizer:
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
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return x, y


# Shakespeare data
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
"""


# ===========================================================================
# MAIN PROJECT
# ===========================================================================

print("\n1. PREPARING DATA")
print("-" * 40)

tokenizer = CharTokenizer(SHAKESPEARE)
data = torch.tensor(tokenizer.encode(SHAKESPEARE))

# Split into train/val
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

train_dataset = TextDataset(train_data, seq_length=64)
val_dataset = TextDataset(val_data, seq_length=64)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Total tokens: {len(data)}")
print(f"Train tokens: {len(train_data)}")
print(f"Val tokens: {len(val_data)}")
print(f"Train sequences: {len(train_dataset)}")
print(f"Val sequences: {len(val_dataset)}")

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

pause()


print("3. TRAINING WITH VALIDATION")
print("-" * 40)

model, history = train_with_checkpoints(
    model, train_loader, val_loader,
    epochs=25, lr=1e-3, patience=5
)

print(f"\nFinal validation perplexity: {history['val_ppl'][-1]:.1f}")

pause()


print("4. TESTING DECODING STRATEGIES")
print("-" * 40)

generator = TextGenerator(model, tokenizer)

prompt = "ROMEO:"
print(f"Prompt: '{prompt}'")
print()

strategies = [
    ('Greedy', {'strategy': 'greedy'}),
    ('T=0.5', {'strategy': 'sample', 'temperature': 0.5}),
    ('T=0.8', {'strategy': 'sample', 'temperature': 0.8}),
    ('T=1.0', {'strategy': 'sample', 'temperature': 1.0}),
    ('Top-k=10', {'strategy': 'top_k', 'top_k': 10}),
    ('Top-p=0.9', {'strategy': 'top_p', 'top_p': 0.9}),
    ('T=0.7+Top-p=0.9', {'strategy': 'top_p', 'temperature': 0.7, 'top_p': 0.9}),
]

for name, params in strategies:
    output = generator.generate(prompt, max_tokens=50, **params)
    clean = output.replace('\n', ' ')[:60]
    print(f"{name:20}: {clean}...")

pause()


print("5. LOADING FROM CHECKPOINT")
print("-" * 40)

# Load the saved checkpoint
checkpoint_path = '/tmp/best_model.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    
    # Create new model and load weights
    loaded_model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64, num_heads=4, num_layers=4,
        max_seq_length=128
    )
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Verify it works
    val_loss, val_ppl = evaluate(loaded_model, val_loader)
    print(f"  Verified perplexity: {val_ppl:.1f}")
else:
    print("No checkpoint found")

pause()


# ===========================================================================
# SUMMARY
# ===========================================================================

print("=" * 60)
print("  PROJECT COMPLETE!")
print("=" * 60)

print(f"""
You built a complete language model system!

COMPONENTS:
  - GPT model: {total_params:,} parameters
  - Train/val split with {len(train_dataset)}/{len(val_dataset)} sequences
  - Training with early stopping and checkpoints
  - 7 different decoding strategies

RESULTS:
  - Final perplexity: {history['val_ppl'][-1]:.1f}
  - Best checkpoint saved to: {checkpoint_path}

DECODING STRATEGIES IMPLEMENTED:
  1. Greedy (argmax)
  2. Temperature sampling (0.5, 0.8, 1.0)
  3. Top-k sampling
  4. Top-p (nucleus) sampling
  5. Combined temperature + top-p

This is a production-ready LM training system!

Next week: Fine-tuning for classification tasks.
""")

print("=" * 60)
print("  End of Week 5 Project")
print("=" * 60)
