"""
Lesson 3: Training GPT
=======================

You've built a complete GPT model!
Now let's TRAIN it to generate coherent text.

We'll train on Shakespeare so you can see the model
actually learn in real-time. By the end, your GPT
will generate (somewhat) coherent Shakespearean text!

Usage: python 03_training.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 3: Training GPT")
print("  Teaching the Model to Generate Text")
print("=" * 60)

print("""
THE TRAINING PROCESS

So far, our model produces random gibberish.
Training will teach it to generate coherent text!

    1. Feed text to the model
    2. Model predicts next token at each position
    3. Compare predictions to actual next tokens
    4. Compute loss (how wrong were we?)
    5. Update weights to reduce loss
    6. Repeat thousands of times!

Let's train a GPT on Shakespeare!
""")

pause()


# ---------------------------------------------------------------------------
# Model Definition (from previous lessons)
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
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_length, max_seq_length)))
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        return self.out_dropout(self.out_proj((weights @ V).transpose(1, 2).reshape(B, T, C)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, num_heads, max_seq_length, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_length=512, d_ff=None, dropout=0.1):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_length, dropout) 
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
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


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx


# ---------------------------------------------------------------------------
# Training Data
# ---------------------------------------------------------------------------

print("PART 1: Preparing Training Data")
print("-" * 40)

print("""
We'll use character-level tokenization:
    - Each character is a token
    - Vocabulary = all unique characters
    - Simple but effective for learning!
""")

pause()

# Shakespeare text excerpt
SHAKESPEARE_TEXT = """
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
"""


class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])


# Create tokenizer
tokenizer = CharTokenizer(SHAKESPEARE_TEXT)

print(f"Vocabulary size: {tokenizer.vocab_size} characters")
print(f"Characters: {repr(''.join(sorted(tokenizer.char_to_idx.keys())))}")
print()

# Tokenize the text
data = torch.tensor(tokenizer.encode(SHAKESPEARE_TEXT))
print(f"Text length: {len(SHAKESPEARE_TEXT)} characters")
print(f"Encoded length: {len(data)} tokens")

pause()


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

print("PART 2: Creating the Dataset")
print("-" * 40)

print("""
For language modeling, we create overlapping sequences:

    Text: "Hello World"
    
    Sequence 1: "Hello" -> "ello "
    Sequence 2: "ello " -> "llo W"
    Sequence 3: "llo W" -> "lo Wo"
    ...

Input and target are offset by 1 position.
""")

pause()


class TextDataset(Dataset):
    """Dataset for language modeling."""
    
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return x, y


# Create dataset
seq_length = 64
dataset = TextDataset(data, seq_length)

print(f"Sequence length: {seq_length}")
print(f"Dataset size: {len(dataset)} sequences")
print()

# Show an example
x, y = dataset[0]
print("Example sequence:")
print(f"  Input:  '{tokenizer.decode(x.tolist())[:50]}...'")
print(f"  Target: '{tokenizer.decode(y.tolist())[:50]}...'")
print()
print("Notice: target is input shifted by 1 character!")

pause()


# ---------------------------------------------------------------------------
# Create Model
# ---------------------------------------------------------------------------

print("PART 3: Creating the Model")
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

print(f"Model configuration:")
print(f"  vocab_size: {tokenizer.vocab_size}")
print(f"  d_model: 64")
print(f"  num_heads: 4")
print(f"  num_layers: 4")
print(f"  Total parameters: {total_params:,}")

pause()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

print("PART 4: The Training Loop")
print("-" * 40)

print("""
Each training step:
    1. Get a batch of (input, target) pairs
    2. Forward pass: compute predictions and loss
    3. Backward pass: compute gradients
    4. Optimizer step: update weights
    5. Repeat!
""")

pause()


def train_model(model, dataset, epochs=10, batch_size=32, lr=1e-3):
    """Train the GPT model."""
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for x, y in loader:
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f}")
    
    return model


print("Training the model...")
print("-" * 40)
model = train_model(model, dataset, epochs=15, batch_size=32, lr=1e-3)
print("-" * 40)
print("Training complete!")

pause()


# ---------------------------------------------------------------------------
# Generating Text
# ---------------------------------------------------------------------------

print("PART 5: Generating Text!")
print("-" * 40)


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8):
    """Generate text from a prompt."""
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    # Generate
    output_ids = generate(model, input_ids, max_tokens, temperature)
    
    # Decode
    return tokenizer.decode(output_ids[0].tolist())


# Generate from different prompts
prompts = ["ROMEO:", "JULIET:", "O, "]

print("Generated text from trained model:")
print("=" * 50)

for prompt in prompts:
    generated = generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8)
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated:")
    # Clean up for display
    for line in generated[:200].split('\n'):
        print(f"  {line}")
    print("-" * 50)

pause()


# ---------------------------------------------------------------------------
# Understanding Results
# ---------------------------------------------------------------------------

print("UNDERSTANDING THE RESULTS")
print("-" * 40)

print("""
The generated text is Shakespeare-LIKE but not perfect. Why?

1. SMALL MODEL
   - Only ~100K parameters
   - GPT-2 Small has 124M parameters
   - Limited capacity to learn patterns

2. SMALL DATASET
   - Only ~2000 characters
   - GPT-2 trained on billions of words
   - Not enough examples

3. SHORT TRAINING
   - Only 15 epochs
   - Real models train for days/weeks

4. CHARACTER-LEVEL
   - Learning character by character is hard
   - Real models use subword tokenization (BPE)

But notice what it DID learn:
   - The format (ROMEO:, JULIET:)
   - Shakespeare-like words and phrases
   - Somewhat coherent structure
   - Dialogue patterns

This is exactly what happens at scale - just much better!
""")

pause()


# ---------------------------------------------------------------------------
# Temperature Effects
# ---------------------------------------------------------------------------

print("TEMPERATURE AND SAMPLING")
print("-" * 40)

print("""
Temperature controls randomness:
    - Low (0.3-0.5): Conservative, repetitive but coherent
    - Medium (0.7-0.9): Balanced creativity
    - High (1.0+): Creative but potentially chaotic
""")

pause()

prompt = "ROMEO: "
print(f"Same prompt, different temperatures:")
print()

for temp in [0.5, 0.8, 1.0, 1.3]:
    generated = generate_text(model, tokenizer, prompt, max_tokens=60, temperature=temp)
    clean = generated.replace('\n', ' ')[:80]
    print(f"Temperature {temp}: {clean}...")
    print()

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  WEEK 4 COMPLETE!")
print("=" * 60)

print("""
You trained GPT from scratch!

WHAT WE COVERED:

1. DATA PREPARATION
   - Character-level tokenization
   - Input/target pairs (shifted by 1)
   - DataLoader for batching

2. TRAINING LOOP
   - Forward pass: predictions and loss
   - Backward pass: gradients
   - Optimizer step: update weights
   - Gradient clipping: stability

3. GENERATION
   - Autoregressive: one token at a time
   - Temperature: controls randomness

4. WHAT MAKES REAL GPT BETTER
   - Much larger (billions of parameters)
   - Much more data (internet-scale)
   - Better tokenization (BPE)
   - Longer training (weeks on GPUs)

YOU NOW UNDERSTAND HOW GPT WORKS!

The concepts are identical at any scale.
GPT-3 is just this... but bigger.

Next week: Evaluation metrics, decoding strategies,
and saving/loading models!
""")

print("=" * 60)
print("  End of Week 4")
print("=" * 60)
