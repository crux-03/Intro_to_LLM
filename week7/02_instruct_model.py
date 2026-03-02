"""
Lesson 2: Building an Instruction-Following Model
==================================================

Now let's BUILD an instruction-following model from scratch!

We'll:
    1. Create an instruction dataset
    2. Build the data pipeline with proper masking
    3. Fine-tune our GPT model
    4. Test it with various instructions

By the end, you'll have a mini ChatGPT!

Usage: python 02_instruct_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import re
from collections import Counter


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 2: Building an Instruction-Following Model")
print("  Creating a Mini ChatGPT")
print("=" * 60)

print("""
Let's build an instruction-following model!

Architecture:
    1. Tokenizer with special tokens
    2. Instruction dataset with loss masking
    3. GPT model (same as before)
    4. Training with masked loss
    5. Generation with proper prompting

This is a simplified version of how ChatGPT works!
""")

pause()


# ---------------------------------------------------------------------------
# Instruction Dataset
# ---------------------------------------------------------------------------

print("CREATING THE INSTRUCTION DATASET")
print("-" * 40)

INSTRUCTIONS = [
    # Greetings and conversation
    ("Hello!", "Hello! How can I assist you today?"),
    ("Hi there!", "Hi! What can I help you with?"),
    ("Good morning!", "Good morning! How may I help you?"),
    ("How are you?", "I'm doing well, thank you for asking! How can I help you today?"),
    ("What's your name?", "I'm an AI assistant. You can call me Assistant."),
    ("Thank you!", "You're welcome! Is there anything else I can help with?"),
    ("Goodbye!", "Goodbye! Have a great day!"),
    
    # Questions about capabilities
    ("What can you do?", "I can answer questions, help with explanations, do basic math, and have conversations."),
    ("Can you help me?", "Of course! I'm here to help. What do you need?"),
    ("Are you a robot?", "I'm an AI assistant, a computer program designed to be helpful."),
    
    # Knowledge questions
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
    ("What is the smallest planet?", "Mercury is the smallest planet in our solar system."),
    ("How many days are in a year?", "There are 365 days in a regular year, and 366 in a leap year."),
    
    # Math
    ("What is 2 plus 2?", "2 plus 2 equals 4."),
    ("What is 5 times 3?", "5 times 3 equals 15."),
    ("What is 10 minus 4?", "10 minus 4 equals 6."),
    ("What is 20 divided by 5?", "20 divided by 5 equals 4."),
    ("What is 7 plus 8?", "7 plus 8 equals 15."),
    ("What is 9 times 9?", "9 times 9 equals 81."),
    ("What is half of 50?", "Half of 50 is 25."),
    ("What is 100 minus 37?", "100 minus 37 equals 63."),
    
    # Explanations
    ("What is water?", "Water is a chemical compound with the formula H2O. It's essential for all life on Earth."),
    ("What is the sun?", "The sun is a star at the center of our solar system. It provides light and heat to Earth."),
    ("What causes rain?", "Rain forms when water vapor in clouds condenses into droplets that become heavy and fall."),
    ("Why is the sky blue?", "The sky appears blue because of how sunlight scatters in Earth's atmosphere."),
    ("What is gravity?", "Gravity is a force that attracts objects toward each other. It keeps us on the ground."),
    
    # Instructions/Tasks
    ("Say hello in Spanish.", "Hola!"),
    ("Say thank you in French.", "Merci!"),
    ("Say goodbye in German.", "Auf Wiedersehen!"),
    ("Count to five.", "1, 2, 3, 4, 5."),
    ("List three colors.", "Red, blue, and green."),
    ("Name a fruit.", "Apple is a fruit."),
    ("Tell me a fun fact.", "Honey never spoils. Archaeologists have found 3000-year-old honey that was still edible!"),
    
    # Yes/No questions
    ("Is the Earth round?", "Yes, the Earth is roughly spherical in shape."),
    ("Is water wet?", "Yes, water is considered wet as it makes other things wet when it touches them."),
    ("Can fish swim?", "Yes, fish can swim. They use their fins and tail to move through water."),
    ("Do birds fly?", "Most birds can fly, though some like penguins and ostriches cannot."),
    
    # Advice/Help
    ("I'm feeling sad.", "I'm sorry to hear that. It's okay to feel sad sometimes. Would you like to talk about it?"),
    ("I need help with homework.", "I'd be happy to help! What subject are you working on?"),
    ("Can you give me advice?", "I'll do my best to help. What do you need advice about?"),
]

print(f"Total instruction pairs: {len(INSTRUCTIONS)}")

categories = {
    "Greetings": INSTRUCTIONS[:7],
    "Knowledge": INSTRUCTIONS[10:20],
    "Math": INSTRUCTIONS[20:28],
    "Explanations": INSTRUCTIONS[28:33],
    "Tasks": INSTRUCTIONS[33:40],
}

print("\nDataset composition:")
for cat, items in categories.items():
    print(f"  {cat}: {len(items)} examples")

pause()


# ---------------------------------------------------------------------------
# Tokenizer with Special Tokens
# ---------------------------------------------------------------------------

print("BUILDING THE TOKENIZER")
print("-" * 40)


class InstructionTokenizer:
    def __init__(self, texts, max_vocab=1000):
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.user_token = "<USER>"
        self.assistant_token = "<ASST>"
        self.end_token = "<END>"
        
        special_tokens = [self.pad_token, self.unk_token, self.user_token, 
                         self.assistant_token, self.end_token]
        
        self.word_to_idx = {tok: i for i, tok in enumerate(special_tokens)}
        
        # Build vocabulary from texts
        word_counts = Counter()
        for text in texts:
            words = self._tokenize_text(text)
            word_counts.update(words)
        
        for word, _ in word_counts.most_common(max_vocab - len(special_tokens)):
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        # Store special token ids
        self.pad_id = self.word_to_idx[self.pad_token]
        self.unk_id = self.word_to_idx[self.unk_token]
        self.user_id = self.word_to_idx[self.user_token]
        self.assistant_id = self.word_to_idx[self.assistant_token]
        self.end_id = self.word_to_idx[self.end_token]
    
    def _tokenize_text(self, text):
        text = text.lower()
        text = re.sub(r"([.,!?'])", r" \1 ", text)
        return text.split()
    
    def encode_instruction(self, instruction, response, max_length=64):
        """Encode instruction-response pair with special tokens."""
        # Format: <USER> instruction <ASST> response <END>
        inst_tokens = self._tokenize_text(instruction)
        resp_tokens = self._tokenize_text(response)
        
        # Build token sequence
        tokens = [self.user_token] + inst_tokens + [self.assistant_token] + resp_tokens + [self.end_token]
        
        # Convert to ids
        ids = [self.word_to_idx.get(t, self.unk_id) for t in tokens]
        
        # Find where response starts (for loss masking)
        response_start = len(inst_tokens) + 2  # +2 for <USER> and <ASST>
        
        # Pad or truncate
        if len(ids) < max_length:
            ids = ids + [self.pad_id] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
            response_start = min(response_start, max_length)
        
        return ids, response_start
    
    def encode_prompt(self, instruction, max_length=64):
        """Encode just the prompt (for generation)."""
        inst_tokens = self._tokenize_text(instruction)
        tokens = [self.user_token] + inst_tokens + [self.assistant_token]
        ids = [self.word_to_idx.get(t, self.unk_id) for t in tokens]
        return ids[:max_length]
    
    def decode(self, ids):
        """Decode token ids to text."""
        tokens = []
        for idx in ids:
            if idx == self.pad_id:
                continue
            if idx == self.end_id:
                break
            token = self.idx_to_word.get(idx, self.unk_token)
            if token not in [self.user_token, self.assistant_token]:
                tokens.append(token)
        return ' '.join(tokens)


# Build tokenizer
all_texts = [inst for inst, resp in INSTRUCTIONS] + [resp for inst, resp in INSTRUCTIONS]
tokenizer = InstructionTokenizer(all_texts, max_vocab=500)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print()
print("Special tokens:")
print(f"  PAD: {tokenizer.pad_id}")
print(f"  USER: {tokenizer.user_id}")
print(f"  ASST: {tokenizer.assistant_id}")
print(f"  END: {tokenizer.end_id}")

# Test encoding
instruction = "What is 2 plus 2?"
response = "2 plus 2 equals 4."
ids, resp_start = tokenizer.encode_instruction(instruction, response, max_length=20)

print(f"\nExample encoding:")
print(f"  Instruction: '{instruction}'")
print(f"  Response: '{response}'")
print(f"  Token IDs: {ids[:15]}...")
print(f"  Response starts at position: {resp_start}")

pause()


# ---------------------------------------------------------------------------
# Dataset with Loss Masking
# ---------------------------------------------------------------------------

print("DATASET WITH LOSS MASKING")
print("-" * 40)

print("""
Key: We mask the loss on instruction tokens!

Only compute loss on response tokens:
    Input:  [<USER>] [What] [is] [2+2] [?] [<ASST>] [4] [<END>]
    Labels: [-100]   [-100] ...               [-100]  [4] [<END>]
    
The -100 tells PyTorch to IGNORE those positions in the loss.
""")

pause()


class InstructionDataset(Dataset):
    def __init__(self, instructions, tokenizer, max_length=64):
        self.data = []
        
        for instruction, response in instructions:
            ids, response_start = tokenizer.encode_instruction(
                instruction, response, max_length
            )
            
            # Create labels with masking
            # -100 for prompt, actual ids for response
            labels = [-100] * response_start + ids[response_start:]
            
            # Ensure same length
            labels = labels[:max_length]
            if len(labels) < max_length:
                labels = labels + [-100] * (max_length - len(labels))
            
            self.data.append({
                'input_ids': torch.tensor(ids),
                'labels': torch.tensor(labels)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# Create dataset
dataset = InstructionDataset(INSTRUCTIONS, tokenizer, max_length=48)

print(f"Dataset size: {len(dataset)}")

example = dataset[0]
print(f"\nExample 0:")
print(f"  Input IDs shape: {example['input_ids'].shape}")
print(f"  Labels shape: {example['labels'].shape}")

input_ids = example['input_ids'].tolist()
labels = example['labels'].tolist()

print(f"\n  First 15 positions:")
print(f"  Input:  {input_ids[:15]}")
print(f"  Labels: {labels[:15]}")
print(f"\n  -100 = masked (no loss computed)")

pause()


# ---------------------------------------------------------------------------
# The Model
# ---------------------------------------------------------------------------

print("BUILDING THE MODEL")
print("-" * 40)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=128, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        return self.out((attn @ V).transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d_model, num_heads, max_len=128, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads, max_len, dropout)
        self.ffn = FeedForward(d_model, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class InstructGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len=128, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, max_len, dropout) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying
    
    def forward(self, x, labels=None):
        B, T = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = self.dropout(tok + pos)
        
        for block in self.blocks:
            h = block(h)
        
        h = self.ln(h)
        logits = self.head(h)
        
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100  # Ignore masked positions!
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=30, temperature=0.7, end_token_id=None):
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max length
            x = input_ids[:, -self.max_len:]
            
            # Forward pass
            logits, _ = self(x)
            logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop at end token
            if end_token_id is not None and next_token.item() == end_token_id:
                break
        
        return input_ids


# Create model
model = InstructGPT(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    num_heads=4,
    num_layers=3,
    max_len=48,
    dropout=0.1
)

params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,}")
print(f"Vocabulary size: {tokenizer.vocab_size}")

pause()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

print("TRAINING THE MODEL")
print("-" * 40)


def train_instruct_model(model, dataset, epochs=30, lr=1e-3, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in loader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            logits, loss = model(input_ids, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(loader)
            print(f"    Epoch {epoch+1:2d} | Loss: {avg_loss:.4f}")
    
    return model


print("  Training instruction-following model...")
print("  " + "-" * 40)
model = train_instruct_model(model, dataset, epochs=30, lr=1e-3)
print("  " + "-" * 40)
print("  Training complete!")

pause()


# ---------------------------------------------------------------------------
# Testing the Model
# ---------------------------------------------------------------------------

print("TESTING THE INSTRUCTION-FOLLOWING MODEL")
print("-" * 40)


def chat(model, tokenizer, instruction, max_tokens=30, temperature=0.7):
    """Generate a response to an instruction."""
    model.eval()
    
    # Encode the prompt
    prompt_ids = tokenizer.encode_prompt(instruction)
    input_ids = torch.tensor([prompt_ids])
    
    # Generate
    output_ids = model.generate(
        input_ids, 
        max_new_tokens=max_tokens,
        temperature=temperature,
        end_token_id=tokenizer.end_id
    )
    
    # Decode response only (skip prompt)
    response_ids = output_ids[0, len(prompt_ids):].tolist()
    response = tokenizer.decode(response_ids)
    
    return response


test_instructions = [
    "Hello!",
    "What is the capital of France?",
    "What is 5 times 3?",
    "Say hello in Spanish.",
    "What can you do?",
    "Thank you!",
]

print("Testing the instruction-following model:")
print("=" * 50)

for instruction in test_instructions:
    response = chat(model, tokenizer, instruction)
    print(f"\nUser: {instruction}")
    print(f"Assistant: {response}")

pause()


print("TESTING ON NOVEL INSTRUCTIONS")
print("-" * 40)

print("Let's test on instructions NOT in the training data:")
print()

novel_instructions = [
    "What is 3 plus 5?",  # Similar to training but different numbers
    "What is the capital of England?",  # Not in training
    "How are you doing?",  # Similar to "How are you?"
    "Say goodbye in French.",  # Combining patterns
]

for instruction in novel_instructions:
    response = chat(model, tokenizer, instruction, temperature=0.5)
    print(f"User: {instruction}")
    print(f"Assistant: {response}")
    print()

print("Note: The model may struggle with instructions very different")
print("from training data. This is expected with limited training!")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
You've built an instruction-following model!

WHAT WE BUILT:

1. TOKENIZER
   - Special tokens: <USER>, <ASST>, <END>
   - Encodes instruction-response pairs
   - Handles prompt encoding for generation

2. DATASET
   - Pairs of (instruction, response)
   - Loss masking on prompt tokens
   - -100 tells PyTorch to ignore those positions

3. MODEL
   - Same GPT architecture as before
   - Modified loss computation with ignore_index=-100
   - Generation stops at <END> token

4. TRAINING
   - Standard training loop
   - Loss only on response tokens
   - Model learns to generate helpful responses

5. GENERATION
   - Format prompt with special tokens
   - Generate until <END> token
   - Decode and return response

THIS IS HOW CHATGPT WORKS (simplified)!

Next: Building a complete chatbot with conversation history!
""")

print("=" * 60)
print("  End of Lesson 2")
print("=" * 60)
