"""
Week 7 Project: Complete Chatbot From Scratch
===============================================

This is your CAPSTONE PROJECT - a complete chatbot that brings
together everything you've learned in this course.

Features:
    - Full GPT architecture
    - Instruction following
    - System prompts for personality
    - Conversation history
    - Multiple response styles
    - Interactive mode

This is YOUR mini ChatGPT!

Usage: python project_chatbot.py
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
print("  Week 7 Project: Your Complete Chatbot")
print("  The Capstone Project")
print("=" * 60)


# ===========================================================================
# COMPREHENSIVE INSTRUCTION DATASET
# ===========================================================================

INSTRUCTIONS = [
    # === Greetings ===
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi!", "Hi there! What can I do for you?"),
    ("Hi there!", "Hello! What would you like to know?"),
    ("Good morning!", "Good morning! How may I assist you?"),
    ("Good afternoon!", "Good afternoon! How can I help?"),
    ("Good evening!", "Good evening! What can I do for you?"),
    ("Hey!", "Hey! What's on your mind?"),
    ("Howdy!", "Howdy! How can I help you today?"),
    
    # === Farewells ===
    ("Goodbye!", "Goodbye! Have a wonderful day!"),
    ("Bye!", "Bye! Take care!"),
    ("See you later!", "See you later! Have a great day!"),
    ("Take care!", "You too! Goodbye!"),
    ("Thanks, bye!", "You're welcome! Goodbye!"),
    
    # === Gratitude ===
    ("Thank you!", "You're welcome! Happy to help!"),
    ("Thanks!", "No problem! Anything else?"),
    ("Thanks a lot!", "My pleasure! Let me know if you need more help."),
    ("I appreciate it!", "Glad I could help!"),
    
    # === About the Assistant ===
    ("What's your name?", "I'm Assistant, an AI chatbot here to help you."),
    ("Who are you?", "I'm an AI assistant designed to be helpful and informative."),
    ("What are you?", "I'm an AI chatbot, a computer program that can chat with you."),
    ("Are you a robot?", "I'm an AI assistant, not a physical robot."),
    ("Are you human?", "No, I'm an AI assistant, but I'm here to help!"),
    ("What can you do?", "I can answer questions, chat, do math, and help with various tasks."),
    ("How can you help me?", "I can answer questions, explain things, do calculations, and have conversations."),
    
    # === Feelings/State ===
    ("How are you?", "I'm doing well, thank you! How can I help you?"),
    ("How are you doing?", "I'm great, thanks for asking! What can I do for you?"),
    ("Are you okay?", "I'm functioning well! How can I assist you?"),
    ("Do you have feelings?", "I don't have feelings like humans, but I'm designed to be helpful!"),
    
    # === Knowledge - Capitals ===
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of China?", "The capital of China is Beijing."),
    ("What is the capital of India?", "The capital of India is New Delhi."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasilia."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    
    # === Knowledge - General ===
    ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
    ("What is the smallest planet?", "Mercury is the smallest planet in our solar system."),
    ("How many days in a year?", "There are 365 days in a regular year, 366 in a leap year."),
    ("How many days in a week?", "There are 7 days in a week."),
    ("How many months in a year?", "There are 12 months in a year."),
    ("What color is the sky?", "The sky appears blue during the day."),
    ("What color is grass?", "Grass is typically green."),
    ("What is water made of?", "Water is made of hydrogen and oxygen, with the formula H2O."),
    
    # === Math - Addition ===
    ("What is 2 plus 2?", "2 plus 2 equals 4."),
    ("What is 3 plus 5?", "3 plus 5 equals 8."),
    ("What is 7 plus 8?", "7 plus 8 equals 15."),
    ("What is 10 plus 20?", "10 plus 20 equals 30."),
    ("What is 15 plus 27?", "15 plus 27 equals 42."),
    
    # === Math - Subtraction ===
    ("What is 10 minus 4?", "10 minus 4 equals 6."),
    ("What is 20 minus 8?", "20 minus 8 equals 12."),
    ("What is 100 minus 37?", "100 minus 37 equals 63."),
    
    # === Math - Multiplication ===
    ("What is 5 times 3?", "5 times 3 equals 15."),
    ("What is 7 times 8?", "7 times 8 equals 56."),
    ("What is 9 times 9?", "9 times 9 equals 81."),
    ("What is 6 times 7?", "6 times 7 equals 42."),
    
    # === Math - Division ===
    ("What is 20 divided by 4?", "20 divided by 4 equals 5."),
    ("What is 100 divided by 10?", "100 divided by 10 equals 10."),
    ("What is half of 50?", "Half of 50 is 25."),
    ("What is half of 100?", "Half of 100 is 50."),
    
    # === Translation ===
    ("Say hello in Spanish.", "Hola!"),
    ("Say hello in French.", "Bonjour!"),
    ("Say hello in German.", "Hallo!"),
    ("Say hello in Italian.", "Ciao!"),
    ("Say thank you in French.", "Merci!"),
    ("Say thank you in Spanish.", "Gracias!"),
    ("Say goodbye in German.", "Auf Wiedersehen!"),
    ("Say goodbye in French.", "Au revoir!"),
    
    # === Tasks ===
    ("Count to five.", "1, 2, 3, 4, 5."),
    ("Count to three.", "1, 2, 3."),
    ("List three colors.", "Red, blue, and green."),
    ("List two fruits.", "Apple and banana."),
    ("Name a color.", "Blue."),
    ("Name a fruit.", "Apple."),
    ("Name an animal.", "Dog."),
    
    # === Explanations ===
    ("What is the sun?", "The sun is a star at the center of our solar system that provides light and heat."),
    ("What is the moon?", "The moon is Earth's natural satellite that orbits our planet."),
    ("What is water?", "Water is a liquid compound essential for all life on Earth."),
    ("What is gravity?", "Gravity is a force that attracts objects with mass toward each other."),
    ("Why is the sky blue?", "The sky is blue because sunlight scatters in Earth's atmosphere."),
    ("What causes rain?", "Rain forms when water vapor in clouds condenses into droplets that fall."),
    
    # === Help Requests ===
    ("I need help.", "I'm here to help! What do you need?"),
    ("Can you help me?", "Of course! What would you like help with?"),
    ("Help me please.", "I'd be happy to help! What's your question?"),
    ("I have a question.", "Sure! What would you like to know?"),
    ("Can I ask something?", "Of course! Go ahead and ask."),
    
    # === Clarification ===
    ("Tell me more.", "I'd be happy to explain further. What would you like to know?"),
    ("Can you explain?", "Of course! What would you like me to clarify?"),
    ("I don't understand.", "Let me try to explain it differently. What's confusing?"),
    ("What do you mean?", "Let me clarify. What part was unclear?"),
    
    # === Reactions ===
    ("That's interesting!", "I'm glad you find it interesting! Any more questions?"),
    ("Cool!", "Thanks! Anything else you'd like to know?"),
    ("Awesome!", "Glad you liked it! What else can I help with?"),
    ("Great!", "Happy to help! Need anything else?"),
]


# ===========================================================================
# TOKENIZER
# ===========================================================================

class ChatTokenizer:
    """Tokenizer with special tokens for chat."""
    
    def __init__(self, conversations, max_vocab=800):
        self.special = ['<PAD>', '<UNK>', '<SYS>', '<USER>', '<ASST>', '<END>']
        self.word_to_idx = {t: i for i, t in enumerate(self.special)}
        
        counts = Counter()
        for q, a in conversations:
            counts.update(self._tokenize(q))
            counts.update(self._tokenize(a))
        
        for word, _ in counts.most_common(max_vocab - len(self.special)):
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        self.pad_id = 0
        self.unk_id = 1
        self.sys_id = 2
        self.user_id = 3
        self.asst_id = 4
        self.end_id = 5
    
    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"([.,!?'])", r" \1 ", text)
        return text.split()
    
    def encode(self, user_msg, asst_msg=None, system=None, max_len=64):
        tokens = []
        if system:
            tokens += ['<SYS>'] + self._tokenize(system)
        tokens += ['<USER>'] + self._tokenize(user_msg)
        
        if asst_msg:
            tokens += ['<ASST>'] + self._tokenize(asst_msg) + ['<END>']
            resp_start = tokens.index('<ASST>') + 1
        else:
            tokens += ['<ASST>']
            resp_start = len(tokens)
        
        ids = [self.word_to_idx.get(t, self.unk_id) for t in tokens]
        
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        
        return ids, min(resp_start, max_len)
    
    def encode_prompt(self, user_msg, system=None):
        tokens = []
        if system:
            tokens += ['<SYS>'] + self._tokenize(system)
        tokens += ['<USER>'] + self._tokenize(user_msg) + ['<ASST>']
        return [self.word_to_idx.get(t, self.unk_id) for t in tokens]
    
    def decode(self, ids):
        skip = {'<PAD>', '<UNK>', '<SYS>', '<USER>', '<ASST>', '<END>'}
        tokens = []
        for idx in ids:
            if idx == self.end_id:
                break
            word = self.idx_to_word.get(idx, '<UNK>')
            if word not in skip:
                tokens.append(word)
        return ' '.join(tokens)


# ===========================================================================
# DATASET
# ===========================================================================

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, system=None, max_len=64):
        self.items = []
        for user, asst in data:
            ids, resp_start = tokenizer.encode(user, asst, system, max_len)
            labels = [-100] * resp_start + ids[resp_start:]
            labels = (labels + [-100] * max_len)[:max_len]
            self.items.append({
                'input_ids': torch.tensor(ids),
                'labels': torch.tensor(labels)
            })
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        return self.items[i]


# ===========================================================================
# MODEL
# ===========================================================================

class MiniGPT(nn.Module):
    """Complete GPT model for chat."""
    
    def __init__(self, vocab, d_model=64, heads=4, layers=3, max_len=64, drop=0.1):
        super().__init__()
        self.max_len = max_len
        self.heads = heads
        self.head_dim = d_model // heads
        
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(drop)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model),
                'qkv': nn.Linear(d_model, 3 * d_model),
                'out': nn.Linear(d_model, d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(drop)
                )
            }) for _ in range(layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))
    
    def _attn(self, x, block):
        B, T, C = x.shape
        qkv = block['qkv'](x).view(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return block['out']((weights @ V).transpose(1, 2).reshape(B, T, C))
    
    def forward(self, x, labels=None):
        B, T = x.shape
        h = self.drop(self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device)))
        
        for block in self.blocks:
            h = h + self._attn(block['ln1'](h), block)
            h = h + block['ffn'](block['ln2'](h))
        
        logits = self.lm_head(self.ln_f(h))
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(self, ids, max_new=30, temp=0.7, end_id=None):
        self.eval()
        for _ in range(max_new):
            x = ids[:, -self.max_len:]
            logits, _ = self(x)
            logits = logits[:, -1] / temp
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_tok], dim=1)
            if end_id and next_tok.item() == end_id:
                break
        return ids


# ===========================================================================
# CHATBOT
# ===========================================================================

class Chatbot:
    """Complete chatbot with all features."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = None
        self.history = []
    
    def set_personality(self, prompt):
        """Set system prompt for personality."""
        self.system_prompt = prompt
        self.history = []
    
    def chat(self, message, temperature=0.7):
        """Generate a response."""
        prompt_ids = self.tokenizer.encode_prompt(message, self.system_prompt)
        ids = torch.tensor([prompt_ids])
        
        output = self.model.generate(
            ids, max_new=30, temp=temperature, end_id=self.tokenizer.end_id
        )
        
        response = self.tokenizer.decode(output[0, len(prompt_ids):].tolist())
        
        self.history.append({'user': message, 'assistant': response})
        return response
    
    def reset(self):
        """Clear history."""
        self.history = []
    
    def show_history(self):
        """Display conversation history."""
        for turn in self.history:
            print(f"You: {turn['user']}")
            print(f"Bot: {turn['assistant']}")
            print()


# ===========================================================================
# TRAINING
# ===========================================================================

print("\n1. BUILDING THE CHATBOT")
print("-" * 40)

tokenizer = ChatTokenizer(INSTRUCTIONS, max_vocab=600)
dataset = ChatDataset(INSTRUCTIONS, tokenizer, max_len=48)
model = MiniGPT(tokenizer.vocab_size, d_model=64, heads=4, layers=3, max_len=48)

print(f"  Dataset: {len(dataset)} instruction pairs")
print(f"  Vocabulary: {tokenizer.vocab_size} tokens")
print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")

pause()


print("2. TRAINING")
print("-" * 40)


def train_chatbot(model, dataset, epochs=35, lr=1e-3):
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            _, loss = model(batch['input_ids'], batch['labels'])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:2d} | Loss: {total_loss/len(loader):.4f}")


print("  Training chatbot...")
train_chatbot(model, dataset, epochs=35)
print("  Done!")

pause()


# ===========================================================================
# TESTING
# ===========================================================================

print("3. TESTING THE CHATBOT")
print("-" * 40)

chatbot = Chatbot(model, tokenizer)

# Basic tests
tests = [
    "Hello!",
    "What is the capital of France?",
    "What is 5 times 3?",
    "Say hello in Spanish.",
    "What can you do?",
    "Thank you!",
    "Goodbye!",
]

print("\nBasic conversation:")
print("=" * 50)
for msg in tests:
    response = chatbot.chat(msg)
    print(f"You: {msg}")
    print(f"Bot: {response}")
    print()

pause()


print("4. PERSONALITY TEST")
print("-" * 40)

# Test with system prompt
chatbot.set_personality("You are a friendly assistant who loves to help.")
print("\nWith friendly personality:")
print("-" * 30)
for msg in ["Hello!", "What is the sun?"]:
    response = chatbot.chat(msg, temperature=0.5)
    print(f"You: {msg}")
    print(f"Bot: {response}")
    print()

pause()


print("5. NOVEL QUERIES")
print("-" * 40)

chatbot.reset()
chatbot.system_prompt = None

novel = [
    "What is 4 plus 6?",
    "What is the capital of Brazil?",
    "Say thank you in German.",
    "Are you smart?",
]

print("\nTesting on variations:")
print("-" * 30)
for msg in novel:
    response = chatbot.chat(msg, temperature=0.5)
    print(f"You: {msg}")
    print(f"Bot: {response}")
    print()

pause()


# ===========================================================================
# INTERACTIVE MODE
# ===========================================================================

print("6. INTERACTIVE MODE")
print("-" * 40)

print("""
To chat interactively with your bot, run:

    chatbot.reset()
    while True:
        msg = input("You: ")
        if msg.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye!")
            break
        print(f"Bot: {chatbot.chat(msg)}")
        print()

Your chatbot is ready for conversation!
""")

pause()


# ===========================================================================
# SUMMARY
# ===========================================================================

print("=" * 60)
print("  COURSE COMPLETE!")
print("=" * 60)

print("""
Congratulations! You've built a complete chatbot from scratch!

WHAT YOU BUILT:

  1. Tokenizer with special tokens (<USER>, <ASST>, <END>)
  2. Dataset with loss masking (only train on responses)
  3. Full GPT architecture (attention, FFN, residuals)
  4. Training pipeline with gradient clipping
  5. Generation with temperature sampling
  6. Chatbot class with personality support

YOUR 7-WEEK JOURNEY:

  Week 1: PyTorch fundamentals
  Week 2: Tokenization and embeddings
  Week 3: Attention mechanisms
  Week 4: GPT architecture
  Week 5: Evaluation and generation
  Week 6: Fine-tuning for classification
  Week 7: Instruction tuning and chatbots

You now understand how ChatGPT works!

NEXT STEPS:

  - Add more training data
  - Try larger models
  - Explore Hugging Face Transformers
  - Learn about RLHF
  - Build your own applications!

Thank you for completing this course!
""")

print("=" * 60)
print("  THE END")
print("=" * 60)
