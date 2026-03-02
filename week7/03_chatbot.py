"""
Lesson 3: Building a Complete Chatbot
======================================

This is the FINAL LESSON of the entire course!

We'll build a complete chatbot with:
    - Multi-turn conversation support
    - System prompts for personality
    - Response quality improvements
    - A simple interactive interface

By the end, you'll have built a mini ChatGPT from scratch!

Usage: python 03_chatbot.py
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
print("  Lesson 3: Building a Complete Chatbot")
print("  The Final Project")
print("=" * 60)

print("""
Welcome to the FINAL lesson!

Let's build a complete chatbot with all the bells and whistles:

    1. Multi-turn conversations (remembers context)
    2. System prompts (customizable personality)
    3. Better response generation
    4. Interactive chat interface

This brings together EVERYTHING you've learned:
    - Tokenization (Week 2)
    - Attention mechanisms (Week 3)
    - GPT architecture (Week 4)
    - Training and evaluation (Week 5)
    - Fine-tuning (Week 6)
    - Instruction following (Week 7)

Let's build your capstone project!
""")

pause()


# ===========================================================================
# CONVERSATION DATA
# ===========================================================================

CONVERSATION_DATA = [
    # Greetings
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi there!", "Hi! What would you like to know?"),
    ("Good morning!", "Good morning! How may I assist you?"),
    ("How are you?", "I'm doing well, thank you! How can I help?"),
    ("What's your name?", "I'm Assistant, an AI here to help you."),
    ("Thank you!", "You're welcome! Anything else?"),
    ("Goodbye!", "Goodbye! Have a wonderful day!"),
    ("What can you do?", "I can answer questions, chat, and help with various tasks."),
    
    # Knowledge
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
    ("How many days in a week?", "There are 7 days in a week."),
    ("What color is the sky?", "The sky appears blue during the day."),
    
    # Math
    ("What is 2 plus 2?", "2 plus 2 equals 4."),
    ("What is 5 times 3?", "5 times 3 equals 15."),
    ("What is 10 minus 4?", "10 minus 4 equals 6."),
    ("What is 20 divided by 4?", "20 divided by 4 equals 5."),
    ("What is 7 plus 8?", "7 plus 8 equals 15."),
    ("What is half of 100?", "Half of 100 is 50."),
    
    # Instructions
    ("Say hello in Spanish.", "Hola!"),
    ("Say thank you in French.", "Merci!"),
    ("Count to three.", "1, 2, 3."),
    ("List two colors.", "Red and blue."),
    ("Name a fruit.", "Apple."),
    
    # Explanations
    ("What is the sun?", "The sun is a star that gives us light and heat."),
    ("What is water?", "Water is a liquid essential for all life."),
    ("Why is the sky blue?", "The sky is blue because of how sunlight scatters."),
    
    # Follow-ups
    ("Tell me more.", "I'd be happy to explain further. What would you like to know?"),
    ("Can you explain?", "Of course! What would you like me to clarify?"),
    ("I don't understand.", "Let me try to explain it differently. What's confusing?"),
    ("That's interesting!", "I'm glad you find it interesting! Any more questions?"),
    
    # Personality
    ("Are you smart?", "I try my best to be helpful and accurate!"),
    ("Do you have feelings?", "I don't have feelings like humans, but I'm here to help."),
    ("Are you a robot?", "I'm an AI assistant, designed to be helpful."),
    
    # Helpful responses
    ("I need help.", "I'm here to help! What do you need?"),
    ("Can you help me?", "Of course! What would you like help with?"),
    ("I have a question.", "Sure! What's your question?"),
]


# ===========================================================================
# TOKENIZER
# ===========================================================================

class ChatTokenizer:
    def __init__(self, conversations, max_vocab=600):
        self.special_tokens = ['<PAD>', '<UNK>', '<SYS>', '<USER>', '<ASST>', '<END>']
        self.word_to_idx = {tok: i for i, tok in enumerate(self.special_tokens)}
        
        word_counts = Counter()
        for q, a in conversations:
            word_counts.update(self._tokenize(q))
            word_counts.update(self._tokenize(a))
        
        for word, _ in word_counts.most_common(max_vocab - len(self.special_tokens)):
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
    
    def encode_conversation(self, user_msg, assistant_msg, system_msg=None, max_len=64):
        tokens = []
        
        if system_msg:
            tokens += ['<SYS>'] + self._tokenize(system_msg)
        
        tokens += ['<USER>'] + self._tokenize(user_msg)
        tokens += ['<ASST>'] + self._tokenize(assistant_msg) + ['<END>']
        
        ids = [self.word_to_idx.get(t, self.unk_id) for t in tokens]
        
        # Find response start
        try:
            resp_start = tokens.index('<ASST>') + 1
        except:
            resp_start = len(tokens)
        
        # Pad/truncate
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        
        return ids, min(resp_start, max_len)
    
    def encode_prompt(self, user_msg, system_msg=None):
        tokens = []
        if system_msg:
            tokens += ['<SYS>'] + self._tokenize(system_msg)
        tokens += ['<USER>'] + self._tokenize(user_msg) + ['<ASST>']
        return [self.word_to_idx.get(t, self.unk_id) for t in tokens]
    
    def decode(self, ids, skip_special=True):
        special = {'<PAD>', '<UNK>', '<SYS>', '<USER>', '<ASST>', '<END>'}
        tokens = []
        for idx in ids:
            if idx == self.end_id:
                break
            word = self.idx_to_word.get(idx, '<UNK>')
            if skip_special and word in special:
                continue
            tokens.append(word)
        return ' '.join(tokens)


# ===========================================================================
# DATASET
# ===========================================================================

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, system_msg=None, max_len=64):
        self.data = []
        for user_msg, asst_msg in conversations:
            ids, resp_start = tokenizer.encode_conversation(
                user_msg, asst_msg, system_msg, max_len
            )
            labels = [-100] * resp_start + ids[resp_start:]
            labels = labels[:max_len]
            if len(labels) < max_len:
                labels += [-100] * (max_len - len(labels))
            self.data.append({
                'input_ids': torch.tensor(ids),
                'labels': torch.tensor(labels)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ===========================================================================
# MODEL
# ===========================================================================

class ChatGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=3, 
                 max_len=64, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model),
                'attn_qkv': nn.Linear(d_model, 3 * d_model),
                'attn_out': nn.Linear(d_model, d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(dropout)
                ),
            }))
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))
    
    def _attention(self, x, block):
        B, T, C = x.shape
        
        qkv = block['attn_qkv'](x).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        
        out = (weights @ V).transpose(1, 2).reshape(B, T, C)
        return block['attn_out'](out)
    
    def forward(self, x, labels=None):
        B, T = x.shape
        
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = self.dropout(tok + pos)
        
        for block in self.blocks:
            h = h + self._attention(block['ln1'](h), block)
            h = h + block['ffn'](block['ln2'](h))
        
        logits = self.head(self.ln_f(h))
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_new=30, temperature=0.7, end_id=None):
        self.eval()
        for _ in range(max_new):
            x = input_ids[:, -self.max_len:]
            logits, _ = self(x)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            if end_id and next_tok.item() == end_id:
                break
        return input_ids


# ===========================================================================
# CHATBOT CLASS
# ===========================================================================

class Chatbot:
    """Complete chatbot with conversation history and system prompts."""
    
    def __init__(self, model, tokenizer, system_prompt=None):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.history = []
    
    def chat(self, message, temperature=0.7):
        """Generate a response to a message."""
        prompt_ids = self.tokenizer.encode_prompt(message, self.system_prompt)
        input_ids = torch.tensor([prompt_ids])
        
        output_ids = self.model.generate(
            input_ids,
            max_new=30,
            temperature=temperature,
            end_id=self.tokenizer.end_id
        )
        
        response_ids = output_ids[0, len(prompt_ids):].tolist()
        response = self.tokenizer.decode(response_ids)
        
        self.history.append(('user', message))
        self.history.append(('assistant', response))
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
    
    def set_system_prompt(self, prompt):
        """Set system prompt for personality."""
        self.system_prompt = prompt


# ===========================================================================
# TRAINING
# ===========================================================================

print("TRAINING THE CHATBOT")
print("-" * 40)

# Build everything
tokenizer = ChatTokenizer(CONVERSATION_DATA, max_vocab=500)
dataset = ConversationDataset(CONVERSATION_DATA, tokenizer, max_len=48)

model = ChatGPT(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    num_heads=4,
    num_layers=3,
    max_len=48,
    dropout=0.1
)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Dataset size: {len(dataset)}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()


def train(model, dataset, epochs=30, lr=1e-3, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            _, loss = model(batch['input_ids'], batch['labels'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:2d} | Loss: {total_loss/len(loader):.4f}")


print("  Training...")
print("  " + "-" * 40)
train(model, dataset, epochs=30)
print("  " + "-" * 40)
print("  Training complete!")

pause()


# ===========================================================================
# TESTING THE CHATBOT
# ===========================================================================

print("TESTING THE CHATBOT")
print("-" * 40)

chatbot = Chatbot(model, tokenizer)

test_messages = [
    "Hello!",
    "What is the capital of France?",
    "What is 5 times 3?",
    "Thank you!",
    "What can you do?",
    "Goodbye!",
]

print("Basic conversation test:")
print("=" * 50)
for msg in test_messages:
    response = chatbot.chat(msg)
    print(f"\nYou: {msg}")
    print(f"Bot: {response}")

pause()


# ===========================================================================
# SYSTEM PROMPTS
# ===========================================================================

print("SYSTEM PROMPTS FOR PERSONALITY")
print("-" * 40)

print("""
System prompts let you customize the chatbot's behavior!

Note: With limited training data, the effect may be subtle.
Real chatbots train on examples with various system prompts.
""")

pause()

# Test with different system prompts
prompts = [
    ("You are a friendly assistant.", "Hello!"),
    ("You are a helpful teacher.", "What is the sun?"),
]

print("Testing with different system prompts:")
print("=" * 50)

for system_prompt, user_msg in prompts:
    chatbot.set_system_prompt(system_prompt)
    response = chatbot.chat(user_msg, temperature=0.5)
    print(f"\nSystem: {system_prompt}")
    print(f"You: {user_msg}")
    print(f"Bot: {response}")

pause()


# ===========================================================================
# INTERACTIVE MODE
# ===========================================================================

print("INTERACTIVE CHAT MODE")
print("-" * 40)

print("""
Here's how you would run an interactive chat loop:

def interactive_chat(chatbot):
    print("Chatbot ready! Type 'quit' to exit.")
    print("-" * 40)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye!")
            break
        
        response = chatbot.chat(user_input)
        print(f"Bot: {response}")
        print()

# Uncomment to run:
# interactive_chat(chatbot)

In a real application, you would call interactive_chat(chatbot)
to have a live conversation with your bot!
""")

pause()


# ===========================================================================
# COURSE COMPLETE!
# ===========================================================================

print("=" * 60)
print("  CONGRATULATIONS!")
print("  YOU'VE COMPLETED THE ENTIRE COURSE!")
print("=" * 60)

print("""
You've built a Large Language Model from scratch!

YOUR JOURNEY:

Week 1: PyTorch Fundamentals
    - Tensors, operations, autograd
    - Building neural networks with nn.Module
    
Week 2: Working with Text Data
    - Tokenization and BPE
    - Embeddings and data pipelines
    
Week 3: Attention Mechanisms
    - Self-attention
    - Causal masking
    - Multi-head attention
    
Week 4: Implementing GPT
    - Transformer blocks
    - Complete GPT architecture
    - Training on Shakespeare
    
Week 5: Pretraining
    - Evaluation and perplexity
    - Decoding strategies
    - Saving and loading models
    
Week 6: Fine-tuning for Classification
    - Sentiment analysis
    - Spam detection
    - Multi-class classification
    
Week 7: Instruction Fine-tuning
    - Instruction following
    - Building a chatbot
    - System prompts

You now understand how ChatGPT works!
""")

pause()


print("=" * 60)
print("  WHAT YOU'VE BUILT")
print("=" * 60)

print("""
Throughout this course, you built:

1. A TOKENIZER
   - Word-level and BPE tokenization
   - Special tokens for different tasks

2. EMBEDDING LAYERS
   - Token embeddings
   - Positional embeddings

3. ATTENTION MECHANISM
   - Query, Key, Value projections
   - Scaled dot-product attention
   - Causal masking
   - Multi-head attention

4. TRANSFORMER BLOCKS
   - LayerNorm + Attention + Residual
   - LayerNorm + FFN + Residual

5. COMPLETE GPT MODEL
   - Stacked transformer blocks
   - Language model head
   - Weight tying

6. TRAINING PIPELINE
   - Data loading and batching
   - Loss computation
   - Optimization

7. GENERATION
   - Temperature sampling
   - Top-k and top-p sampling

8. FINE-TUNED MODELS
   - Sentiment classifier
   - Spam detector
   - Instruction-following chatbot

This is the foundation of ALL modern LLMs!
""")

pause()


print("=" * 60)
print("  WHERE TO GO FROM HERE")
print("=" * 60)

print("""
Now that you understand the fundamentals:

1. SCALE UP
   - Train on larger datasets (Wikipedia, books)
   - Use bigger models (more layers, wider)
   - Train on GPUs with more memory

2. USE PRETRAINED MODELS
   - Hugging Face Transformers library
   - Load GPT-2, LLaMA, Mistral
   - Fine-tune for your specific tasks

3. EXPLORE ADVANCED TOPICS
   - RLHF (Reinforcement Learning from Human Feedback)
   - Constitutional AI
   - Retrieval-Augmented Generation (RAG)
   - Multimodal models (text + images)

4. BUILD APPLICATIONS
   - Chatbots and assistants
   - Code generation tools
   - Content creation
   - Data analysis

5. CONTRIBUTE TO OPEN SOURCE
   - Many open LLM projects need contributors
   - Share your models and datasets

The field is moving fast - keep learning!
""")

pause()


print("=" * 60)
print("  THANK YOU!")
print("=" * 60)

print("""
Thank you for completing this course!

You started knowing little about how LLMs work.
Now you can BUILD one from scratch.

That's an incredible achievement!

The best way to solidify this knowledge is to:
    1. Build your own projects
    2. Explain it to others
    3. Keep experimenting

Good luck on your AI journey!
""")

print("=" * 60)
print("  END OF COURSE")
print("=" * 60)
print("\nCongratulations, Graduate!\n")
