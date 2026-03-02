# Week 7: Instruction Fine-Tuning

Welcome to the final week! You'll learn how to make GPT follow instructions, just like ChatGPT.

## The Journey to ChatGPT

```
Step 1: Pretraining (Weeks 4-5)
    - Train on massive text data
    - Learn language patterns
    - Result: Good at completing text
    
Step 2: Supervised Fine-Tuning (This week!)
    - Train on instruction-response pairs
    - Learn to follow commands
    - Result: Follows instructions
    
Step 3: RLHF (Advanced - not covered)
    - Reinforcement Learning from Human Feedback
    - Learn human preferences
    - Result: More helpful and safe
```

## Lessons

Run each lesson in order:

```bash
python 01_instruction_intro.py   # ~20 min - Instruction tuning concepts
python 02_instruct_model.py      # ~25 min - Building the model
python 03_chatbot.py             # ~30 min - Complete chatbot
```

## Final Project

Your capstone: a complete chatbot from scratch!

```bash
python project_chatbot.py
```

## Key Concept: Loss Masking

We only compute loss on the RESPONSE, not the prompt:

```
Full sequence:  <USER> What is 2+2? <ASST> 4 <END>
Labels:         [-100] [-100] ...   [-100] 4 [END]

Loss is ONLY computed on response tokens!
```

The `-100` tells PyTorch to ignore those positions.

## Training Data Format

```python
# Format: <USER> instruction <ASSISTANT> response <END>

examples = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Say hello in Spanish.", "Hola!"),
    ("What is 2 plus 2?", "2 plus 2 equals 4."),
]
```

## Task Diversity

A good instruction dataset covers many task types:

| Type | Example Input | Example Output |
|------|--------------|----------------|
| QA | "What is the largest planet?" | "Jupiter" |
| Math | "What is 5 times 3?" | "15" |
| Translation | "Say hello in French" | "Bonjour" |
| Creative | "Write a haiku about rain" | "[haiku]" |
| Conversation | "How are you?" | "I'm doing well!" |

## Quick Reference

```python
# Tokenizer with special tokens
class ChatTokenizer:
    special_tokens = ['<PAD>', '<UNK>', '<USER>', '<ASST>', '<END>']
    
    def encode_prompt(self, instruction):
        return [USER_ID] + tokenize(instruction) + [ASST_ID]

# Dataset with loss masking
labels = [-100] * prompt_length + response_ids

# Training with ignore_index
loss = F.cross_entropy(logits, labels, ignore_index=-100)

# Generation
def generate(prompt):
    ids = encode_prompt(prompt)
    while True:
        next_token = model.sample(ids)
        ids.append(next_token)
        if next_token == END_ID:
            break
    return decode(ids)
```

## Course Complete!

After this week, you'll have built:
- A tokenizer with special tokens
- Instruction dataset with loss masking
- Complete GPT model
- Instruction-following chatbot

You now understand how ChatGPT works!
