"""
Lesson 1: Introduction to Instruction Fine-Tuning
==================================================

Welcome to the final week!

You've learned to build GPT and fine-tune it for classification.
Now we'll learn how to make it FOLLOW INSTRUCTIONS - just like ChatGPT!

This is the secret sauce that transforms a text predictor into
a helpful AI assistant:

    Base GPT: "The capital of France is" -> "Paris, which is known for..."
    
    Instruction-tuned GPT:
        User: "What is the capital of France?"
        Assistant: "The capital of France is Paris."

By the end of this week, you'll understand how ChatGPT was built!

Usage: python 01_instruction_intro.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 1: Introduction to Instruction Fine-Tuning")
print("  Making GPT Follow Commands")
print("=" * 60)

print("""
THE JOURNEY TO CHATGPT

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

We'll focus on Step 2 - Supervised Fine-Tuning (SFT).
This is what transforms GPT into a useful assistant!
""")

pause()


# ---------------------------------------------------------------------------
# Base Model vs Instruction-Tuned Model
# ---------------------------------------------------------------------------

print("BASE MODEL vs INSTRUCTION-TUNED MODEL")
print("-" * 40)

print("""
BASE GPT (pretrained only):
    Trained to predict the next token.
    Given text, it continues it naturally.
    
    Input: "Write a poem about cats"
    Output: "Write a poem about cats and dogs. The poem should..."
    
    It's CONTINUING the text, not FOLLOWING the instruction!

INSTRUCTION-TUNED GPT:
    Trained on instruction-response pairs.
    Given an instruction, it follows it.
    
    Input: "Write a poem about cats"
    Output: "Soft paws padding through the night,
            Whiskers twitching in moonlight..."
    
    It FOLLOWS the instruction!

The difference is in the TRAINING DATA.
""")

pause()


print("WHAT CHANGES IN TRAINING")
print("-" * 40)

print("""
PRETRAINING DATA:
    Raw text from the internet, books, etc.
    "The cat sat on the mat. It was a sunny day..."
    
    Model learns: Given text, predict what comes next.

INSTRUCTION FINE-TUNING DATA:
    Pairs of (instruction, response):
    
    Instruction: "Translate 'hello' to French"
    Response: "Bonjour"
    
    Instruction: "Summarize this article: [article text]"
    Response: "[Summary of the article]"
    
    Instruction: "Write a haiku about mountains"
    Response: "Peaks touch the blue sky
              Snow melts into rushing streams
              Nature stands silent"

Model learns: Given instruction, generate appropriate response.
""")

pause()


# ---------------------------------------------------------------------------
# Instruction Data Format
# ---------------------------------------------------------------------------

print("INSTRUCTION DATA FORMAT")
print("-" * 40)

print("""
How do we format instruction-response pairs for training?

OPTION 1: Simple Format
    "<instruction> {instruction} <response> {response}"
    
    Example:
    "<instruction> What is 2+2? <response> 4"

OPTION 2: Chat Format (like ChatGPT)
    "User: {instruction}
     Assistant: {response}"
    
    Example:
    "User: What is 2+2?
     Assistant: 4"

OPTION 3: Special Tokens
    "<USER> {instruction} <ASSISTANT> {response} <END>"
    
    Example:
    "<USER> What is 2+2? <ASSISTANT> 4 <END>"

We'll use special tokens for clarity.
""")

pause()


def format_instruction(instruction, response):
    """Format an instruction-response pair for training."""
    return f"User: {instruction}\nAssistant: {response}"


examples = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Translate 'hello' to Spanish.", "Hola"),
    ("Write a short poem about rain.", "Drops fall from gray skies,\nPuddles form on empty streets,\nNature's soft refrain."),
]

print("FORMATTED TRAINING EXAMPLES")
print("-" * 40)
print()

for instruction, response in examples:
    formatted = format_instruction(instruction, response)
    print(formatted)
    print("-" * 40)

pause()


# ---------------------------------------------------------------------------
# The Training Objective
# ---------------------------------------------------------------------------

print("THE TRAINING OBJECTIVE: LOSS MASKING")
print("-" * 40)

print("""
Key insight: We only compute loss on the RESPONSE part!

Full training example:
    "User: What is 2+2?
     Assistant: 4"

We tokenize the whole thing, but:
    - DON'T compute loss on "User: What is 2+2? Assistant:"
    - ONLY compute loss on "4"

Why? We want the model to learn to GENERATE responses,
not to predict the instruction (which is given).

This is called "masking the prompt" or "instruction masking".
""")

pause()


print("LOSS MASKING VISUALIZATION")
print("-" * 40)

print("""
Example: 'What is 2+2?' -> '4'

Tokens:
    [<USER>] [What] [is] [2] [+] [2] [?] [<ASST>] [4] [<END>]
    
Labels (for loss computation):
    [-100]  [-100] [-100] ... [-100]   [-100]   [4]  [<END>]
    
The -100 tells PyTorch: "Ignore this position in the loss!"

Loss is ONLY computed on the response tokens: [4] and [<END>]

This teaches the model:
    "Given <USER> ... <ASST>, generate the right response"
""")

pause()


# ---------------------------------------------------------------------------
# Types of Instructions
# ---------------------------------------------------------------------------

print("TYPES OF INSTRUCTIONS")
print("-" * 40)

print("""
A good instruction dataset covers many task types:

1. QUESTION ANSWERING
   "What is the largest planet?" -> "Jupiter"
   
2. SUMMARIZATION
   "Summarize: [long text]" -> "[short summary]"
   
3. TRANSLATION
   "Translate to French: Hello" -> "Bonjour"
   
4. CREATIVE WRITING
   "Write a poem about love" -> "[poem]"
   
5. CODE GENERATION
   "Write Python code to sort a list" -> "[code]"
   
6. REASONING
   "If A > B and B > C, is A > C?" -> "Yes, by transitivity"
   
7. CONVERSATION
   "How are you today?" -> "I'm doing well, thank you!"
   
8. TASK COMPLETION
   "Extract names from: John met Mary" -> "John, Mary"

Diversity in training data = more capable model!
""")

pause()


# ---------------------------------------------------------------------------
# Example Instruction Dataset
# ---------------------------------------------------------------------------

INSTRUCTION_DATASET = [
    # Question Answering
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("What is the speed of light?", "The speed of light is approximately 299,792 km/s."),
    ("How many continents are there?", "There are 7 continents on Earth."),
    
    # Math
    ("What is 15 + 27?", "15 + 27 = 42"),
    ("Calculate 8 times 7.", "8 times 7 equals 56."),
    ("What is half of 100?", "Half of 100 is 50."),
    
    # Translation
    ("Translate 'goodbye' to Spanish.", "Adios"),
    ("How do you say 'thank you' in French?", "Merci"),
    ("What is 'water' in German?", "Wasser"),
    
    # Creative
    ("Write a haiku about the ocean.", "Waves crash on the shore,\nSalt and foam dance in the wind,\nEndless blue expanse."),
    ("Give me a metaphor for time.", "Time is a river, flowing ever onward, never to return."),
    
    # Explanation
    ("Explain what gravity is.", "Gravity is a force that attracts objects with mass toward each other."),
    ("What causes rain?", "Rain forms when water vapor condenses into droplets that fall."),
    
    # Task completion
    ("List three primary colors.", "Red, blue, and yellow."),
    ("Name the four seasons.", "Spring, summer, fall, and winter."),
    
    # Conversation
    ("Hello!", "Hello! How can I help you today?"),
    ("Thank you for your help.", "You're welcome! Feel free to ask if you need anything else."),
    ("What can you do?", "I can answer questions, help with tasks, and have conversations."),
    
    # Reasoning
    ("If all cats have tails and Whiskers is a cat, does Whiskers have a tail?", 
     "Yes, if all cats have tails and Whiskers is a cat, then Whiskers has a tail."),
]

print("EXAMPLE INSTRUCTION DATASET")
print("-" * 40)

print(f"Dataset size: {len(INSTRUCTION_DATASET)} examples")
print()

categories = {
    "QA": [0, 1],
    "Math": [4, 5],
    "Translation": [7, 8],
    "Creative": [10],
    "Conversation": [16, 17],
}

print("Sample instructions by category:")
for category, indices in categories.items():
    print(f"\n{category}:")
    for idx in indices[:2]:
        inst, resp = INSTRUCTION_DATASET[idx]
        print(f"  Q: {inst}")
        print(f"  A: {resp[:50]}{'...' if len(resp) > 50 else ''}")

pause()


# ---------------------------------------------------------------------------
# Why Instruction Tuning Works
# ---------------------------------------------------------------------------

print("WHY INSTRUCTION TUNING WORKS")
print("-" * 40)

print("""
The pretrained model ALREADY knows how to:
    - Understand language
    - Generate coherent text
    - Reason (to some degree)
    - Follow patterns

Instruction tuning teaches it a NEW PATTERN:
    "When you see '<USER> ... <ASST>', generate a helpful response"

It's like teaching someone a new format:
    - They already know math
    - You teach them: "When someone asks, give just the answer"
    - Now they follow that format!

Key insight: We're not teaching NEW knowledge,
we're teaching a new WAY TO USE existing knowledge.

That's why instruction tuning needs relatively little data
(thousands of examples, not billions).
""")

pause()


print("THE INSTRUCTION-FOLLOWING PATTERN")
print("-" * 40)

print("""
After instruction tuning, the model learns:

    Pattern: "<USER> {question}" -> "<ASST> {answer}"
    
    This pattern becomes STRONGLY weighted in the model.
    
When you give it:
    "<USER> What is the capital of France? <ASST>"
    
The model KNOWS the next tokens should be:
    "The capital of France is Paris. <END>"
    
NOT:
    "<USER> What is the population of France?"
    (continuing the user's pattern)

The training examples establish the ASSISTANT role!
""")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
1. THE GOAL
   - Transform text predictor into instruction follower
   - Model learns to generate helpful responses
   
2. TRAINING DATA FORMAT
   - Instruction-response pairs
   - "<USER> {instruction} <ASST> {response} <END>"

3. LOSS MASKING
   - Only compute loss on response tokens
   - Prompt tokens masked with -100
   - Model learns to GENERATE responses

4. TASK DIVERSITY
   - QA, translation, math, creative, coding...
   - More diverse = more capable

5. WHY IT WORKS
   - Model already has knowledge from pretraining
   - Just learning a new PATTERN
   - Needs relatively little data (thousands, not billions)

Next: Building and training an instruction-tuned model!
""")

print("=" * 60)
print("  End of Lesson 1")
print("=" * 60)
