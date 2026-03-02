"""
Lesson 3: Saving, Loading, and Pretrained Weights
===================================================

Training a model from scratch takes enormous resources:
    - GPT-3: Estimated $4.6 million in compute
    - Days to weeks on hundreds of GPUs

Instead, we can use PRETRAINED WEIGHTS - models that
others have already trained!

This lesson covers:
    - Saving and loading your model checkpoints
    - Understanding state_dict
    - Working with pretrained models
    - Transfer learning basics

Usage: python 03_saving_loading.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 3: Saving and Loading Models")
print("  Working with Pretrained Weights")
print("=" * 60)

print("""
WHY THIS MATTERS

Training GPT from scratch is expensive:
    - GPT-2: ~$50,000 in compute (2019)
    - GPT-3: ~$4.6 million (2020)
    - GPT-4: Estimated $100+ million

But you can use PRETRAINED models for free!
OpenAI, Meta, Google all release trained weights.

This lesson teaches you to:
    1. Save your trained models
    2. Load them back later
    3. Use pretrained weights from others
""")

pause()


# Simple model for demonstrations
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# State Dict
# ---------------------------------------------------------------------------

print("UNDERSTANDING STATE_DICT")
print("-" * 40)

print("""
Every PyTorch model has a STATE_DICT - a Python dictionary
that maps layer names to their parameter tensors.

    state_dict = {
        'fc1.weight': tensor(...),
        'fc1.bias': tensor(...),
        'fc2.weight': tensor(...),
        'fc2.bias': tensor(...),
    }

This contains ALL the learned parameters of your model.
Saving the state_dict = saving the model's "brain".
""")

pause()


# Create a model and examine its state_dict
model = SimpleModel(input_size=10, hidden_size=20, output_size=5)

print("Model state_dict:")
print("-" * 50)
for name, param in model.state_dict().items():
    print(f"  {name:15} | Shape: {str(list(param.shape)):12} | {param.numel():>5} values")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

pause()


# ---------------------------------------------------------------------------
# Saving Models
# ---------------------------------------------------------------------------

print("SAVING MODELS")
print("-" * 40)

print("""
Two ways to save:

1. SAVE STATE_DICT ONLY (Recommended)
   torch.save(model.state_dict(), 'weights.pt')
   
   + Saves just the parameters
   + Smaller file size
   + Works even if code changes
   - Need model class to load

2. SAVE ENTIRE MODEL (Not recommended)
   torch.save(model, 'model.pt')
   
   + Can load without class definition
   - Larger file size
   - Breaks if code changes
   - Pickle security issues

Always use method 1 for production!
""")

pause()


# Save the model
save_path = '/tmp/model_weights.pt'
torch.save(model.state_dict(), save_path)

file_size = os.path.getsize(save_path)
print(f"Model saved to: {save_path}")
print(f"File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
print()

# Peek at what's in the file
loaded_state = torch.load(save_path, weights_only=True)
print(f"Contents: {type(loaded_state)}")
print(f"Keys: {list(loaded_state.keys())}")

pause()


# ---------------------------------------------------------------------------
# Loading Models
# ---------------------------------------------------------------------------

print("LOADING MODELS")
print("-" * 40)

print("""
To load a saved model:

    1. Create model with SAME architecture
    2. Load state_dict from file
    3. Apply state_dict to model

    model = MyModel(...)  # Same architecture!
    state = torch.load('weights.pt')
    model.load_state_dict(state)
""")

pause()


# Create a NEW model (starts with random weights)
new_model = SimpleModel(input_size=10, hidden_size=20, output_size=5)

print("Before loading (random weights):")
print(f"  fc1.weight[0,:5]: {[f'{x:.4f}' for x in new_model.fc1.weight[0,:5].tolist()]}")
print()

# Load the saved weights
new_model.load_state_dict(torch.load(save_path, weights_only=True))

print("After loading (saved weights):")
print(f"  fc1.weight[0,:5]: {[f'{x:.4f}' for x in new_model.fc1.weight[0,:5].tolist()]}")
print()

# Verify it matches original
original = [f'{x:.4f}' for x in model.fc1.weight[0,:5].tolist()]
loaded = [f'{x:.4f}' for x in new_model.fc1.weight[0,:5].tolist()]
print(f"Matches original? {original == loaded}")

pause()


# ---------------------------------------------------------------------------
# Training Checkpoints
# ---------------------------------------------------------------------------

print("SAVING TRAINING CHECKPOINTS")
print("-" * 40)

print("""
During training, save MORE than just model weights:

    checkpoint = {
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss,
        'config': model_config,
    }
    torch.save(checkpoint, 'checkpoint.pt')

Why save optimizer state?
    - Optimizers like Adam have momentum terms
    - These are learned during training
    - Need them to resume training properly
""")

pause()


# Simulate training and save checkpoint
model = SimpleModel(10, 20, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Fake some training steps
for _ in range(10):
    x = torch.randn(32, 10)
    y = torch.randn(32, 5)
    loss = F.mse_loss(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save complete checkpoint
checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
    'config': {'input_size': 10, 'hidden_size': 20, 'output_size': 5}
}

checkpoint_path = '/tmp/checkpoint.pt'
torch.save(checkpoint, checkpoint_path)

print("Checkpoint saved!")
print(f"  Path: {checkpoint_path}")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Loss: {checkpoint['loss']:.4f}")
print(f"  Config: {checkpoint['config']}")

pause()


print("LOADING CHECKPOINTS TO RESUME TRAINING")
print("-" * 40)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, weights_only=False)

# Recreate model with same config
config = checkpoint['config']
model = SimpleModel(**config)
model.load_state_dict(checkpoint['model_state_dict'])

# Recreate optimizer and load its state
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Resume from saved epoch
start_epoch = checkpoint['epoch']

print("Checkpoint loaded!")
print(f"  Resuming from epoch: {start_epoch}")
print(f"  Previous loss: {checkpoint['loss']:.4f}")
print()
print("Model and optimizer states restored - ready to continue training!")

pause()


# ---------------------------------------------------------------------------
# Strict Loading
# ---------------------------------------------------------------------------

print("STRICT VS NON-STRICT LOADING")
print("-" * 40)

print("""
load_state_dict has a 'strict' parameter:

    model.load_state_dict(state, strict=True)  # Default
        - All keys must match exactly
        - Raises error if keys don't match

    model.load_state_dict(state, strict=False)
        - Loads matching keys only
        - Ignores missing/extra keys
        - Useful for transfer learning

When to use strict=False?
    - Loading partial weights
    - Model architecture changed slightly
    - Transfer learning from similar model
""")

pause()


# Demonstrate strict=False
class BiggerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)  # Same as SimpleModel
        self.fc2 = nn.Linear(20, 5)   # Same as SimpleModel
        self.fc3 = nn.Linear(5, 2)    # NEW layer
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


bigger_model = BiggerModel()

# Load SimpleModel weights (missing fc3)
try:
    bigger_model.load_state_dict(torch.load(save_path, weights_only=True), strict=True)
    print("Strict loading succeeded")
except RuntimeError as e:
    print(f"Strict loading failed: missing keys")

# Try with strict=False
result = bigger_model.load_state_dict(
    torch.load(save_path, weights_only=True), 
    strict=False
)

print(f"\nNon-strict loading:")
print(f"  Missing keys: {result.missing_keys}")
print(f"  Unexpected keys: {result.unexpected_keys}")
print()
print("fc1 and fc2 loaded, fc3 stays randomly initialized!")

pause()


# ---------------------------------------------------------------------------
# Pretrained Models
# ---------------------------------------------------------------------------

print("USING PRETRAINED MODELS")
print("-" * 40)

print("""
The easiest way to use pretrained models is Hugging Face:

    pip install transformers

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    # Load pretrained GPT-2 (downloads automatically)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Generate text!
    inputs = tokenizer("Hello, I am", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(outputs[0]))

Available GPT-2 models:
    'gpt2':        124M parameters
    'gpt2-medium': 355M parameters
    'gpt2-large':  774M parameters
    'gpt2-xl':     1.5B parameters
""")

pause()


print("HUGGING FACE MODEL ZOO")
print("-" * 40)

print("""
Hugging Face hosts thousands of pretrained models:

    huggingface.co/models

Popular models:
    GPT-2 (OpenAI):   Text generation
    BERT (Google):    Understanding/classification
    T5 (Google):      Text-to-text
    LLaMA (Meta):     Open-source GPT alternative
    Mistral:          Efficient open-source LLM
    Falcon:           Open-source LLM

Universal loading:
    from transformers import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained('model-name')
    tokenizer = AutoTokenizer.from_pretrained('model-name')

This is how most people use LLMs in practice!
""")

pause()


# ---------------------------------------------------------------------------
# Weight Mapping
# ---------------------------------------------------------------------------

print("LOADING WEIGHTS INTO CUSTOM MODELS")
print("-" * 40)

print("""
To load OpenAI's GPT-2 weights into our custom GPT:

    1. Download the weights
    2. Map their names to our names
    3. Handle any shape differences
    4. Load with load_state_dict

Name mapping example:
    OpenAI GPT-2          Our GPT
    -------------         -------
    wte.weight         -> token_embedding.weight
    wpe.weight         -> position_embedding.weight
    h.0.ln_1.weight    -> blocks.0.ln1.weight
    h.0.attn.c_attn    -> blocks.0.attn.qkv_proj
    h.0.mlp.c_fc       -> blocks.0.ffn.fc1
    ln_f.weight        -> ln_final.weight

In practice, just use Hugging Face - it handles all this!
""")

pause()


# ---------------------------------------------------------------------------
# Transfer Learning
# ---------------------------------------------------------------------------

print("TRANSFER LEARNING")
print("-" * 40)

print("""
TRANSFER LEARNING: Use pretrained model as starting point.

Instead of:
    Random weights → Train on your data

Use:
    Pretrained weights → Fine-tune on your data

Benefits:
    - Much faster training
    - Works with less data
    - Often better results
    - Leverages billions of training tokens

This is how most LLM applications work!
    1. Start with GPT-2 / LLaMA / etc.
    2. Fine-tune on your specific task
    3. Deploy!

We'll cover fine-tuning in detail next week!
""")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  WEEK 5 COMPLETE!")
print("=" * 60)

print("""
You now know how to evaluate, generate, and manage models!

WEEK 5 RECAP:

LESSON 1 - Evaluation:
    - Cross-entropy loss measures "surprise"
    - Perplexity = exp(loss), intuitive metric
    - Train/val/test splits prevent overfitting
    - Early stopping saves best model

LESSON 2 - Decoding Strategies:
    - Greedy: argmax (deterministic, boring)
    - Temperature: control randomness
    - Top-k: sample from top k tokens
    - Top-p: sample from cumulative mass

LESSON 3 - Saving/Loading:
    - state_dict contains all parameters
    - torch.save/load for persistence
    - Checkpoints include optimizer state
    - Pretrained models via Hugging Face

Next week: FINE-TUNING for classification!
We'll adapt pretrained models for sentiment analysis,
spam detection, and more.
""")

print("=" * 60)
print("  End of Week 5")
print("=" * 60)
