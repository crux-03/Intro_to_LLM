"""
Lesson 1: Evaluating Language Models
=====================================

You've built and trained GPT. But how do we know if it's good?

This lesson covers:
    - Cross-entropy loss (what it actually means)
    - Perplexity (the standard LM metric)
    - Train/validation/test splits
    - Detecting and preventing overfitting

Understanding evaluation is crucial for knowing when to stop
training and how to compare different models.

Usage: python 01_evaluation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 1: Evaluating Language Models")
print("  Loss, Perplexity, and Metrics")
print("=" * 60)


# ---------------------------------------------------------------------------
# Cross-Entropy Loss
# ---------------------------------------------------------------------------

print("""
UNDERSTANDING CROSS-ENTROPY LOSS

We've been using cross-entropy loss, but what does it actually mean?

Cross-entropy measures how "surprised" the model is by the correct answer:
    - Model outputs probabilities for each possible next token
    - Loss = -log(probability assigned to correct token)
    - If model is confident in the correct answer: low loss
    - If model is surprised by the correct answer: high loss

Let's see this in action.
""")

pause()

# Example: model predicting next token
vocab = ["the", "cat", "sat", "dog"]
print("Example: Predicting the next word")
print(f"Vocabulary: {vocab}")
print()

# Scenario 1: Model is confident and correct
probs_confident = torch.tensor([0.7, 0.1, 0.1, 0.1])
correct_token = 0  # "the"
loss_confident = -torch.log(probs_confident[correct_token])

print("Scenario 1: Model thinks 'the' is likely, and 'the' is correct")
print(f"  Probabilities: {[f'{p:.1f}' for p in probs_confident.tolist()]}")
print(f"  P(correct='the') = {probs_confident[correct_token]:.2f}")
print(f"  Loss = -log(0.70) = {loss_confident:.4f}")
print()

# Scenario 2: Model is surprised
probs_surprised = torch.tensor([0.1, 0.1, 0.7, 0.1])
loss_surprised = -torch.log(probs_surprised[correct_token])

print("Scenario 2: Model thinks 'sat' is likely, but 'the' is correct")
print(f"  Probabilities: {[f'{p:.1f}' for p in probs_surprised.tolist()]}")
print(f"  P(correct='the') = {probs_surprised[correct_token]:.2f}")
print(f"  Loss = -log(0.10) = {loss_surprised:.4f}")
print()

print("Key insight: Higher loss when model is wrong!")

pause()


print("LOSS AT DIFFERENT CONFIDENCE LEVELS")
print("-" * 40)

print()
print("How does loss change with confidence?")
print()
print("  P(correct)  |  Loss")
print("  " + "-" * 25)

for prob in [0.99, 0.90, 0.70, 0.50, 0.30, 0.10, 0.01]:
    loss = -math.log(prob)
    print(f"    {prob:.2f}      |  {loss:.4f}")

print()
print("Note: Loss approaches infinity as probability approaches 0!")
print("This is why softmax always assigns SOME probability to every token.")

pause()


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

print("PERPLEXITY - THE STANDARD LM METRIC")
print("-" * 40)

print("""
Perplexity converts loss into an intuitive metric:

    Perplexity = exp(average cross-entropy loss)

Intuition: "How many tokens is the model choosing between?"

    - Perplexity of 10: Model is as confused as randomly
      choosing from 10 equally likely tokens
    - Perplexity of 100: Choosing from 100 tokens
    - Perplexity of 1: Perfect prediction!

LOWER PERPLEXITY = BETTER MODEL
""")

pause()


def compute_perplexity(loss):
    """Convert cross-entropy loss to perplexity."""
    return math.exp(loss)


print("Loss to Perplexity conversion:")
print()
print("  Loss  |  Perplexity  |  Interpretation")
print("  " + "-" * 50)

interpretations = {
    0.5: "Very confident",
    1.0: "Fairly confident",
    2.0: "Moderately confused",
    3.0: "Quite confused",
    4.0: "Very confused",
    5.0: "Extremely confused"
}

for loss in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
    ppl = compute_perplexity(loss)
    print(f"  {loss:.1f}   |    {ppl:6.1f}    |  {interpretations[loss]}")

pause()


print("WHY PERPLEXITY IS USEFUL")
print("-" * 40)

print("""
Why use perplexity instead of just loss?

1. INTUITIVE INTERPRETATION
   - "Loss of 3.5" doesn't mean much to humans
   - "Perplexity of 33" = "choosing between ~33 options"

2. COMPARABLE ACROSS MODELS
   - Standard benchmark metric
   - "GPT-2 has perplexity 20 on this dataset"

3. BASELINE REFERENCE
   - Random model has perplexity = vocabulary size
   - If vocab is 50,000, random perplexity = 50,000
   - Any trained model should beat this!

Typical perplexities in practice:
    Random (50K vocab):  50,000
    N-gram model:        200-500
    LSTM:                50-100
    GPT-2:               15-30
    GPT-3:               10-20
""")

pause()


# ---------------------------------------------------------------------------
# Train/Validation/Test Splits
# ---------------------------------------------------------------------------

print("TRAIN / VALIDATION / TEST SPLITS")
print("-" * 40)

print("""
We split data into three sets:

TRAINING SET (e.g., 80%)
    - Data the model learns from
    - Weights are updated based on this

VALIDATION SET (e.g., 10%)
    - Model NEVER trains on this
    - Used to check generalization during training
    - Also called "dev set" or "holdout set"

TEST SET (e.g., 10%)
    - Final evaluation only
    - Used ONCE at the very end
    - Reports final model quality

Why separate sets?
    - Model could MEMORIZE training data
    - Validation tells us if it GENERALIZES
    - This detects OVERFITTING
""")

pause()


print("Example data split:")
print()

total_tokens = 10000

train_size = int(0.80 * total_tokens)
val_size = int(0.10 * total_tokens)
test_size = total_tokens - train_size - val_size

print(f"Total tokens: {total_tokens:,}")
print(f"  Training:   {train_size:,} ({100*train_size/total_tokens:.0f}%)")
print(f"  Validation: {val_size:,} ({100*val_size/total_tokens:.0f}%)")
print(f"  Test:       {test_size:,} ({100*test_size/total_tokens:.0f}%)")

pause()


# ---------------------------------------------------------------------------
# Overfitting
# ---------------------------------------------------------------------------

print("DETECTING OVERFITTING")
print("-" * 40)

print("""
Overfitting: Model memorizes training data instead of learning patterns.

How to detect it:

    Training loss:   ↓ decreasing (good)
    Validation loss: ↓ then ↑ (bad!)
    
              Loss
                │
        Train   │    ___________
                │   /
                │  /
                │ /     Val
                │/  ___________
                │  /           \\
                │ /             \\
                │/               \\
                └───────────────────── Epochs
                          ↑
                   Stop training here!
                   (validation loss minimum)

When validation loss starts increasing: STOP TRAINING!
This is called "early stopping".
""")

pause()


# Simulated training curves
print("Simulated training showing overfitting:")
print()
print("  Epoch | Train Loss | Val Loss  | Status")
print("  " + "-" * 50)

train_losses = [3.5, 2.8, 2.3, 1.9, 1.5, 1.2, 0.9, 0.7, 0.5, 0.4]
val_losses =   [3.6, 2.9, 2.5, 2.2, 2.0, 2.0, 2.1, 2.3, 2.5, 2.8]

best_val = float('inf')
best_epoch = 0

for epoch, (train, val) in enumerate(zip(train_losses, val_losses), 1):
    if val < best_val:
        best_val = val
        best_epoch = epoch
        status = "New best"
    elif val > best_val + 0.1:
        status = "OVERFITTING"
    else:
        status = ""
    
    print(f"    {epoch:2d}  |    {train:.2f}     |   {val:.2f}    | {status}")

print()
print(f"Best model: Epoch {best_epoch} with validation loss {best_val:.2f}")
print("Training beyond epoch 5-6 made validation WORSE!")

pause()


# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------

print("COMPLETE EVALUATION FUNCTION")
print("-" * 40)

print("""
A proper evaluation function:
    1. Puts model in eval mode (disables dropout)
    2. Disables gradient computation (faster, less memory)
    3. Computes average loss over entire dataset
    4. Returns both loss and perplexity
""")

pause()


@torch.no_grad()
def evaluate(model, data_loader):
    """
    Evaluate model on a dataset.
    
    Args:
        model: The language model
        data_loader: DataLoader with (input, target) pairs
    
    Returns:
        loss: Average cross-entropy loss
        perplexity: exp(loss)
    """
    model.eval()  # Disable dropout
    
    total_loss = 0
    total_tokens = 0
    
    for x, y in data_loader:
        logits, loss = model(x, y)
        
        # Accumulate loss weighted by batch size
        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


print("Evaluation function:")
print()
print("  @torch.no_grad()  # No gradients needed")
print("  def evaluate(model, data_loader):")
print("      model.eval()  # Disable dropout")
print("      ...")
print("      return avg_loss, perplexity")
print()
print("Key points:")
print("  - @torch.no_grad() speeds up and saves memory")
print("  - model.eval() disables dropout (important!)")
print("  - Weight loss by number of tokens for accuracy")

pause()


# ---------------------------------------------------------------------------
# Training with Validation
# ---------------------------------------------------------------------------

print("TRAINING LOOP WITH EARLY STOPPING")
print("-" * 40)

print("""
A complete training loop should:
    1. Train on training data
    2. Evaluate on validation data each epoch
    3. Save the best model (by validation loss)
    4. Stop early if no improvement
""")

pause()


def train_with_early_stopping(model, train_loader, val_loader, 
                               epochs=10, lr=3e-4, patience=3):
    """
    Train with validation and early stopping.
    
    Args:
        patience: Stop if no improvement for this many epochs
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
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
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            status = "Saved!"
        else:
            epochs_without_improvement += 1
            status = f"No improvement ({epochs_without_improvement}/{patience})"
        
        print(f"Epoch {epoch+1:2d} | Train: {train_loss:.3f} | "
              f"Val: {val_loss:.3f} (PPL: {val_ppl:.1f}) | {status}")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping! No improvement for {patience} epochs.")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (Val Loss: {best_val_loss:.3f})")
    
    return model


print("Training function with early stopping implemented!")
print()
print("Key features:")
print("  - Tracks best validation loss")
print("  - Saves best model state")
print("  - Stops after 'patience' epochs without improvement")
print("  - Restores best model at the end")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
1. CROSS-ENTROPY LOSS
   - Measures "surprise" at correct answer
   - Loss = -log(P(correct token))
   - Lower = more confident = better

2. PERPLEXITY
   - Perplexity = exp(loss)
   - Intuition: "choosing from N equally likely tokens"
   - Standard metric for comparing language models
   - Lower = better

3. TRAIN/VAL/TEST SPLITS
   - Training: model learns from this
   - Validation: check generalization during training
   - Test: final evaluation only

4. OVERFITTING
   - Training loss decreases but validation increases
   - Model memorizes instead of generalizes
   - Solution: early stopping

5. BEST PRACTICES
   - Always evaluate on held-out data
   - Save best model by validation loss
   - Use early stopping

Next: Decoding strategies for better text generation!
""")

print("=" * 60)
print("  End of Lesson 1")
print("=" * 60)
