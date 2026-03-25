import numpy as np
import re
from collections import Counter
import time
import pickle
import os

# ==========================================
# 1. Math Utilities
# ==========================================
def safe_softmax(x):
    """Numerically stable softmax to prevent overflow/NaN errors."""
    shifted_x = x - np.max(x)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=0)

# ==========================================
# 2. CBOW Neural Network (Pure NumPy)
# ==========================================
class ScratchCBOW:
    def __init__(self, vocab_size, embed_dim=300, lr=0.025):
        self.vocab_size = vocab_size
        self.dim = embed_dim
        self.lr = lr
        
        # Initialize weights with standard normal distribution, scaled down
        print(f"Initializing {(vocab_size * embed_dim * 2):,} parameters...")
        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.05
        self.W_out = np.random.randn(embed_dim, vocab_size) * 0.05

    def train_step(self, ctx_indices, target_idx):
        # --- FORWARD PASS ---
        # Average the embeddings of the context words
        hidden = np.mean([self.W_in[idx] for idx in ctx_indices], axis=0)
        
        # Calculate scores for all words in vocabulary and apply softmax
        logits = np.dot(hidden, self.W_out)
        probs = safe_softmax(logits)

        # --- BACKWARD PASS ---
        error = probs.copy()
        error[target_idx] -= 1.0  # The true word gets a 1, everything else is 0
        
        # Calculate gradients
        grad_W_out = np.outer(hidden, error)
        grad_hidden = np.dot(self.W_out, error)
        
        # --- UPDATE WEIGHTS ---
        self.W_out -= self.lr * grad_W_out
        
        # Backpropagate to the input layer (split among context words)
        for idx in ctx_indices:
            self.W_in[idx] -= self.lr * (grad_hidden / len(ctx_indices))
            
        return -np.log(probs[target_idx] + 1e-9)

# ==========================================
# 3. Data Pipeline
# ==========================================
def prepare_full_corpus(filepath, min_freq=2, window_size=2):
    print(f"\n--- Reading and Processing {filepath} ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}. Please ensure it is in the same directory.")

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    # Tokenize
    tokens = re.findall(r'\b[a-z]{2,}\b', text)
    print(f"Total tokens in raw corpus: {len(tokens):,}")
    
    # Frequency filtering
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_freq]
    
    # CRITICAL: Ensure our heatmap target words are strictly in the vocabulary
    # so the visualization script doesn't fail later.
    target_words = ['btech', 'mtech', 'phd', 'student', 'faculty', 'research', 'algorithm', 'data']
    for w in target_words:
        if w not in vocab and w in counts:
            vocab.append(w)
            
    vocab_size = len(vocab)
    print(f"Active Vocabulary Size (min_freq={min_freq}): {vocab_size:,}")
    
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    
    # Filter corpus to only keep known words
    valid_tokens = [w for w in tokens if w in w2i]
    
    print("Mapping context-target pairs in memory (this takes a moment)...")
    training_pairs = []
    for i in range(window_size, len(valid_tokens) - window_size):
        context = (
            [valid_tokens[i - j - 1] for j in range(window_size)] +
            [valid_tokens[i + j + 1] for j in range(window_size)]
        )
        target = valid_tokens[i]
        
        ctx_indices = [w2i[w] for w in context]
        target_idx = w2i[target]
        training_pairs.append((ctx_indices, target_idx))
        
    print(f"Ready: {len(training_pairs):,} total training pairs generated.")
    return training_pairs, w2i, i2w, vocab_size

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    CORPUS_FILE = "corpus.txt"
    DIMENSIONS = 300       # Required by the assignment
    EPOCHS = 20
    LEARNING_RATE = 0.025
    MIN_FREQUENCY = 3      # Drops highly obscure words/typos to save RAM
    
    # 1. Prepare Data
    pairs, word_to_idx, idx_to_word, v_size = prepare_full_corpus(CORPUS_FILE, MIN_FREQUENCY)
    
    # 2. Save Vocabulary Mapping IMMEDIATELY
    # This ensures your heatmap script has the mapping even if training gets interrupted
    vocab_file = "001_vocab_mapping.pkl"
    with open(vocab_file, 'wb') as f:
        pickle.dump({'w2i': word_to_idx, 'i2w': idx_to_word}, f)
    print(f"✅ Vocabulary mapped and saved to '{vocab_file}'\n")
    
    # 3. Initialize Model
    model = ScratchCBOW(v_size, embed_dim=DIMENSIONS, lr=LEARNING_RATE)
    
    print("="*50)
    print("🚀 INITIATING FULL NUMPY TRAINING")
    print("="*50)
    
    total_samples = len(pairs)
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        running_loss = 0.0
        
        for step, (ctx, tgt) in enumerate(pairs):
            loss = model.train_step(ctx, tgt)
            running_loss += loss
            
            # Print a status update every 20,000 steps so you know it's not frozen
            if (step + 1) % 20000 == 0:
                pct = ((step + 1) / total_samples) * 100
                print(f"  [Epoch {epoch+1:02d}] {pct:05.1f}% complete | Current Loss: {loss:.4f}")
                
        duration = time.time() - start_time
        avg_loss = running_loss / total_samples
        
        print("-" * 50)
        print(f"✅ Epoch {epoch+1}/{EPOCHS} Finished!")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Time Taken:   {duration/60:.2f} minutes")
        
        # 5. Save Checkpoint
        # We save the W_in matrix, which acts as our final word embeddings
        weights_file = f"001_embeddings_ep{epoch+1}.npy"
        np.save(weights_file, model.W_in)
        print(f"   Checkpoint saved: {weights_file}")
        print("-" * 50 + "\n")

    print("🎉 ALL EPOCHS COMPLETE. Full corpus training finished successfully.")