import os
import logging
from gensim.models import Word2Vec

# 1. Setup Logging to track training progress
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_all_configurations(corpus_path, output_dir="models"):
    """Trains 16 variations of Word2Vec models based on IITJ corpus."""
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and tokenize the corpus
    print(f"--- Loading corpus from {corpus_path} ---")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        # Assumes each line in corpus.txt is a preprocessed sentence/document
        sentences = [line.split() for line in f if line.strip()]

    # Configuration Hyperparameters
    architectures = [0, 1]      # 0 for CBOW, 1 for Skip-gram
    dimensions = [100, 300]     # Low vs High dimensionality
    windows = [5, 10]           # Narrow vs Broad context
    negative_samples = [5, 10]  # Negative sampling counts

    total_models = len(architectures) * len(dimensions) * len(windows) * len(negative_samples)
    count = 1

    for sg in architectures:
        arch_name = "sg" if sg == 1 else "cbow"
        
        for dim in dimensions:
            for win in windows:
                for neg in negative_samples:
                    
                    # Define model filename using the convention
                    model_name = f"{arch_name}_dim{dim}_win{win}_neg{neg}.model"
                    model_path = os.path.join(output_dir, model_name)
                    
                    print(f"\n[{count}/{total_models}] Training: {model_name}")
                    
                    # 2. Initialize and Train Model
                    model = Word2Vec(
                        sentences=sentences,
                        vector_size=dim,
                        window=win,
                        negative=neg,
                        sg=sg,
                        min_count=2,    # Ignore very rare words
                        workers=4,      # Use 4 CPU cores
                        epochs=20       # Sufficient for a small 96K token corpus
                    )
                    
                    # 3. Save the model
                    model.save(model_path)
                    print(f"✅ Saved to {model_path}")
                    count += 1

    print("\n" + "="*30)
    print(f"Successfully trained {total_models} models.")
    print("="*30)

if __name__ == "__main__":
    # Ensure your 'corpus.txt' is in the same directory
    CORPUS_FILE = "corpus.txt"
    
    if os.path.exists(CORPUS_FILE):
        train_all_configurations(CORPUS_FILE)
    else:
        print(f"❌ Error: {CORPUS_FILE} not found. Please run your preprocessing script first.")