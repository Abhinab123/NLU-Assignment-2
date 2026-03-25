import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import seaborn as sns
import os

# 1. Ensure the plots folder exists
if not os.path.exists('plots'):
    os.makedirs('plots')

def plot_tsne_clusters(model, words_to_plot, title="t-SNE Word Clusters"):
    """Generates a 2D cluster plot for specific word groups."""
    valid_words = [word for word in words_to_plot if word in model.wv]
    if not valid_words:
        print("❌ No valid words found for t-SNE.")
        return

    vectors = np.array([model.wv[word] for word in valid_words])

    # FIXED: Changed 'fit_手柄' to 'fit_transform'
    tsne = TSNE(n_components=2, perplexity=min(5, len(valid_words)-1), init='pca', random_state=42)
    embeddings_2d = tsne.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], edgecolors='k', c='skyblue', s=100)
    
    for i, label in enumerate(valid_words):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                     xytext=(5, 2), textcoords='offset points', fontsize=12)
    
    plt.title(title, fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('plots/tsne_clusters.png')
    plt.show()
    print("✅ Saved t-SNE plot to plots/tsne_clusters.png")

def plot_weight_heatmap(numpy_embeddings_path, gensim_model_path, word="research"):
    """Compares weight distribution between custom NumPy and Gensim models."""
    if not os.path.exists(numpy_embeddings_path):
        print(f"❌ File not found: {numpy_embeddings_path}")
        return

    # Load your custom weights (001_embeddings_ep20.npy)
    custom_weights = np.load(numpy_embeddings_path) 
    
    # Load Gensim model
    try:
        gensim_model = Word2Vec.load(gensim_model_path)
        gensim_vector = gensim_model.wv[word]
    except Exception as e:
        print(f"❌ Error loading Gensim model: {e}")
        return
    
    # Take a sample slice (first 50 dimensions) to make it readable
    slice_dim = 50
    # Comparing Custom index 0 vs Gensim Word vector
    comparison = np.vstack([custom_weights[0][:slice_dim], gensim_vector[:slice_dim]])
    
    plt.figure(figsize=(14, 5))
    sns.heatmap(comparison, annot=False, cmap='RdYlBu', center=0,
                yticklabels=['Custom NumPy (001)', 'Gensim Optimized'])
    plt.title(f"Weight Distribution Comparison (First {slice_dim} dimensions for '{word}')")
    plt.xlabel("Dimension Index")
    plt.savefig('plots/similarity_heatmap.png')
    plt.show()
    print("✅ Saved Heatmap to plots/similarity_heatmap.png")

if __name__ == "__main__":
    # Words to visualize
    academic_terms = ['btech', 'mtech', 'phd', 'student', 'faculty', 'research', 'jodhpur', 'semester']

    # 2. UPDATED FILENAMES
    # Make sure 'models/' folder exists and contains these files
    GENSIM_MODEL = "models/sg_dim300_win5_neg10.model" 
    NUMPY_NPY = "001_embeddings_ep20.npy" # Fixed as per your filename

    # Run t-SNE
    try:
        sg_model = Word2Vec.load(GENSIM_MODEL)
        plot_tsne_clusters(sg_model, academic_terms, "IITJ Word Clusters")
    except Exception as e:
        print(f"Error: {e}")

    # Run Heatmap
    plot_weight_heatmap(NUMPY_NPY, GENSIM_MODEL)