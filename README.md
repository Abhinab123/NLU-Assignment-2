# NLU Assignment 2: Word Embeddings & Character-Level Sequence Modeling
**Student:** Abhinab Bezbaruah  
**Roll No:** M25CSE001  
**Institute:** IIT Jodhpur

---

## 📂 Project Overview
This repository contains the implementation for Assignment 2 of the Natural Language Understanding (NLU) course. It covers the end-to-end pipeline for two distinct NLP tasks:
1. **Problem 1:** Building a custom Word2Vec model from scratch using the IIT Jodhpur web corpus.
2. **Problem 2:** Developing a character-level name generator using RNN, BiLSTM, and Attention architectures.

---

## 📁 Problem 1: IIT Jodhpur Word2Vec
I curated a domain-specific corpus by scraping `iitj.ac.in` to understand how word embeddings capture the unique semantic relationships within our campus.

### 1. Data Acquisition & Preprocessing
* **Scraping:** Implemented a BFS-based crawler (`scraper.py`) to extract text from HTML and academic PDFs.
* **Cleaning:** Normalized text to lowercase and used Regex to strip URLs, emails, and non-alphabetic noise.
* **Corpus Stats:** Final vocabulary size is ~8,000 words with ~96,000 total tokens.

### 2. Implementation & Training
* **Gensim Models:** Trained 16 variations comparing **CBOW** vs **Skip-gram** across multiple dimensions (100/300) and window sizes (5/10).
* **NumPy from Scratch:** Developed `numpy_word2vec.py` to manually implement the Skip-gram architecture. This version uses raw Linear Algebra for weight updates and gradient descent.
* **Best Model:** Skip-gram with 300 dimensions and a window of 10 provided the most accurate semantic analogies for IITJ terms.

### 3. Visual Analysis
* **t-SNE Clusters:** (`001_clustering_visualization.png`) Visual proof showing that terms like *BTech, MTech, PhD,* and *Faculty* naturally cluster together.
* **Weight Heatmaps:** (`001_heatmap_comparison.png`) A comparative "fingerprint" showing the similarity between my custom NumPy weights and the Gensim optimized vectors.

---

## 📁 Problem 2: Indian Name Generation
A deep learning project focused on character-level sequence modeling to generate authentic-sounding Indian names.

### 1. Architectures Explored
* **Vanilla RNN:** Served as the baseline. While it produced realistic names, it had lower novelty due to its tendency to "memorize" the training data.
* **BiLSTM (Top Performer):** By processing name sequences in both directions, this model captured complex phonetic structures and generated the highest variety of unique names.
* **RNN + Attention:** Added an attention mechanism to allow the decoder to "focus" on specific character dependencies, resulting in highly creative and novel name generation.

### 2. Performance Comparison

| Model | Realism | Novelty | Diversity |
| :--- | :--- | :--- | :--- |
| **Vanilla RNN** | High | Low | Low |
| **BiLSTM** | **Very High** | **High** | **High** |
| **Attention** | High | Very High | Medium |

---

## 🛠️ Setup and Execution

### 1. Clone the Repository
```bash
git clone [https://github.com/Abhinab123/NLU-Assignment-2.git](https://github.com/Abhinab123/NLU-Assignment-2.git)

2. Install Dependencies
Bash
pip install -r requirements.txt
3. Run Problem 1 Visualization
Bash
python Problem_1/scripts/visualize_results.py
4. Run Problem 2 Generation
Bash
python Problem_2/train_and_generate.py
📄 Final Artifacts
NumPy Weights: Problem_1/models/001_embeddings_ep20.npy

Vocab Map: Problem_1/data/001_vocab_mapping.pkl

Generated Names: Problem_2/Problem2_Results.txt


---

