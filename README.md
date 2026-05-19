# LexiShift

**Lexical Semantic Change Detection System**

A computational tool for detecting and visualizing semantic shift in English words using contextual embeddings and the SemEval-2020 Task 1 dataset.

---

## 📖 Overview

LexiShift is a thesis project that implements an unsupervised approach to lexical semantic change detection. The system analyzes how word meanings evolve over time by comparing contextual embeddings from two diachronic corpora:

- **Corpus 1 (C1)**: 19th-century texts (1810–1860)
- **Corpus 2 (C2)**: 20th-century texts (1960–2010)

The prototype uses cosine similarity between temporal embedding centroids to compute a semantic shift score, then classifies each target word as either **Changed** or **Stable** based on a calibrated threshold (θ = 1.16).

---

## 🎯 Features

- ✅ Binary classification of 37 SemEval-2020 English target words
- ✅ Interactive web-based interface built with Streamlit
- ✅ Real-time semantic shift score computation
- ✅ PCA-based 2D visualization of embedding distributions
- ✅ Color-coded result verdicts and gauge bars
- ✅ Pre-computed contextual embeddings for all target words

---

## 🚀 Demo

![LexiShift Interface](screenshot.png)
*Example: Analysis of "attack_nn" showing semantic shift detection*

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jayztar/LexiShift.git
   cd LexiShift
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`

---

## 📂 Project Structure

```
LexiShift/
├── embeddings_by_word/       # Pre-computed word embeddings (JSON)
│   ├── attack.json
│   ├── plane.json
│   ├── tree.json
│   └── ... (37 total)
├── streamlit_app.py           # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .gitattributes             # Git LFS configuration
├── .gitignore
└── README.md
```

---

## 🎓 Dataset

This project uses the **SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection** English dataset.

### Target Words (37 total)

**Changed (16 words):**  
attack_nn, bit_nn, circle_vb, edge_nn, graft_nn, head_nn, land_nn, lass_nn, plane_nn, player_nn, prop_nn, rag_nn, record_nn, stab_nn, thump_nn, tip_vb

**Stable (21 words):**  
bag_nn, ball_nn, chairman_nn, contemplation_nn, donkey_nn, face_nn, fiction_nn, gas_nn, lane_nn, multitude_nn, ounce_nn, part_nn, pin_vb, quilt_nn, relationship_nn, risk_nn, savage_nn, stroke_vb, tree_nn, twist_nn, word_nn

---

## 🔬 Methodology

### 1. Embedding Extraction
Contextual word embeddings are extracted from both corpora using pre-trained language models. Each word has separate embedding sets for C1 and C2 contexts.

### 2. Shift Computation
```python
# Compute centroid (mean) vectors
μ₁ = mean(C1_embeddings)
μ₂ = mean(C2_embeddings)

# Calculate cosine similarity
cos_sim = dot(μ₁, μ₂) / (||μ₁|| × ||μ₂||)

# Derive semantic shift score
shift_score = 1 - cos_sim
```

### 3. Binary Classification
```python
if shift_score >= 1.16:
    label = "Changed"
else:
    label = "Stable"
```

### 4. Visualization
- **PCA projection**: Reduces high-dimensional embeddings to 2D for visualization
- **Centroid drift arrow**: Shows magnitude and direction of semantic change
- **Shift score gauge**: Visual indicator of shift magnitude relative to threshold

---

## 💻 Usage

1. **Enter a target word** in the input field (e.g., `plane_nn`, `tree_nn`, `attack_nn`)
2. Click **"Analyze"** or press Enter
3. View the results:
   - **Shift Score** and **Cosine Similarity**
   - **Verdict banner** (Changed/Stable)
   - **Shift score gauge** with threshold marker
   - **PCA scatter plot** showing C1 vs C2 embedding distributions

### Sample Queries

```
plane_nn    → Changed (aircraft sense emerged)
tree_nn     → Stable (consistent botanical meaning)
attack_nn   → Changed (military → broader usage)
```

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **Streamlit** — Web application framework
- **NumPy** — Numerical computation
- **scikit-learn** — PCA dimensionality reduction
- **Matplotlib** — Data visualization
- **Git LFS** — Large file storage for embeddings

---

## 📊 Results

The system achieves accurate classification on the SemEval-2020 English test set using:
- **Binary threshold**: θ = 1.16 (empirically calibrated)
- **Metric**: Cosine distance between temporal embedding centroids
- **Evaluation**: 37 annotated English target words

---

## 📝 Research Context

This system was developed as part of a Computer Science thesis at the **University of Science and Technology of Southern Philippines (USTP)**. The project explores computational approaches to historical linguistics and distributional semantics.

**Related Work:**
- Schlechtweg et al. (2020). SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection
- Hamilton et al. (2016). Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change

---

## 📄 License

This project is for academic purposes. Please contact the author for any commercial use.

---

## 👤 Author

**Shan Chai M. Manlunas**  
Computer Science Student  
University of Science and Technology of Southern Philippines

📧 manlunas.shanchai108@gmail.com  
🔗 [GitHub](https://github.com/dakkielle)

Jayza Joy Castillo
Computer Science Student
University of Science and Technology of Southern Philippines

📧 castillo.jayzajoy09@gmail.com
🔗 [GitHub](https://github.com/jayztar)

Zynnah Marie Ortiz
Computer Science Student
University of Science and Technology of Southern Philippines

📧 ortizzynnahmarie01@gmail.com
🔗 [GitHub](https://github.com/zinnah)

**Casey Jan D. Saguing**  
Computer Science Student  
University of Science and Technology of Southern Philippines

📧 saguing.casey0121@gmail.com  
🔗 [GitHub](https://github.com/mingkycas)

---

## 🙏 Acknowledgments

- SemEval-2020 Task 1 organizers for the benchmark dataset
- University of Science and Technology of Southern Philippines
- Thesis advisors and committee members

---

## 🔮 Future Work

- [ ] Extend to multilingual datasets (German, Latin, Swedish)
- [ ] Implement graded (continuous) semantic change scoring
- [ ] Add real-time embedding extraction for custom corpora
- [ ] Deploy as a public web service
- [ ] Support batch analysis for multiple words

---
