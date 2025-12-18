# CAFA 6: Protein Function Prediction 

This repository contains the solution for the **Kaggle CAFA 6** competition. We utilize a weighted ensemble of three distinct models: **Logistic Regression**, **Multi-Layer Perceptron (MLP)**, and **k-Nearest Neighbors (k-NN)**, leveraging both sequence-based features and Protein Language Model (ESM2) embeddings.

## Methodology & Mathematical Formulation

### 1. Feature Extraction

We employ two types of feature representations:
* **Hand-crafted Features ($\mathbf{x}_{seq}$):** Composed of $k$-mer counts ($k=1$) and positional amino acid distributions.
* **ESM2 Embeddings ($\mathbf{x}_{emb}$):** We use the `esm2_t33_650M_UR50D` model to extract representations, where for a protein sequence $S$ of length $L$:
    $$\mathbf{x}_{emb} = \text{Pool}(\text{ESM2}(S)) \in \mathbb{R}^{1280}$$

### 2. Models Architecture

#### A. Logistic Regression (Baseline)
For each Gene Ontology (GO) aspect, we train a multi-label logistic regression model. The prediction for term $j$ is given by:

$$\hat{y}_j = \sigma(\mathbf{w}_j^T \mathbf{x}_{seq} + b_j)$$

**Loss Function:** We use Binary Cross Entropy weighted by **Information Accretion (IA)** to prioritize informative terms:

$$\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} \alpha_j \cdot [y_{ij} \log(\hat{y}_{ij}) + (1 - y_{ij}) \log(1 - \hat{y}_{ij})]$$

Where:
* $\alpha_j$: The IA weight for GO term $j$.
* $C$: Total number of GO terms.

#### B. Multi-Layer Perceptron (MLP)
We use a 3-layer neural network trained on ESM2 embeddings. The architecture can be described as:

$$\mathbf{h}_1 = \text{ReLU}(\text{BN}(\mathbf{W}_1 \mathbf{x}_{emb} + \mathbf{b}_1))$$
$$\mathbf{h}_2 = \text{ReLU}(\text{BN}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2))$$
$$\hat{\mathbf{y}} = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3$$

* **Dropout** ($p=0.3$) is applied after activation layers.
* Dimensions: $1280 \to 1024 \to 512 \to N_{classes}$.

#### C. k-Nearest Neighbors (k-NN)
We perform label propagation based on cosine similarity in the embedding space. For a query protein $u$ and training protein $v$:

$$\text{sim}(u, v) = \frac{\mathbf{x}_{emb}^{(u)} \cdot \mathbf{x}_{emb}^{(v)}}{\|\mathbf{x}_{emb}^{(u)}\| \|\mathbf{x}_{emb}^{(v)}\|}$$

The score for term $t$ is the weighted average of the top-$k$ neighbors $\mathcal{N}_k(u)$:

$$\text{Score}(u, t) = \frac{\sum_{v \in \mathcal{N}_k(u)} \text{sim}(u, v) \cdot y_{v,t}}{\sum_{v \in \mathcal{N}_k(u)} \text{sim}(u, v)}$$

### 3. Ensemble Strategy

The final prediction score $S_{final}$ is a weighted linear combination of the three models:

$$S_{final} = w_{log} \cdot S_{log} + w_{knn} \cdot S_{knn} + w_{mlp} \cdot S_{mlp}$$

**Weights configuration:**
* $w_{log} = 0.1$
* $w_{knn} = 0.4$
* $w_{mlp} = 0.5$

## 🚀 Project Structure

```text
├── data/                   # Input data
├── preprocessing/          # Feature extraction scripts (K-mers, ESM2 embeddings)
├── models/
│   ├── Logistics.py        # PyTorch implementation of weighted Logistic Regression
│   ├── MLP.py              # Neural Network implementation
│   └── KNN.py              # k-NN inference script
├── ensemble.py             # Script for weighted blending
└── README.md
