# Vardrug: Machine Learning Framework for Predicting Variant-Drug Interactions

## 🧬 Abstract

Predicting variant-drug interactions is crucial for precision medicine, yet the Pharmacogenomics Knowledge Base (PharmGKB) dataset (~11,000 samples) remains underutilized in machine learning (ML) due to its small size. After filtering for variant mappings and excluding metabolizer-related conditions, we obtain ~4,000 samples for a six-class prediction task (increase/decrease of toxicity, efficacy, and dosage).

We introduce **Vardrug**, the first ML framework for variant-drug interaction prediction on PharmGKB. Vardrug combines:

- 🧠 **VariantEncoder**: A self-supervised encoder pre-trained on 100,000 variant samples.
- 💊 **MolFormer**: A transformer-based drug encoder.
- 🧬 **Gene co-expression profiles**: For enhanced variant representation.

We use **SMOTE** for class balancing and apply **5-fold cross-validation** to evaluate five ML models: `RandomForest`, `CatBoost`, `ExtraTree`, `DecisionTree`, and `SVC`. These are compared against label encoding and rule-based baselines.

**Key Results**:
- **RandomForest** achieves:
  - 🎯 Weighted F1 Score: **0.66**
  - 🎯 Top-2 Accuracy: **0.93**
- Outperforms all baselines (best baseline weighted F1: **0.39**).
- Ablation studies confirm the critical impact of VariantEncoder.
- A case study validates biological plausibility by aligning predictions with known interactions.

**Vardrug** offers a robust and scalable framework to enhance pharmacogenomic predictions, guiding personalized treatments and reducing adverse drug reactions.

> *KarimiNejad et al., 2025*

---

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install dependencies**:
   ```bash
   streamlit run streamlit_app.py
   ```

## 📸 Screenshot
![VarDrug](App.png)



## 📚 Citation
KarimiNejad M., et al. (2025). Vardrug: A Machine Learning Framework for Variant-Drug Interaction Prediction. [Preprint].

