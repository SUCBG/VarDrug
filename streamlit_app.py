import streamlit as st
import joblib
import numpy as np
import torch
import requests
import plotly.express as px
import pandas as pd
import sklearn
import re
from variant_encoder import VMAE
from transformers import AutoTokenizer, AutoModel, BertConfig
from utils import smiles_to_fingerprint, get_gene_vec

# Set page configuration with a professional layout
st.set_page_config(
    page_title="VarDrug Pharmacogenomics Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }
        .title { font-size: 3rem; font-weight: 700; color: #60a5fa; text-align: center; margin-top: 1rem; margin-bottom: 1.5rem; }
        .subtitle { font-size: 1.25rem; color: #94a3b8; text-align: center; margin-bottom: 2rem; }
        .input-section { background-color: #1e293b; padding: 2rem; border-radius: 0.75rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; }
        .results-section { background-color: #1e293b; padding: 2rem; border-radius: 0.75rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .error { color: #fca5a5; font-weight: 600; background-color: #7f1d1d; padding: 1rem; border-radius: 0.5rem; }
        .success { color: #86efac; font-weight: 600; background-color: #14532d; padding: 1rem; border-radius: 0.5rem; }
        .stButton>button { 
            background-color: #3b82f6; 
            color: white; 
            font-weight: 600; 
            padding: 0.75rem 1.5rem; 
            border-radius: 0.5rem; 
            transition: background-color 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover { background-color: #1d4ed8; }
        .stTextInput>div>div>input { 
            background-color: #1e293b;
            border: 1px solid #334155; 
            color: #f8fafc;
            border-radius: 0.5rem; 
            padding: 0.75rem; 
            font-size: 1rem; 
        }
        .sidebar-title { font-size: 1.5rem; font-weight: 600; color: #60a5fa; margin-bottom: 1rem; }
        .dataframe { background-color: #1e293b !important; color: #f8fafc !important; }
        .stDataFrame { background-color: #1e293b !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all necessary models and resources."""
    try:
        device = torch.device("cpu")
        if sklearn.__version__ < '1.0':
            st.error(f"scikit-learn version {sklearn.__version__} is outdated. Please upgrade to 1.5.2 or later.")
            raise ValueError("Incompatible scikit-learn version")
        randfst_model = joblib.load('random_forest_model.pkl')
        stdScale = joblib.load('standard_scaler.pkl')
        vmae_model = VMAE().to(device)
        vmae_model.load_state_dict(torch.load('vmae_best_final.pth', map_location=device))
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        dna2_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config).to(device)
        return randfst_model, stdScale, vmae_model, tokenizer, dna2_model, device
    except Exception as e:
        st.markdown(f'<p class="error">Failed to load models: {str(e)}</p>', unsafe_allow_html=True)
        st.stop()

randfst_model, stdScale, vmae_model, tokenizer, dna2_model, device = load_models()

def validate_varid(varid):
    """Validate the variant ID format."""
    pattern = r'^([1-9]|1[0-9]|2[0-2]|X|Y)_[0-9]+_[A-Z]_[A-Z]$'
    return bool(re.match(pattern, varid))

def validate_smiles(smiles):
    """Basic validation for SMILES string (non-empty and contains valid characters)."""
    valid_chars = set("CcNnOoSsPpBrClF=()[]1234567890@#+-.")
    return smiles and all(char in valid_chars or char.isspace() for char in smiles)

def get_flank_200(chrom, pos):
    """Retrieve 101-bp sequence around the variant position."""
    try:
        server = "https://rest.ensembl.org"
        endpoint = f"/sequence/region/human/{chrom}:{pos-50}..{pos+50}?coord_system_version=GRCh38"
        headers = {"Content-Type": "application/json"}
        response = requests.get(server + endpoint, headers=headers)
        if response.status_code == 200:
            sequence_data = response.json()
            sequence = sequence_data.get('seq', '')
            if len(sequence) == 101:
                return sequence.upper()
            else:
                raise RuntimeError(f"Expected 101-bp sequence, got {len(sequence)} bp.")
        else:
            raise RuntimeError(f"API request failed with status {response.status_code}")
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve sequence: {str(e)}")

def get_gene_name(varid):
    """Retrieve gene name for the variant."""
    try:
        chrom = varid.split('_')[0].replace('chr', '')
        pos = int(varid.split('_')[1])
        server = "https://rest.ensembl.org"
        endpoint = f"/overlap/region/human/{chrom}:{pos}-{pos}?feature=gene;content-type=application/json"
        headers = {"Content-Type": "application/json"}
        response = requests.get(server + endpoint, headers=headers)
        if response.status_code == 200:
            genes = response.json()
            for gene in genes:
                if 'external_name' in gene:
                    return gene.get('external_name')
            return 'Unknown'
        else:
            raise RuntimeError(f"API request failed with status {response.status_code}")
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve gene name: {str(e)}")

def get_dnabert2_embedding(sequence, tokenizer, dna2_model, device):
    """Generate DNABERT-2 embedding for the sequence."""
    try:
        inputs = tokenizer([sequence], return_tensors='pt', padding=True, truncation=True, max_length=201).to(device)
        with torch.no_grad():
            hidden = dna2_model(inputs['input_ids'])[0]
            emb = torch.mean(hidden, dim=1)
        return emb.cpu().numpy()
    except Exception as e:
        raise RuntimeError(f"Failed to generate DNABERT-2 embedding: {str(e)}")

def extract_latent_representations(varid, vmae_model, tokenizer, dna2_model, device):
    """Extract latent representations for the variant."""
    try:
        chrom = varid.split('_')[0].replace('chr', '')
        pos = int(varid.split('_')[1])
        sequence = get_flank_200(chrom, pos)
        original_emb = get_dnabert2_embedding(sequence, tokenizer, dna2_model, device)[0]
        chrom_val = int(chrom) if chrom.isdigit() else 23 if chrom == 'X' else 24
        original_emb_tensor = torch.from_numpy(original_emb).float().unsqueeze(0).to(device)
        chrom_val_tensor = torch.tensor([chrom_val], device=device)
        with torch.no_grad():
            z, _, _, _, _ = vmae_model.encode(original_emb_tensor, chrom_val_tensor)
        return z.cpu().numpy()
    except Exception as e:
        raise RuntimeError(f"Failed to extract latent representations: {str(e)}")

def predict(varid, smiles, randfst_model, stdScale, vmae_model, tokenizer, dna2_model, device):
    """Predict pharmacogenomic outcomes."""
    try:
        gene_vec = np.array(get_gene_vec(get_gene_name(varid)))
        drug_vec = np.array(smiles_to_fingerprint(smiles))
        var_vec = extract_latent_representations(varid, vmae_model, tokenizer, dna2_model, device).flatten()
        input_vec = np.hstack([var_vec, gene_vec, drug_vec]).reshape(1, -1)
        scaled = stdScale.transform(input_vec)
        probs = randfst_model.predict_proba(scaled)[0]
        classes = np.array(['Dosage-decreased', 'Dosage-increased', 'Efficacy-decreased',
                            'Efficacy-increased', 'Toxicity-decreased', 'Toxicity-increased'])
        return classes, probs
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

# Sidebar for additional information
with st.sidebar:
    st.markdown('<div class="sidebar-title">About VarDrug</div>', unsafe_allow_html=True)
    st.markdown("""Predicting variant-drug interactions is crucial for precision medicine, yet the Pharmacogenomics Knowledge Base (PharmGKB) dataset (~11,000 samples) is underutilized in machine learning (ML) due to its small size. After filtering for variant mappings and excluding metabolizer-related conditions, we obtain ~4,000 samples for a six-class prediction task (increasing or decreasing of toxicity, efficacy and dosage). We introduce Vardrug, the first ML framework for variant-drug interaction prediction on PharmGKB. Vardrug employs a self-supervised VariantEncoder pre-trained on 100,000 samples, MolFormer for drug encoding, and gene co-expression profiles for variant encoding. Using SMOTE for class balancing and 5-fold cross-validation, we evaluate five ML models (CatBoost, RandomForest, ExtraTree, DecisionTree, SVC) against label encoding and rule-based baselines. RandomForest achieves a weighted F1 score of 0.66 and top-2 accuracy of 0.93, significantly outperforming baselines (best weighted F1: 0.39). Ablation studies highlight the VariantEncoder’s critical role, and a case study confirms biological plausibility by aligning predictions with known interactions. Vardrug provides a scalable, robust framework for pharmacogenomic prediction, with potential to guide personalized treatments and reduce adverse drug reactions.""")
    st.markdown("**Paper:** VarDrug: A Machine Learning Approach for Variant-Drug Interaction")

# Main content
st.markdown('<div class="title">VarDrug: Variant Drug Interaction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A Machine Learning Approach for Variant-Drug Interaction</div>', unsafe_allow_html=True)

# Input form
with st.container():
    st.markdown("### Input Variant and Drug Information")
    col1, col2 = st.columns(2)
    with col1:
        varid = st.text_input("Variant ID", placeholder="e.g., 1_123456_G_A", key="varid")
    with col2:
        smiles = st.text_input("SMILES String", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O", key="smiles")
    predict_button = st.button("Predict Outcomes", key="predict_button")

# Prediction and visualization
if predict_button:
    if not varid or not smiles:
        st.markdown('<p class="error">Please provide both Variant ID and SMILES string.</p>', unsafe_allow_html=True)
    elif not validate_varid(varid):
        st.markdown('<p class="error">Invalid Variant ID format. Use format like chr1_123456_G_A.</p>', unsafe_allow_html=True)
    elif not validate_smiles(smiles):
        st.markdown('<p class="error">Invalid SMILES string. Ensure it contains valid chemical notation.</p>', unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing variant-drug interaction..."):
            try:
                classes, probs = predict(varid, smiles, randfst_model, stdScale, vmae_model, tokenizer, dna2_model, device)
                st.markdown('<p class="success">Prediction completed successfully!</p>', unsafe_allow_html=True)

                # Detailed class probabilities
                df = pd.DataFrame({'Class': classes, 'Probability': probs})
                fig1 = px.bar(
                    df,
                    x='Class',
                    y='Probability',
                    title='Detailed Pharmacogenomic Outcome Probabilities',
                    color='Class',
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    height=500,
                    text='Probability'
                )
                fig1.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig1.update_layout(
                    xaxis_title="Outcome",
                    yaxis_title="Probability",
                    showlegend=False,
                    title_x=0.5,
                    font=dict(family="Inter", size=12),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='#1e293b',
                    font_color='#f8fafc',
                    xaxis=dict(tickangle=-45),
                    hoverlabel=dict(bgcolor="#1e293b", font_size=12)
                )

                # Aggregated probabilities
                agg_probs = {
                    'Dosage': probs[0] + probs[1],
                    'Efficacy': probs[2] + probs[3],
                    'Toxicity': probs[4] + probs[5]
                }
                agg_df = pd.DataFrame({
                    'Category': list(agg_probs.keys()),
                    'Probability': list(agg_probs.values())
                })
                fig2 = px.bar(
                    agg_df,
                    x='Category',
                    y='Probability',
                    title='Aggregated Outcome Probabilities',
                    color='Category',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    height=500,
                    text='Probability'
                )
                fig2.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig2.update_layout(
                    xaxis_title="Category",
                    yaxis_title="Probability",
                    showlegend=False,
                    title_x=0.5,
                    font=dict(family="Inter", size=12),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='#1e293b',
                    font_color='#f8fafc',
                    hoverlabel=dict(bgcolor="#1e293b", font_size=12)
                )

                # Display plots
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig2, use_container_width=True)

                # Display raw probabilities in a table
                st.markdown("### Raw Probability Values")
                st.dataframe(df.style.format({"Probability": "{:.2%}"}).set_properties(**{
                    'background-color': '#1e293b',
                    'color': '#f8fafc',
                    'border-color': '#334155'
                }))

            except Exception as e:
                st.markdown(f'<p class="error">Error: {str(e)}</p>', unsafe_allow_html=True)