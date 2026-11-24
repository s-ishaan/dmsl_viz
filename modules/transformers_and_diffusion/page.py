"""
Module Page: Transformers & Diffusion Generative Models
Conceptual Flow: Subword Tokenization & Semantic Vector Arithmetic,
Self-Attention Coreference Disambiguation, Scaled Dot-Product & Multi-Head Projections,
Forward Gaussian Markov Diffusion & Closed-Form Reparameterization,
DDPM Reverse Denoising & Score-Matching Sampling, and Multimodal Classifier-Free Guidance.
"""

import streamlit as st
from .tiles import (
    tokenization_and_embeddings,
    self_attention_disambiguation,
    multihead_attention_matrix,
    forward_diffusion_reparameterization,
    ddpm_reverse_sampling,
    multimodal_guidance_and_diffusion,
)


def render():
    st.title("Transformers & Diffusion Generative Models")
    st.caption("Attention Architectures, Denoising Probabilistic Diffusion & Multimodal Guidance")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔤 1. Tokenization & Embeddings",
        "🎯 2. Self-Attention & Context",
        "⚡ 3. Multi-Head Attention",
        "🌫️ 4. Forward Diffusion",
        "🔄 5. DDPM Reverse Sampling",
        "🎨 6. Multimodal Guidance",
    ])

    with tab1:
        tokenization_and_embeddings.render()

    with tab2:
        self_attention_disambiguation.render()

    with tab3:
        multihead_attention_matrix.render()

    with tab4:
        forward_diffusion_reparameterization.render()

    with tab5:
        ddpm_reverse_sampling.render()

    with tab6:
        multimodal_guidance_and_diffusion.render()
