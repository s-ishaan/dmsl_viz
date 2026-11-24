"""
Tile 3: Scaled Dot-Product & Multi-Head Attention
Focus: Slides 352-353 - Matrix operations of self-attention (Q, K, V),
scaled dot-product similarity, and multi-head subspace specialization.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.plotting import (
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def get_token_matrices(tokens: list, d_model: int = 16, seed: int = 42) -> dict:
    """Generates synthetic embeddings and projection weight matrices for demonstration."""
    np.random.seed(seed)
    n = len(tokens)

    # Synthetic token embeddings X (N x d_model)
    X = np.random.randn(n, d_model) * 0.8

    # Define 3 specialized heads with distinct linguistic projection biases
    heads = {}
    head_names = ["Head 1 (Modifier-Noun Affinity)", "Head 2 (Sequential / Adjacent Focus)", "Head 3 (Global Sentence Cohesion)"]

    for h_idx, h_name in enumerate(head_names):
        d_k = d_model // 2
        W_Q = np.random.randn(d_model, d_k) * 0.5
        W_K = np.random.randn(d_model, d_k) * 0.5
        W_V = np.random.randn(d_model, d_k) * 0.5

        # Inject realistic linguistic bias
        if h_idx == 0:
            # Head 1: High similarity between "delicious" and "main course", "truly" and "delicious"
            for i, ti in enumerate(tokens):
                for j, tj in enumerate(tokens):
                    if (ti in ["delicious", "truly"] and tj in ["course", "main"]) or (ti == "course" and tj == "delicious"):
                        W_Q[:, 0] += 0.3
                        W_K[:, 0] += 0.3
        elif h_idx == 1:
            # Head 2: High similarity for adjacent positional tokens |i - j| <= 1
            for i in range(n):
                for j in range(n):
                    if abs(i - j) == 1:
                        W_Q[:, 1] += 0.2
                        W_K[:, 1] += 0.2
        else:
            # Head 3: Global subject-verb agreement ("course" and "was", "the" and "course")
            for i, ti in enumerate(tokens):
                for j, tj in enumerate(tokens):
                    if ti in ["course", "the"] and tj in ["was", "course"]:
                        W_Q[:, 2] += 0.25
                        W_K[:, 2] += 0.25

        Q = np.dot(X, W_Q)
        K = np.dot(X, W_K)
        V = np.dot(X, W_V)

        heads[h_name] = {
            "W_Q": W_Q,
            "W_K": W_K,
            "W_V": W_V,
            "Q": Q,
            "K": K,
            "V": V,
            "d_k": d_k,
        }

    return {"X": X, "heads": heads, "d_model": d_model}


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Matrix & Multi-Head Knobs")

            sentence_choice = st.selectbox(
                "Input Sequence X",
                [
                    "the main course was truly delicious",
                    "attention without recurrence is all you need",
                ],
                index=0,
                key="mha_sentence",
                help="Token sequence to transform through the self-attention matrix pipeline.",
            )

            tokens = sentence_choice.split()
            n_tokens = len(tokens)

            active_head_name = st.selectbox(
                "Active Attention Head",
                [
                    "Head 1 (Modifier-Noun Affinity)",
                    "Head 2 (Sequential / Adjacent Focus)",
                    "Head 3 (Global Sentence Cohesion)",
                ],
                index=0,
                key="mha_head",
                help="Inspect the distinct projection subspaces learned across different attention heads.",
            )

            enable_scaling = st.toggle(
                "Scale by 1 / √d_k",
                value=True,
                key="mha_scaling",
                help="Vaswani et al. scaling factor to counteract large dot-product magnitudes and prevent softmax saturation.",
            )

            d_k_display = st.select_slider(
                "Head Key Dimension d_k",
                options=[4, 8, 16, 32, 64],
                value=8,
                key="mha_dk",
                help="Dimension of query and key projections. Larger d_k increases the dot-product variance.",
            )

    mat_data = get_token_matrices(tokens, d_model=16)
    head_info = mat_data["heads"][active_head_name]
    Q = head_info["Q"]
    K = head_info["K"]
    V = head_info["V"]
    d_k = d_k_display

    # Compute dot products Q K^T
    raw_dot_products = np.dot(Q, K.T)

    # Scale factor
    scale_factor = np.sqrt(d_k) if enable_scaling else 1.0
    scaled_logits = raw_dot_products / scale_factor

    # Softmax row-wise
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    attention_matrix = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Output contextual embeddings Z = A * V
    output_Z = np.dot(attention_matrix, V)

    # Measure entropy of attention distribution to evaluate sharpness
    entropy_rows = -np.sum(attention_matrix * np.log(np.maximum(1e-12, attention_matrix)), axis=-1)
    mean_entropy = float(np.mean(entropy_rows))
    max_attn_weight = float(np.max(attention_matrix))

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sequence Length N", f"{n_tokens} Tokens")
        m2.metric("Peak Attention Weight", f"{max_attn_weight * 100:.1f}%")
        m3.metric("Attention Entropy", f"{mean_entropy:.2f} nats", help="Low entropy indicates sharp focused attention; high entropy indicates diffuse attention.")
        m4.metric("Scaling Factor", f"1 / √{d_k} = {1.0/np.sqrt(d_k):.3f}" if enable_scaling else "Disabled (1.00)")

        tab_heatmap, tab_pipeline, tab_multi_compare = st.tabs([
            "🔥 Attention Weight Heatmap A",
            "🧮 Full Matrix Algebra Pipeline",
            "👥 Multi-Head Subspace Comparison",
        ])

        with tab_heatmap:
            st.markdown(r"##### 🔥 Attention Matrix $A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times N}$")

            fig_hm = go.Figure()

            fig_hm.add_trace(
                go.Heatmap(
                    z=attention_matrix,
                    x=tokens,
                    y=tokens,
                    colorscale="YlOrRd",
                    colorbar=dict(
                        title=dict(text="Weight α", font=dict(color="#CBD5E1", size=10)),
                        tickfont=dict(color="#94A3B8"),
                    ),
                    text=[[f"{val*100:.1f}%" for val in row] for row in attention_matrix],
                    texttemplate="%{text}",
                    textfont=dict(size=10, color="#0F172A"),
                    hovertemplate="Query (Row): %{y}<br>Key (Col): %{x}<br>Attention: %{z:.3f}<extra></extra>",
                )
            )

            fig_hm.update_layout(
                height=440,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(title="Keys (k_j)", gridcolor="#1E293B", tickfont=dict(color="#CBD5E1", size=10), side="bottom"),
                yaxis=dict(title="Queries (q_i)", gridcolor="#1E293B", tickfont=dict(color="#CBD5E1", size=10), autorange="reversed"),
                margin=dict(l=40, r=20, t=20, b=45),
            )

            st.plotly_chart(fig_hm, width="stretch", key="mha_fig_hm")

        with tab_pipeline:
            st.markdown("##### 🧮 Matrix Algebra Pipeline: $X \to (Q, K, V) \to Q K^T \to \text{softmax} \to Z$")

            col_p1, col_p2 = st.columns(2)

            with col_p1:
                st.markdown(f"**1. Raw Dot-Product Similarities $Q K^T$ (Range: [{np.min(raw_dot_products):.1f}, {np.max(raw_dot_products):.1f}])**")
                fig_qk = go.Figure(
                    go.Heatmap(
                        z=raw_dot_products,
                        x=tokens,
                        y=tokens,
                        colorscale="Blues",
                        text=[[f"{val:.1f}" for val in row] for row in raw_dot_products],
                        texttemplate="%{text}",
                        textfont=dict(size=8, color="#F8FAFC"),
                    )
                )
                fig_qk.update_layout(height=280, paper_bgcolor=BG_PAPER, plot_bgcolor=BG_PLOT, margin=dict(l=20, r=20, t=20, b=30), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_qk, width="stretch", key="mha_fig_qk")

            with col_p2:
                st.markdown(f"**2. Scaled Logits $\\frac{{Q K^T}}{{\\sqrt{{d_k}}}}$ (Scale = {scale_factor:.2f})**")
                fig_scaled = go.Figure(
                    go.Heatmap(
                        z=scaled_logits,
                        x=tokens,
                        y=tokens,
                        colorscale="Teal",
                        text=[[f"{val:.1f}" for val in row] for row in scaled_logits],
                        texttemplate="%{text}",
                        textfont=dict(size=8, color="#F8FAFC"),
                    )
                )
                fig_scaled.update_layout(height=280, paper_bgcolor=BG_PAPER, plot_bgcolor=BG_PLOT, margin=dict(l=20, r=20, t=20, b=30), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_scaled, width="stretch", key="mha_fig_scaled")

            st.markdown(
                r"""
                - **Step 1**: Linear projections $Q = X W^Q$, $K = X W^K$, $V = X W^V$ allow each token to query for relevant content across different geometric axes.
                - **Step 2**: Pairwise dot-product $Q K^T$ measures inner-product similarity between every query $q_i$ and every key $k_j$.
                - **Step 3**: Division by $\sqrt{d_k}$ stabilizes the variance of the dot products. Without scaling, dot products with large dimension $d_k$ blow up to extreme magnitudes, causing softmax to saturate with virtually zero gradients.
                - **Step 4**: Multiplying normalized attention weights $A$ by Value matrix $V$ computes the new contextual embeddings $Z = A V$.
                """
            )

        with tab_multi_compare:
            st.markdown("##### 👥 Multi-Head Attention Specialization: Different 'Views' of the Same Text")

            head_names = list(mat_data["heads"].keys())
            h_col1, h_col2, h_col3 = st.columns(3)

            for col_h, h_name in zip([h_col1, h_col2, h_col3], head_names):
                with col_h:
                    st.caption(f"**{h_name.split('(')[0].strip()}**")
                    h_q = mat_data["heads"][h_name]["Q"]
                    h_k = mat_data["heads"][h_name]["K"]
                    h_qk = np.dot(h_q, h_k.T) / np.sqrt(d_k)
                    h_exp = np.exp(h_qk - np.max(h_qk, axis=-1, keepdims=True))
                    h_attn = h_exp / np.sum(h_exp, axis=-1, keepdims=True)

                    fig_sub = go.Figure(
                        go.Heatmap(
                            z=h_attn,
                            x=tokens,
                            y=tokens,
                            colorscale="Viridis",
                            showscale=False,
                        )
                    )
                    fig_sub.update_layout(
                        height=240,
                        paper_bgcolor=BG_PAPER,
                        plot_bgcolor=BG_PLOT,
                        margin=dict(l=10, r=10, t=10, b=25),
                        xaxis=dict(tickfont=dict(size=7)),
                        yaxis=dict(tickfont=dict(size=7), autorange="reversed"),
                    )
                    st.plotly_chart(fig_sub, width="stretch", key=f"mha_sub_{h_name[:6]}")

            st.info(
                "💡 **Multi-Head Power**: Instead of performing a single attention function with $d_{model}$-dimensional queries, keys, and values, "
                "multi-head attention projects queries, keys, and values $h$ times with different learned linear projections. "
                "This allows the model to jointly attend to information from different representation subspaces at different positions."
            )

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Scaled Dot-Product & Multi-Head Attention", expanded=False):
        st.markdown(
            r"""
            **Scaled Dot-Product Attention**:
            - Given query matrix $Q \in \mathbb{R}^{N \times d_k}$, key matrix $K \in \mathbb{R}^{M \times d_k}$, and value matrix $V \in \mathbb{R}^{M \times d_v}$:
              $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
            - **Why Scale by $\frac{1}{\sqrt{d_k}}$?**:
              - Assume components of $\mathbf{q}$ and $\mathbf{k}$ are independent random variables with zero mean and unit variance.
              - The dot product $\mathbf{q}^T \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ has mean $0$ and variance $d_k$:
                $$\mathbb{E}[\mathbf{q}^T \mathbf{k}] = 0, \quad \text{Var}(\mathbf{q}^T \mathbf{k}) = d_k$$
              - For large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients (vanishing gradient problem).
              - Scaling by $\frac{1}{\sqrt{d_k}}$ restores the variance to $1$, keeping gradients healthy during backpropagation.

            **Multi-Head Attention**:
            - Multi-head attention allows the model to jointly attend to information from different representation subspaces:
              $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
              $$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
            - Where parameter matrices are:
              $$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}, \quad W_i^K \in \mathbb{R}^{d_{model} \times d_k}, \quad W_i^V \in \mathbb{R}^{d_{model} \times d_v}, \quad W^O \in \mathbb{R}^{h d_v \times d_{model}}$$
            - Commonly $h = 8$ heads with $d_k = d_v = d_{model} / h = 64$ such that the total computational cost is similar to single-head attention with full dimensionality.
            """
        )
