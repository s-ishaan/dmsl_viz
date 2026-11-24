"""
Tile 1: Tokenization, Semantic Vector Spaces & Word Embeddings
Focus: Discrete Tokenization, Continuous Semantic Vector Spaces, Vector Analogy Algebra,
and Sinusoidal Positional Encodings.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.plotting import (
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def get_curated_embeddings() -> dict:
    """
    Returns curated 3D semantic coordinates for illustrative vocabulary clusters.
    Designed to faithfully demonstrate cosine similarity and linear analogy arithmetic.
    """
    words = {
        # Royalty Cluster (Blue / Purple)
        "king": np.array([2.2, 3.8, 1.5]),
        "queen": np.array([2.1, 3.9, -1.4]),
        "man": np.array([0.5, 3.7, 1.4]),
        "woman": np.array([0.4, 3.8, -1.5]),
        "father": np.array([-0.8, 3.5, 1.2]),
        "mother": np.array([-0.9, 3.6, -1.6]),
        "boy": np.array([0.2, 2.5, 1.3]),
        "girl": np.array([0.1, 2.6, -1.4]),

        # Geography & Capitals Cluster (Amber / Orange)
        "paris": np.array([-3.8, 1.2, 2.2]),
        "france": np.array([-3.2, 0.4, 2.1]),
        "berlin": np.array([-3.7, 1.1, -1.8]),
        "germany": np.array([-3.1, 0.3, -1.9]),
        "london": np.array([-3.9, 1.3, 0.2]),
        "england": np.array([-3.3, 0.5, 0.1]),
        "rome": np.array([-3.6, 1.0, 3.8]),
        "italy": np.array([-3.0, 0.2, 3.7]),

        # Vehicles Cluster (Rose / Red)
        "car": np.array([1.2, -3.2, 2.0]),
        "bus": np.array([1.8, -3.4, 1.8]),
        "train": np.array([0.4, -3.1, 2.2]),
        "bicycle": np.array([2.5, -3.6, 1.5]),
        "vehicle": np.array([1.5, -2.8, 2.1]),

        # Fruits & Food Cluster (Emerald / Green)
        "apple": np.array([3.5, 0.5, -2.8]),
        "banana": np.array([3.8, -0.4, -2.5]),
        "orange": np.array([4.2, 0.8, -3.1]),
        "fruit": np.array([3.6, 0.2, -2.6]),
    }
    return words


def compute_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-9 or norm_v < 1e-9:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def generate_positional_encoding(seq_len: int, d_model: int, n: float = 10000.0) -> np.ndarray:
    """Generates standard Vaswani et al. (2017) sinusoidal positional encodings."""
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            div_term = np.exp(i * -np.log(n) / d_model)
            pe[pos, i] = np.sin(pos * div_term)
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos * div_term)
    return pe


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    words_dict = get_curated_embeddings()

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Embedding & Vector Knobs")

            analogy_choice = st.selectbox(
                "Semantic Analogy Equation",
                [
                    "king - man + woman ≈ queen",
                    "paris - france + germany ≈ berlin",
                    "rome - italy + england ≈ london",
                    "apple - fruit + vehicle ≈ car",
                ],
                index=0,
                key="te_analogy",
                help="Linear arithmetic relationship in semantic embedding vector space.",
            )

            lambda_scale = st.slider(
                "Analogy Offset Scale λ",
                min_value=0.0,
                max_value=1.5,
                value=1.0,
                step=0.05,
                key="te_lambda",
                help="Controls the magnitude of the added gender/relation vector.",
            )

            dist_metric = st.radio(
                "Similarity Metric",
                ["Cosine Similarity", "Euclidean Distance"],
                index=0,
                key="te_metric",
            )

            with st.expander("⚙️ Positional Encoding Settings", expanded=False):
                seq_length = st.slider("Sequence Length L", 8, 48, 24, 4, key="te_seq_len")
                d_model = st.slider("Embedding Dim d_model", 16, 64, 32, 8, key="te_dmodel")
                base_freq = st.number_input("Base Wavelength (n)", 1000, 50000, 10000, 1000, key="te_base_freq")

            with st.expander("⚙️ Tokenizer Settings", expanded=False):
                sample_text = st.text_input(
                    "Input Text to Tokenize",
                    value="We're no strangers to love",
                    key="te_sample_text",
                )

    # Parse selected analogy
    if "king" in analogy_choice:
        w_a, w_b, w_c, w_target = "king", "man", "woman", "queen"
    elif "paris" in analogy_choice:
        w_a, w_b, w_c, w_target = "paris", "france", "germany", "berlin"
    elif "rome" in analogy_choice:
        w_a, w_b, w_c, w_target = "rome", "italy", "england", "london"
    else:
        w_a, w_b, w_c, w_target = "apple", "fruit", "vehicle", "car"

    v_a = words_dict[w_a]
    v_b = words_dict[w_b]
    v_c = words_dict[w_c]
    # Resulting probe vector: v_a - v_b + lambda * v_c
    v_query = v_a - v_b + lambda_scale * v_c

    # Find ranking of words closest to v_query
    ranked_words = []
    for w, vec in words_dict.items():
        if w in [w_a, w_b, w_c]:
            continue  # ignore input operand words
        cos_sim = compute_cosine_similarity(v_query, vec)
        euc_dist = float(np.linalg.norm(v_query - vec))
        ranked_words.append((w, cos_sim, euc_dist))

    if dist_metric == "Cosine Similarity":
        ranked_words.sort(key=lambda x: x[1], reverse=True)
        best_word, best_score = ranked_words[0][0], ranked_words[0][1]
        score_label = f"Cosine: {best_score:.3f}"
    else:
        ranked_words.sort(key=lambda x: x[2], reverse=False)
        best_word, best_score = ranked_words[0][0], ranked_words[0][2]
        score_label = f"Distance: {best_score:.2f}"

    # Tokenizer mock mapping
    toy_vocab = {
        "we": 1678,
        "'re": 2351,
        "no": 912,
        "strangers": 40721,
        "to": 311,
        "love": 3021,
        "the": 101,
        "animal": 4132,
        "didn't": 3054,
        "cross": 2892,
        "street": 2395,
        "because": 2138,
        "it": 2009,
        "was": 2001,
        "too": 2205,
        "tired": 5458,
        "wide": 2898,
        "attention": 3089,
        "is": 2003,
        "all": 2035,
        "you": 2017,
        "need": 2342,
    }

    # Clean words for token display
    raw_tokens = sample_text.replace("'", " '").split()
    token_ids = []
    token_displays = []
    for tok in raw_tokens:
        tok_clean = tok.lower().strip(".,!?")
        tid = toy_vocab.get(tok_clean, abs(hash(tok_clean)) % 30000 + 1000)
        token_ids.append(tid)
        token_displays.append(tok)

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Analogy", f"{best_word.capitalize()}", f"{'Target Match! 🎉' if best_word == w_target else 'In Progress'}")
        m2.metric("Similarity Score", score_label)
        m3.metric("Semantic Dim D", "D = 256 → 3D")
        m4.metric("Extracted Tokens", f"{len(token_ids)} Tokens")

        tab_vector_space, tab_positional, tab_tokenizer = st.tabs([
            "🌐 3D Semantic Vector Space & Analogies",
            "📐 Positional Encoding Harmonic Waves",
            "🔤 Subword Tokenizer & Integer IDs",
        ])

        with tab_vector_space:
            st.markdown(f"##### 🌐 Vector Space Arithmetic: $\\mathbf{{v}}_{{\\text{{{w_a}}}}} - \\mathbf{{v}}_{{\\text{{{w_b}}}}} + {lambda_scale:.2f} \\cdot \\mathbf{{v}}_{{\\text{{{w_c}}}}} \\to \\mathbf{{v}}_{{\\text{{{best_word}}}}}$")

            fig_3d = go.Figure()

            # Color categories
            category_styles = {
                "Royalty": (["king", "queen", "man", "woman", "father", "mother", "boy", "girl"], "#818CF8"),
                "Geography": (["paris", "france", "berlin", "germany", "london", "england", "rome", "italy"], "#F59E0B"),
                "Vehicles": (["car", "bus", "train", "bicycle", "vehicle"], "#F43F5E"),
                "Fruits": (["apple", "banana", "orange", "fruit"], "#10B981"),
            }

            for cat_name, (cat_words, cat_color) in category_styles.items():
                xs = [words_dict[w][0] for w in cat_words]
                ys = [words_dict[w][1] for w in cat_words]
                zs = [words_dict[w][2] for w in cat_words]
                labels = [w.capitalize() for w in cat_words]

                fig_3d.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="markers+text",
                        marker=dict(size=5.5, color=cat_color, opacity=0.85),
                        text=labels,
                        textposition="top center",
                        textfont=dict(color="#CBD5E1", size=10),
                        name=cat_name,
                        hoverinfo="text",
                    )
                )

            # Highlight Query vector resulting from arithmetic
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[v_query[0]],
                    y=[v_query[1]],
                    z=[v_query[2]],
                    mode="markers+text",
                    marker=dict(size=9.0, color="#38BDF8", symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
                    text=[f"Result Vector ({best_word})"],
                    textposition="bottom center",
                    textfont=dict(color="#38BDF8", size=11),
                    name="Query Vector",
                )
            )

            # Draw arithmetic parallelogram line: w_a to w_b, w_c to query
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[v_b[0], v_a[0]],
                    y=[v_b[1], v_a[1]],
                    z=[v_b[2], v_a[2]],
                    mode="lines",
                    line=dict(color="#94A3B8", width=3, dash="dash"),
                    name=f"Offset Vector ({w_b} → {w_a})",
                )
            )
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[v_c[0], v_query[0]],
                    y=[v_c[1], v_query[1]],
                    z=[v_c[2], v_query[2]],
                    mode="lines",
                    line=dict(color="#38BDF8", width=4),
                    name=f"Applied Offset ({w_c} → Query)",
                )
            )

            fig_3d.update_layout(
                height=450,
                paper_bgcolor=BG_PAPER,
                margin=dict(l=10, r=10, t=10, b=10),
                scene=dict(
                    xaxis=dict(title="Dim 1", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                    yaxis=dict(title="Dim 2", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                    zaxis=dict(title="Dim 3", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.05,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#CBD5E1", size=9),
                ),
            )

            st.plotly_chart(fig_3d, width="stretch", key="te_fig_3d")

        with tab_positional:
            st.markdown(r"##### 📐 Positional Encoding Spectrum: $PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \, PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$")

            pe_matrix = generate_positional_encoding(seq_length, d_model, float(base_freq))

            fig_pe = go.Figure()

            # Heatmap of PE matrix
            fig_pe.add_trace(
                go.Heatmap(
                    z=pe_matrix,
                    x=[f"i={i}" for i in range(d_model)],
                    y=[f"pos={p}" for p in range(seq_length)],
                    colorscale="Viridis",
                    colorbar=dict(
                        title=dict(text="PE Value", font=dict(color="#CBD5E1", size=10)),
                        tickfont=dict(color="#94A3B8"),
                    ),
                    hovertemplate="Position: %{y}<br>Dim: %{x}<br>PE: %{z:.3f}<extra></extra>",
                )
            )

            fig_pe.update_layout(
                height=430,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(title="Embedding Dimension index i", gridcolor="#1E293B", tickfont=dict(color="#94A3B8", size=8)),
                yaxis=dict(title="Token Position pos", gridcolor="#1E293B", tickfont=dict(color="#94A3B8", size=8), autorange="reversed"),
                margin=dict(l=40, r=20, t=15, b=45),
            )

            st.plotly_chart(fig_pe, width="stretch", key="te_fig_pe")

        with tab_tokenizer:
            st.markdown("##### 🔤 Subword Tokenization & Discrete Vectorization Pipeline")

            # Render colored token blocks
            token_palette = ["#38BDF8", "#F43F5E", "#10B981", "#F59E0B", "#A855F7", "#EC4899", "#6366F1"]
            chips_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;">'
            for idx, (tok, tid) in enumerate(zip(token_displays, token_ids)):
                color = token_palette[idx % len(token_palette)]
                chips_html += f"""
                <div style="background-color: {color}22; border: 1px solid {color}; border-radius: 6px; padding: 6px 12px; text-align: center;">
                    <div style="font-weight: 700; color: #F8FAFC; font-size: 1.05rem;">{tok}</div>
                    <div style="font-size: 0.75rem; color: {color}; margin-top: 2px;">ID: {tid}</div>
                </div>
                """
            chips_html += '</div>'
            st.markdown(chips_html, unsafe_allow_html=True)

            st.code(f"Raw Text Input:   \"{sample_text}\"\nTokens Extracted: {token_displays}\nVectorized IDs:   {token_ids}", language="python")

            st.markdown(
                r"""
                - **Tokenization**: Breaks raw natural text into manageable linguistic sub-word tokens. In English, 1 token roughly corresponds to 3/4 of a word (or ~4 characters).
                - **Vectorization**: Each unique token is mapped to a standardized integer ID from the model's vocabulary dictionary $\mathcal{D}$.
                - **Embedding Lookup**: Each integer ID is subsequently converted into a continuous floating-point vector $\mathbf{x} \in \mathbb{R}^D$ via an embedding lookup matrix $E \in \mathbb{R}^{|\mathcal{D}| \times D}$.
                """
            )

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Tokenization, Semantic Spaces & Positional Encodings", expanded=False):
        st.markdown(
            r"""
            **Tokenization & Vectorization**:
            - Before feeding text into a neural architecture, raw strings must be transformed into discrete numerical indices.
            - Given a vocabulary dictionary $\mathcal{D} = \{w_1, w_2, \dots, w_V\}$ with cardinality $|\mathcal{D}| = V$, tokenization segments sentence $S$ into a sequence of token IDs:
              $$S = (t_1, t_2, \dots, t_N), \quad t_i \in \{1, 2, \dots, V\}$$
            - An embedding layer maps each integer token ID $t_i$ to a dense continuous vector:
              $$\mathbf{x}_i = E[t_i] \in \mathbb{R}^D$$

            **Semantic Vector Spaces & Vector Arithmetic**:
            - Neural language representations trained on vast text corpora learn linear relational properties.
            - The distance between tokens encodes semantic similarity using **Cosine Similarity**:
              $$S_C(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \in [-1, 1]$$
            - Linear relationships manifest as consistent spatial displacement vectors:
              $$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} \approx \mathbf{v}_{\text{queen}} - \mathbf{v}_{\text{woman}} \implies \mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

            **Sinusoidal Positional Encoding (Vaswani et al., 2017)**:
            - Because Transformer self-attention contains no recurrent loops or convolutions, the model is inherently **permutation-invariant**.
            - To inject order information, positional vectors $PE \in \mathbb{R}^{N \times D}$ are added directly to the token embeddings:
              $$\mathbf{z}_i = \mathbf{x}_i + PE_{(i)}$$
            - Using geometric progressions of sinusoidal wavelengths allows the model to learn relative positional shifts via linear transformations:
              $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/D}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/D}}\right)$$
            """
        )
