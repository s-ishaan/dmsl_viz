"""
Tile 2: Self-Attention Mechanics & Contextual Disambiguation
Focus: Slide 350-351 - The limitation of static embeddings, coreference resolution,
and Query-Key-Value attention as dynamic contextual embedding.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.plotting import (
    COLOR_HYPERPLANE,
    COLOR_CLASS_POS,
    COLOR_CLASS_NEG,
    BG_PAPER,
    BG_PLOT,
)


def compute_attention_distribution(
    tokens: list,
    query_token: str,
    context_type: str,
    temperature: float = 1.0,
    context_strength: float = 1.0,
) -> tuple[np.ndarray, list]:
    """
    Simulates learned self-attention scores for coreference disambiguation.
    Recreates Slide 350 attention distribution:
    - In "tired" context: 'it' attends strongly to 'animal' (~42%).
    - In "wide" context: 'it' attends strongly to 'street' (~38%).
    """
    n = len(tokens)
    raw_scores = np.zeros(n)

    q_idx = -1
    for idx, tok in enumerate(tokens):
        if tok.lower().strip(".,!?") == query_token.lower().strip(".,!?"):
            q_idx = idx
            break

    if q_idx == -1:
        q_idx = 0

    # Base attention biases for grammatical proximity and syntactic relations
    for i, tok in enumerate(tokens):
        clean_tok = tok.lower().strip(".,!?")
        # Self-attention self-loop
        if i == q_idx:
            raw_scores[i] = 1.2
        # Punctuation gets minimal attention
        elif clean_tok in [".", ",", "!", "?"]:
            raw_scores[i] = 0.2
        # Determiners and function words
        elif clean_tok in ["the", "because", "was"]:
            raw_scores[i] = 0.8
        # Verbs
        elif clean_tok in ["didn't", "cross"]:
            raw_scores[i] = 1.4
        # Modifiers
        elif clean_tok in ["too", "tired", "wide"]:
            raw_scores[i] = 1.6
        else:
            raw_scores[i] = 1.0

    # Coreference resolution logic
    if query_token.lower() == "it":
        for i, tok in enumerate(tokens):
            clean_tok = tok.lower().strip(".,!?")
            if context_type == "tired":
                # 'tired' predicates an animate agent -> 'animal'
                if clean_tok == "animal":
                    raw_scores[i] += 3.2 * context_strength
                elif clean_tok == "street":
                    raw_scores[i] += 0.8
                elif clean_tok == "tired":
                    raw_scores[i] += 2.0 * context_strength
            elif context_type == "wide":
                # 'wide' predicates a spatial noun -> 'street'
                if clean_tok == "street":
                    raw_scores[i] += 3.2 * context_strength
                elif clean_tok == "animal":
                    raw_scores[i] += 0.8
                elif clean_tok == "wide":
                    raw_scores[i] += 2.0 * context_strength
    else:
        # Non-'it' query: distribute attention based on distance
        for i in range(n):
            dist = abs(i - q_idx)
            raw_scores[i] += max(0.0, 2.5 - 0.4 * dist)

    # Softmax with temperature tau
    scaled_scores = raw_scores / max(0.05, temperature)
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    attention_weights = exp_scores / np.sum(exp_scores)

    return attention_weights, q_idx


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Attention & Context Knobs")

            sentence_scenario = st.radio(
                "Linguistic Context Scenario",
                [
                    "A: '...because it was too tired' (Agent focus)",
                    "B: '...because it was too wide' (Spatial focus)",
                ],
                index=0,
                key="sad_scenario",
                help="Demonstrates how adjective context radically alters coreference resolution for the pronoun 'it'.",
            )

            context_type = "tired" if "tired" in sentence_scenario else "wide"
            if context_type == "tired":
                default_tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired", "."]
            else:
                default_tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "wide", "."]

            query_token = st.selectbox(
                "Query Token (q)",
                default_tokens,
                index=default_tokens.index("it"),
                key="sad_query_token",
                help="The token emitting the query vector q to attend over the rest of the sentence.",
            )

            temperature = st.slider(
                "Softmax Temperature τ",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                key="sad_temp",
                help="Scaling factor in softmax. Low τ sharpens attention into argmax; high τ diffuses attention uniformly.",
            )

            context_strength = st.slider(
                "Context Bias Strength",
                min_value=0.2,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="sad_strength",
                help="Scales the semantic affinity between the adjective predicate and candidate nouns.",
            )

    # Compute attention distribution
    weights, q_idx = compute_attention_distribution(
        default_tokens,
        query_token,
        context_type,
        temperature,
        context_strength,
    )

    # Find dominant target token (excluding query itself)
    candidate_indices = [i for i in range(len(default_tokens)) if i != q_idx and default_tokens[i] not in [".", ","]]
    if candidate_indices:
        top_cand_idx = candidate_indices[int(np.argmax(weights[candidate_indices]))]
        top_cand_word = default_tokens[top_cand_idx]
        top_cand_pct = weights[top_cand_idx] * 100
    else:
        top_cand_word = "None"
        top_cand_pct = 0.0

    animal_idx = default_tokens.index("animal")
    street_idx = default_tokens.index("street")
    animal_pct = weights[animal_idx] * 100
    street_pct = weights[street_idx] * 100

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Query Token", f"'{query_token}'")
        m2.metric("Primary Attention", f"'{top_cand_word}'", f"{top_cand_pct:.1f}% weight")
        m3.metric("'animal' Attention", f"{animal_pct:.1f}%")
        m4.metric("'street' Attention", f"{street_pct:.1f}%")

        tab_tree, tab_bars, tab_embedding_shift = st.tabs([
            "🌳 Attention Flow Tree Diagram",
            "📊 Attention Distribution Profile",
            "📍 Contextual Embedding Shift",
        ])

        with tab_tree:
            active_color = COLOR_CLASS_POS if context_type == "tired" else COLOR_CLASS_NEG
            st.markdown(f"##### 🌳 Self-Attention Flow: Query `'{query_token}'` $\\to$ Contextual Weights $\\alpha(q, k_i)$")

            fig_tree = go.Figure()

            # Coordinates for query node (top center) and key nodes (bottom row)
            n_toks = len(default_tokens)
            x_keys = np.linspace(0, n_toks - 1, n_toks)
            y_keys = np.zeros(n_toks)

            x_q = float(q_idx)
            y_q = 2.2

            # Draw radiating attention lines from query to keys
            for i in range(n_toks):
                w = weights[i]
                line_width = max(0.8, w * 32.0)
                opacity = min(1.0, max(0.12, w * 2.5))

                if default_tokens[i] in ["animal", "street"] and w > 0.15:
                    branch_color = active_color
                else:
                    branch_color = "#64748B"

                fig_tree.add_trace(
                    go.Scatter(
                        x=[x_q, x_keys[i]],
                        y=[y_q, y_keys[i]],
                        mode="lines",
                        line=dict(width=line_width, color=branch_color),
                        opacity=opacity,
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

            # Draw bottom key nodes
            node_colors = []
            for i, tok in enumerate(default_tokens):
                if tok == "animal" and context_type == "tired":
                    node_colors.append(COLOR_CLASS_POS)
                elif tok == "street" and context_type == "wide":
                    node_colors.append(COLOR_CLASS_NEG)
                elif i == q_idx:
                    node_colors.append("#38BDF8")
                else:
                    node_colors.append("#334155")

            fig_tree.add_trace(
                go.Scatter(
                    x=x_keys,
                    y=y_keys,
                    mode="markers+text",
                    marker=dict(size=24, color=node_colors, line=dict(color="#0F172A", width=1.5)),
                    text=[f"<b>{tok}</b><br>{weights[i]*100:.1f}%" for i, tok in enumerate(default_tokens)],
                    textposition="bottom center",
                    textfont=dict(color="#F8FAFC", size=9.5),
                    name="Context Tokens (Keys k_i)",
                    hoverinfo="text",
                )
            )

            # Draw top Query node
            fig_tree.add_trace(
                go.Scatter(
                    x=[x_q],
                    y=[y_q],
                    mode="markers+text",
                    marker=dict(size=28, color="#38BDF8", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
                    text=[f"<b>Query (q)</b><br>'{query_token}'"],
                    textposition="top center",
                    textfont=dict(color="#38BDF8", size=11),
                    name="Query Token (q)",
                    hoverinfo="text",
                )
            )

            fig_tree.update_layout(
                height=450,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, n_toks - 0.2]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, 3.0]),
                margin=dict(l=10, r=10, t=25, b=25),
                legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(color="#CBD5E1", size=9)),
            )

            st.plotly_chart(fig_tree, width="stretch", key="sad_fig_tree")

        with tab_bars:
            st.markdown("##### 📊 Discrete Attention Probability Distribution $\\alpha(q, k_i) = \\text{softmax}(q k_i^T / \\tau)$")

            fig_bars = go.Figure()

            bar_colors = []
            for i, tok in enumerate(default_tokens):
                if tok == "animal":
                    bar_colors.append(COLOR_CLASS_POS)
                elif tok == "street":
                    bar_colors.append(COLOR_CLASS_NEG)
                elif i == q_idx:
                    bar_colors.append("#38BDF8")
                else:
                    bar_colors.append("#64748B")

            fig_bars.add_trace(
                go.Bar(
                    x=default_tokens,
                    y=weights * 100,
                    marker=dict(color=bar_colors, line=dict(color="#0F172A", width=1.0)),
                    text=[f"{w*100:.1f}%" for w in weights],
                    textposition="outside",
                    textfont=dict(color="#F8FAFC", size=9.5),
                    name="Attention Weight α",
                )
            )

            fig_bars.update_layout(
                height=430,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(title="Sentence Tokens", gridcolor="#1E293B", tickfont=dict(color="#CBD5E1", size=10)),
                yaxis=dict(title="Attention Weight (%)", gridcolor="#1E293B", tickfont=dict(color="#94A3B8"), range=[0, max(50.0, float(np.max(weights) * 115))]),
                margin=dict(l=40, r=20, t=20, b=45),
            )

            st.plotly_chart(fig_bars, width="stretch", key="sad_fig_bars")

        with tab_embedding_shift:
            st.markdown("##### 📍 Contextual Embedding Shift: From Static Word Vector to Dynamic Latent Position")

            # 2D coordinates illustrating static vs contextualized embeddings
            static_locs = {
                "animal": np.array([-2.2, 1.8]),
                "street": np.array([2.5, -1.5]),
                "tired": np.array([-3.0, 3.2]),
                "wide": np.array([3.4, -2.8]),
                "it (static)": np.array([0.0, 0.0]),
            }

            # In "tired" context, 'it' moves towards 'animal'
            if context_type == "tired":
                it_contextual = np.array([-1.7, 1.4])
            else:
                it_contextual = np.array([1.9, -1.1])

            fig_shift = go.Figure()

            # Plot static background words
            for w, coord in static_locs.items():
                col = "#F43F5E" if "street" in w or "wide" in w else ("#38BDF8" if "animal" in w or "tired" in w else "#94A3B8")
                fig_shift.add_trace(
                    go.Scatter(
                        x=[coord[0]],
                        y=[coord[1]],
                        mode="markers+text",
                        marker=dict(size=12, color=col),
                        text=[w],
                        textposition="top center",
                        textfont=dict(color="#CBD5E1", size=10),
                        name=w,
                        showlegend=False,
                    )
                )

            # Draw vector path from static 'it' to contextualized 'it'
            fig_shift.add_trace(
                go.Scatter(
                    x=[static_locs["it (static)"][0], it_contextual[0]],
                    y=[static_locs["it (static)"][1], it_contextual[1]],
                    mode="lines+markers",
                    line=dict(color=COLOR_HYPERPLANE, width=3.5),
                    marker=dict(size=14, color=COLOR_HYPERPLANE, symbol=["circle", "diamond"]),
                    text=["Static Vector x", f"Contextualized z ('{context_type}')"],
                    textposition="bottom right",
                    name="Self-Attention Shift",
                )
            )

            fig_shift.update_layout(
                height=430,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(title="Semantic Latent Axis 1", gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                yaxis=dict(title="Semantic Latent Axis 2", gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                margin=dict(l=40, r=20, t=20, b=45),
            )

            st.plotly_chart(fig_shift, width="stretch", key="sad_fig_shift")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: The Self-Attention Mechanism", expanded=False):
        st.markdown(
            r"""
            **The Core Limitation of Static Embeddings**:
            - Traditional static word representations (e.g. Word2Vec, GloVe) assign a single fixed vector $\mathbf{x} \in \mathbb{R}^D$ to each token.
            - In the sentences:
              1. *"The animal didn't cross the street because **it** was too tired."*
              2. *"The animal didn't cross the street because **it** was too wide."*
            - The token `"it"` has an identical static vector in both sentences, completely blind to whether it refers to the animate entity (*animal*) or the physical path (*street*).

            **The Query-Key-Value Paradigm**:
            - The self-attention mechanism maps each token's embedding vector $\mathbf{x}_i \in \mathbb{R}^D$ into three distinct learned functional representations:
              $$\mathbf{q}_i = W^Q \mathbf{x}_i \in \mathbb{R}^{d_k} \quad (\text{Query: What am I looking for?})$$
              $$\mathbf{k}_i = W^K \mathbf{x}_i \in \mathbb{R}^{d_k} \quad (\text{Key: What information do I offer?})$$
              $$\mathbf{v}_i = W^V \mathbf{x}_i \in \mathbb{R}^{d_v} \quad (\text{Value: What content do I transmit?})$$

            **Contextualized Output Representation**:
            - For query token $\mathbf{q}$, attention weights $\alpha(q, k_i)$ compute similarity against all candidate keys $\mathbf{k}_i$ across the dictionary $\mathcal{D}$:
              $$\text{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$$
            - The softmax operation guarantees non-negativity and probability normalization ($\sum_i \alpha(\mathbf{q}, \mathbf{k}_i) = 1$):
              $$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp\left(\frac{\mathbf{q}^T \mathbf{k}_i}{\tau}\right)}{\sum_{j=1}^m \exp\left(\frac{\mathbf{q}^T \mathbf{k}_j}{\tau}\right)}$$
            - In this sense, **self-attention generates dynamic contextual embeddings**, moving the latent position of the token toward the specific context words it resolves to.
            """
        )
