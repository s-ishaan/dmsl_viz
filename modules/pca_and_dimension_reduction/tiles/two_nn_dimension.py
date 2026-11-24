"""
Tile 6: Intrinsic Dimension Estimation (The TWO-NN Algorithm)
Focus: Slides 342-343 - Facco, d’Errico, Rodriguez, and Laio (2017) Scientific Reports.
Direct interactive implementation of the 4-step minimal neighborhood algorithm:
1. Distance ratio: μ_i = r_2 / r_1
2. Empirical CDF: F_emp(μ_i) = i / N
3. Coordinate plane fit: (log μ_i, -log(1 - F_emp(μ_i))) with line passing through origin
4. Slope = estimated intrinsic dimension p!
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from utils.datasets import get_swiss_roll_3d
from utils.plotting import (
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ TWO-NN Estimator Knobs")

            manifold_scenario = st.selectbox(
                "Manifold Test Geometry",
                [
                    "2D Swiss Roll in 3D Space (True d = 2)",
                    "1D Spiral Helix in 3D Space (True d = 1)",
                    "2D Non-linear Sphere Shell in 6D (True d = 2)",
                    "4D Hyper-manifold in 10D Space (True d = 4)",
                ],
                index=0,
                key="tnn_scenario",
                help="Test the TWO-NN algorithm on manifolds with known ground-truth intrinsic dimensions.",
            )

            n_samples = st.slider(
                "Sample Count N",
                min_value=150,
                max_value=700,
                value=350,
                step=50,
                key="tnn_n",
                help="Number of points sampled from the manifold.",
            )

            noise_lvl = st.slider(
                "Ambient Noise σ",
                min_value=0.0,
                max_value=0.15,
                value=0.02,
                step=0.01,
                key="tnn_noise",
            )

            discard_fraction = st.slider(
                "Discard Boundary Tail (%)",
                min_value=0,
                max_value=25,
                value=10,
                step=5,
                key="tnn_discard",
                help="Facco et al. (2017) recommend discarding the top 10% highest ratios to avoid finite sample edge distortions.",
            )

            with st.expander("⚙️ Advanced Settings", expanded=False):
                seed_val = st.number_input("Random Seed", value=42, step=1, key="tnn_seed")

    # Generate manifold data based on scenario
    rng = np.random.RandomState(seed_val)

    if "2D Swiss Roll" in manifold_scenario:
        true_d = 2
        ambient_D = 3
        X, t_color = get_swiss_roll_3d(n_samples=n_samples, noise=noise_lvl, random_state=seed_val)
    elif "1D Spiral" in manifold_scenario:
        true_d = 1
        ambient_D = 3
        t = np.linspace(0, 4.0 * np.pi, n_samples)
        x1 = np.cos(t) + rng.randn(n_samples) * noise_lvl
        x2 = np.sin(t) + rng.randn(n_samples) * noise_lvl
        x3 = t * 0.4 + rng.randn(n_samples) * noise_lvl
        X = np.column_stack([x1, x2, x3])
        t_color = t
    elif "Sphere Shell" in manifold_scenario:
        true_d = 2
        ambient_D = 6
        theta = rng.uniform(0, 2.0 * np.pi, n_samples)
        phi = np.arccos(rng.uniform(-1.0, 1.0, n_samples))
        x1 = np.sin(phi) * np.cos(theta)
        x2 = np.sin(phi) * np.sin(theta)
        x3 = np.cos(phi)
        extra_zeros = rng.randn(n_samples, 3) * noise_lvl
        X = np.column_stack([x1, x2, x3, extra_zeros])
        t_color = phi
    else:
        true_d = 4
        ambient_D = 10
        # 4D ball embedded in 10D
        pts_4d = rng.randn(n_samples, 4)
        pts_4d = pts_4d / np.linalg.norm(pts_4d, axis=1, keepdims=True)
        radii = (rng.uniform(0, 1, n_samples) ** (1.0 / 4.0)).reshape(-1, 1)
        pts_4d = pts_4d * radii
        extra_noise = rng.randn(n_samples, 6) * noise_lvl
        X = np.column_stack([pts_4d, extra_noise])
        t_color = radii.ravel()

    # TWO-NN Algorithm Implementation (Slide 343)
    # Step 1: For each point, find r_1 and r_2 (nearest and 2nd nearest neighbors)
    # n_neighbors=3 because neighbor 0 is the point itself!
    nbrs = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    distances, _ = nbrs.kneighbors(X)

    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Prevent divide-by-zero
    r1 = np.maximum(r1, 1e-10)
    mu = r2 / r1  # mu_i >= 1.0

    # Step 2: Sort mu in ascending order and compute empirical CDF F_emp(mu) = i / N
    sort_perm = np.argsort(mu)
    mu_sorted = mu[sort_perm]
    N = len(mu_sorted)
    F_emp = np.arange(1, N + 1) / float(N)

    # Discard boundary tail if requested (Facco et al. 2017)
    valid_mask = F_emp <= (1.0 - discard_fraction / 100.0)
    mu_fit = mu_sorted[valid_mask]
    F_fit = F_emp[valid_mask]

    # Step 3: Coordinates x_coord = log(mu), y_coord = -log(1 - F_emp(mu))
    x_coords = np.log(mu_fit)
    y_coords = -np.log(1.0 - F_fit)

    # Step 4: Fit straight line through the origin y = d * x
    # Least squares with zero intercept: d = sum(x * y) / sum(x^2)
    estimated_slope = float(np.sum(x_coords * y_coords) / np.sum(x_coords ** 2))
    rounded_p = int(np.round(estimated_slope))

    # Compute R^2 of fit through origin
    y_pred = estimated_slope * x_coords
    ss_res = np.sum((y_coords - y_pred) ** 2)
    ss_tot = np.sum(y_coords ** 2)
    r2_fit = float(1.0 - (ss_res / max(1e-10, ss_tot)))

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estimated Intrinsic d", f"{estimated_slope:.2f} → p = {rounded_p}")
        m2.metric("True Intrinsic d", f"d = {true_d}", f"{'Perfect Match! 🎉' if rounded_p == true_d else 'Approximation'}")
        m3.metric("Ambient Dimension D", f"D = {ambient_D}")
        m4.metric("Regression Fit R²", f"{r2_fit:.4f}", help="Linearity verifies the Pareto law of distance ratios")

        tab_regression, tab_ratio_dist, tab_manifold = st.tabs([
            "📈 TWO-NN Regression Plane",
            "🎲 Distance Ratio Distribution (μ = r₂ / r₁)",
            "🌐 Ambient vs Intrinsic Manifold",
        ])

        with tab_regression:
            st.markdown(r"##### 📈 TWO-NN Coordinate Plane: $-\log(1 - F_{\text{emp}}(\mu))$ vs $\log(\mu)$")

            fig_reg = go.Figure()

            # Empirical points
            fig_reg.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    marker=dict(color="#38BDF8", size=5, opacity=0.8),
                    name="Empirical Points (log μ, -log(1 - F))",
                    hovertemplate="log μ: %{x:.3f}<br>-log(1-F): %{y:.3f}<extra></extra>",
                )
            )

            # Fitted regression line through origin: y = d * x
            x_line = np.linspace(0.0, np.max(x_coords) * 1.05, 100)
            y_line = estimated_slope * x_line

            fig_reg.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"Fitted Slope: d̂ = {estimated_slope:.2f} (Round to p = {rounded_p})",
                    line=dict(color=COLOR_HYPERPLANE, width=3.2),
                    hovertemplate="x: %{x:.2f}<br>Fitted y: %{y:.2f}<extra></extra>",
                )
            )

            # Ground truth slope reference
            fig_reg.add_trace(
                go.Scatter(
                    x=x_line,
                    y=true_d * x_line,
                    mode="lines",
                    name=f"True Slope: d_true = {true_d}",
                    line=dict(color="#F59E0B", width=2.0, dash="dash"),
                    hovertemplate="True y: %{y:.2f}<extra></extra>",
                )
            )

            fig_reg.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="log(μ) = log(r₂ / r₁)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="-log(1 - F_emp(μ))", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                margin=dict(l=40, r=20, t=15, b=85),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#CBD5E1", size=9.5),
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    bordercolor="#334155",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(fig_reg, width="stretch", key="tnn_fig_reg")

        with tab_ratio_dist:
            st.markdown(r"##### 🎲 Empirical Density of Neighbor Ratios $\mu = r_2 / r_1 \ge 1$ vs Pareto Distribution $f(\mu) = d \mu^{-(d+1)}$")

            fig_dist = go.Figure()

            # Histogram of empirical ratios
            fig_dist.add_trace(
                go.Histogram(
                    x=mu[mu <= 4.0],
                    nbinsx=40,
                    histnorm="probability density",
                    name="Empirical Density",
                    marker=dict(color="#38BDF8", opacity=0.7, line=dict(color="#0F172A", width=0.8)),
                )
            )

            # Theoretical Pareto density f(mu) = d * mu^(-d-1) for mu >= 1
            mu_dens_axis = np.linspace(1.0, 4.0, 200)
            f_theo = estimated_slope * (mu_dens_axis ** (-estimated_slope - 1.0))

            fig_dist.add_trace(
                go.Scatter(
                    x=mu_dens_axis,
                    y=f_theo,
                    mode="lines",
                    name=f"Theoretical Pareto Density (d = {estimated_slope:.2f})",
                    line=dict(color=COLOR_HYPERPLANE, width=3.0),
                    hovertemplate="μ: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>",
                )
            )

            fig_dist.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Distance Ratio μ = r₂ / r₁", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    range=[1.0, 4.0],
                ),
                yaxis=dict(
                    title=dict(text="Probability Density", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                margin=dict(l=40, r=20, t=15, b=85),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#CBD5E1", size=9.5),
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    bordercolor="#334155",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(fig_dist, width="stretch", key="tnn_fig_dist")

        with tab_manifold:
            st.markdown(f"##### 🌐 3D Projection of Active Manifold ({manifold_scenario})")

            fig_mani = go.Figure()

            fig_mani.add_trace(
                go.Scatter3d(
                    x=X[:, 0],
                    y=X[:, 1],
                    z=X[:, 2],
                    mode="markers",
                    marker=dict(size=4.5, color=t_color, colorscale="Viridis", opacity=0.85),
                    hovertemplate="3D Coord: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
                )
            )

            fig_mani.update_layout(
                height=430,
                paper_bgcolor=BG_PAPER,
                margin=dict(l=20, r=20, t=15, b=45),
                scene=dict(
                    xaxis=dict(title="x₁", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                    yaxis=dict(title="x₂", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                    zaxis=dict(title="x₃", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                ),
            )

            st.plotly_chart(fig_mani, width="stretch", key="tnn_fig_mani")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: The TWO-NN Intrinsic Dimension Algorithm", expanded=False):
        st.markdown(
            r"""
            **Facco, d’Errico, Rodriguez, and Laio (2017) Algorithm**:
            - **Foundation**: Under the assumption of local Poisson process homogeneity around a point $x_i$, the volume of a $d$-dimensional sphere of radius $r$ is $V(r) = \omega_d r^d$.
            - The ratio $\mu_i = \frac{r_2(x_i)}{r_1(x_i)}$ between the distance to the second-nearest neighbor $r_2$ and the first-nearest neighbor $r_1$ follows a **Pareto distribution**:
              $$P(\mu \le x) = 1 - x^{-d}, \quad x \ge 1$$
            - Taking the empirical cumulative distribution $F_{\text{emp}}(\mu) \approx 1 - \mu^{-d}$, we rearrange:
              $$1 - F_{\text{emp}}(\mu) = \mu^{-d} \implies -\log(1 - F_{\text{emp}}(\mu)) = d \cdot \log(\mu)$$
            - This is a simple linear equation $y = d \cdot x$ passing through the origin $(0, 0)$!
            - **Four Step Summary**:
              1. Calculate $\mu_i = \frac{r_2}{r_1}$ for each point.
              2. Sort $\mu_{(i)}$ in ascending order and compute $F_{\text{emp}}(\mu_{(i)}) = \frac{i}{N}$.
              3. Plot coordinates $\{(\log \mu_i, -\log(1 - F_{\text{emp}}(\mu_i)))\}$ on a 2D plane.
              4. Fit a straight line through the origin. **The slope of this line rounded to the nearest integer is the intrinsic dimension $p$!**
            """
        )
