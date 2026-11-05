"""
DMSL Visualization Utilities - Plotting
High-performance, elegant Plotly helpers for decision boundaries, margins, normal vectors, and support vectors.
Legends are placed cleanly at the bottom of figures to guarantee zero collision with titles or axes.
"""

from typing import Optional, List, Dict, Tuple
import numpy as np
import plotly.graph_objects as go


# Professional Modern Dark Analytics Palette
COLOR_CLASS_NEG = "#F43F5E"      # Crisp Rose for Class -1
COLOR_CLASS_POS = "#38BDF8"      # Bright Sky Blue for Class +1
COLOR_HYPERPLANE = "#10B981"     # Emerald Green
COLOR_MARGIN = "#F59E0B"         # Amber
COLOR_SV_RING = "#EAB308"        # Gold
COLOR_SLACK_LINE = "#EC4899"     # Fuchsia / Pink
COLOR_PROJECTION = "#A855F7"     # Purple
BG_PLOT = "#0F172A"              # Slate 900
BG_PAPER = "#1E293B"             # Slate 800
GRID_COLOR = "#1E293B"           # Subtle slate for inner grids
AXIS_LINE_COLOR = "#334155"      # Crisp zero/axis lines


def get_base_2d_figure(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title: str = "",
    height: int = 440,
    show_axes_labels: bool = True,
    show_legend: bool = True,
) -> go.Figure:
    """
    Create a standardized Plotly figure with clean mathematical coordinates.
    Titles are placed outside in Streamlit HTML by default; internal Plotly titles are optional.
    Horizontal legend is cleanly placed at the bottom to guarantee zero overlap.
    """
    fig = go.Figure()
    
    top_margin = 40 if title else 15
    bottom_margin = 85 if show_legend else 35

    layout_kwargs = dict(
        margin=dict(l=40, r=20, t=top_margin, b=bottom_margin),
        height=height,
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        xaxis=dict(
            range=x_range,
            zeroline=True,
            zerolinecolor=AXIS_LINE_COLOR,
            zerolinewidth=1.2,
            gridcolor=GRID_COLOR,
            gridwidth=1,
            tickfont=dict(color="#94A3B8", size=9.5),
            title=dict(text="Feature x₁", font=dict(color="#CBD5E1", size=11)) if show_axes_labels else None,
        ),
        yaxis=dict(
            range=y_range,
            zeroline=True,
            zerolinecolor=AXIS_LINE_COLOR,
            zerolinewidth=1.2,
            gridcolor=GRID_COLOR,
            gridwidth=1,
            tickfont=dict(color="#94A3B8", size=9.5),
            title=dict(text="Feature x₂", font=dict(color="#CBD5E1", size=11)) if show_axes_labels else None,
            scaleanchor="x",
            scaleratio=1,
        ),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#CBD5E1", size=9.5, family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
        ) if show_legend else None,
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="#475569",
            font=dict(color="#F8FAFC", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=11),
        ),
        hovermode="closest",
    )

    if title:
        layout_kwargs["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(size=13, color="#F8FAFC", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
        )

    fig.update_layout(**layout_kwargs)
    return fig


def add_scatter_points(
    fig: go.Figure,
    X: np.ndarray,
    y: np.ndarray,
    marker_size: int = 9,
    name_prefix: str = "",
    show_legend: bool = True,
) -> go.Figure:
    """Add class -1 and class +1 scatter points to figure."""
    mask_neg = (y == -1)
    mask_pos = (y == 1)

    # Class -1
    if np.any(mask_neg):
        fig.add_trace(
            go.Scatter(
                x=X[mask_neg, 0],
                y=X[mask_neg, 1],
                mode="markers",
                name=f"{name_prefix}Class -1",
                marker=dict(
                    color=COLOR_CLASS_NEG,
                    size=marker_size,
                    line=dict(color="#FFFFFF", width=1.2),
                    symbol="circle",
                ),
                hovertemplate="<b>Class -1</b><br>x₁: %{x:.2f}<br>x₂: %{y:.2f}<extra></extra>",
                showlegend=show_legend,
            )
        )

    # Class +1
    if np.any(mask_pos):
        fig.add_trace(
            go.Scatter(
                x=X[mask_pos, 0],
                y=X[mask_pos, 1],
                mode="markers",
                name=f"{name_prefix}Class +1",
                marker=dict(
                    color=COLOR_CLASS_POS,
                    size=marker_size,
                    line=dict(color="#FFFFFF", width=1.2),
                    symbol="circle",
                ),
                hovertemplate="<b>Class +1</b><br>x₁: %{x:.2f}<br>x₂: %{y:.2f}<extra></extra>",
                showlegend=show_legend,
            )
        )

    return fig


def add_linear_boundary_and_margins(
    fig: go.Figure,
    w: np.ndarray,
    b: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    show_margins: bool = True,
    show_margin_band: bool = True,
    boundary_color: str = COLOR_HYPERPLANE,
    boundary_name: str = "Boundary f(x)=0",
    line_width: int = 3,
) -> go.Figure:
    """
    Renders w^T x + b = 0, and canonical margins w^T x + b = +1, -1.
    """
    w0, w1 = w[0], w[1]
    xs = np.linspace(x_range[0] - 2, x_range[1] + 2, 400)

    if abs(w1) > 1e-5:
        y_decision = -(w0 * xs + b) / w1
        y_pos_margin = -(w0 * xs + b - 1.0) / w1
        y_neg_margin = -(w0 * xs + b + 1.0) / w1

        # Shaded margin corridor
        if show_margins and show_margin_band:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([xs, xs[::-1]]),
                    y=np.concatenate([y_pos_margin, y_neg_margin[::-1]]),
                    fill="toself",
                    fillcolor="rgba(245, 158, 11, 0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Margin Band (2M)",
                )
            )

        # Decision boundary
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=y_decision,
                mode="lines",
                name=boundary_name,
                line=dict(color=boundary_color, width=line_width),
                hoverinfo="skip",
            )
        )

        # Margins
        if show_margins:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=y_pos_margin,
                    mode="lines",
                    name="Margin f(x)=+1",
                    line=dict(color=COLOR_MARGIN, width=1.5, dash="dash"),
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=y_neg_margin,
                    mode="lines",
                    name="Margin f(x)=-1",
                    line=dict(color=COLOR_MARGIN, width=1.5, dash="dash"),
                    hoverinfo="skip",
                )
            )
    else:
        # Near vertical line
        denom = w0 if abs(w0) > 1e-6 else 1e-6
        x_decision = -b / denom
        x_pos_margin = -(b - 1.0) / denom
        x_neg_margin = -(b + 1.0) / denom
        ys = np.array([y_range[0] - 2, y_range[1] + 2])

        fig.add_trace(
            go.Scatter(
                x=[x_decision, x_decision],
                y=ys,
                mode="lines",
                name=boundary_name,
                line=dict(color=boundary_color, width=line_width),
                hoverinfo="skip",
            )
        )
        if show_margins:
            fig.add_trace(
                go.Scatter(
                    x=[x_pos_margin, x_pos_margin],
                    y=ys,
                    mode="lines",
                    name="Margin f(x)=+1",
                    line=dict(color=COLOR_MARGIN, width=1.5, dash="dash"),
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_neg_margin, x_neg_margin],
                    y=ys,
                    mode="lines",
                    name="Margin f(x)=-1",
                    line=dict(color=COLOR_MARGIN, width=1.5, dash="dash"),
                    hoverinfo="skip",
                )
            )

    return fig


def highlight_support_vectors(
    fig: go.Figure,
    X_sv: np.ndarray,
    y_sv: Optional[np.ndarray] = None,
    alpha_sv: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    color: str = COLOR_SV_RING,
    name: str = "Support Vectors",
) -> go.Figure:
    """Surround support vectors with high-visibility halo rings."""
    if len(X_sv) == 0:
        return fig

    hover_text = []
    for i in range(len(X_sv)):
        text = f"<b>Support Vector #{i+1}</b><br>x₁: {X_sv[i,0]:.2f}<br>x₂: {X_sv[i,1]:.2f}"
        if alpha_sv is not None:
            text += f"<br>α: {alpha_sv[i]:.4f}"
        if labels is not None and i < len(labels):
            text += f"<br>{labels[i]}"
        hover_text.append(text)

    fig.add_trace(
        go.Scatter(
            x=X_sv[:, 0],
            y=X_sv[:, 1],
            mode="markers",
            name=name,
            marker=dict(
                size=18,
                color="rgba(0,0,0,0)",
                line=dict(color=color, width=2.5),
                symbol="circle",
            ),
            hovertext=hover_text,
            hoverinfo="text",
        )
    )
    return fig


def add_normal_vector(
    fig: go.Figure,
    w: np.ndarray,
    b: float,
    scale: float = 1.5,
) -> go.Figure:
    """
    Draws the unit normal vector beta* = beta / ||beta|| rooted at the closest point to origin on the line.
    """
    norm_w = np.linalg.norm(w)
    if norm_w < 1e-6:
        return fig

    w_unit = w / norm_w
    x0 = - (b / (norm_w ** 2)) * w
    x1 = x0 + w_unit * scale

    fig.add_annotation(
        x=x1[0],
        y=x1[1],
        ax=x0[0],
        ay=x0[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2.5,
        arrowcolor="#38BDF8",
    )
    fig.add_trace(
        go.Scatter(
            x=[x1[0]],
            y=[x1[1]],
            mode="text",
            text=["<b>β* (Normal)</b>"],
            textposition="top right",
            textfont=dict(color="#38BDF8", size=11),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig


def add_contour_boundary_and_margins(
    fig: go.Figure,
    decision_func,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int = 120,
    show_margins: bool = True,
    show_surface: bool = False,
    boundary_color: str = COLOR_HYPERPLANE,
    boundary_name: str = "Boundary f(x)=0",
    line_width: int = 3,
) -> go.Figure:
    """
    Renders 2D non-linear decision boundary contour f(x) = 0 and canonical margins f(x) = +/- 1.
    decision_func should accept (N, 2) array and return continuous decision values f(x).
    """
    gx = np.linspace(x_range[0], x_range[1], resolution)
    gy = np.linspace(y_range[0], y_range[1], resolution)
    XX, YY = np.meshgrid(gx, gy)
    grid_pts = np.c_[XX.ravel(), YY.ravel()]
    
    Z = decision_func(grid_pts).reshape(XX.shape)

    # Optional background decision intensity surface
    if show_surface:
        fig.add_trace(
            go.Contour(
                x=gx,
                y=gy,
                z=Z,
                colorscale=[[0, "rgba(244, 63, 94, 0.22)"], [0.5, "rgba(15, 23, 42, 0.0)"], [1, "rgba(56, 189, 248, 0.22)"]],
                showscale=False,
                contours=dict(coloring="heatmap"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Decision Boundary f(x) = 0
    fig.add_trace(
        go.Contour(
            x=gx,
            y=gy,
            z=Z,
            contours_coloring="none",
            contours=dict(
                start=0,
                end=0,
                size=1,
                coloring="none",
            ),
            line=dict(color=boundary_color, width=line_width),
            name=boundary_name,
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # Margins f(x) = +1 and f(x) = -1
    if show_margins:
        fig.add_trace(
            go.Contour(
                x=gx,
                y=gy,
                z=Z,
                contours_coloring="none",
                contours=dict(
                    start=1,
                    end=1,
                    size=1,
                    coloring="none",
                ),
                line=dict(color=COLOR_MARGIN, width=1.8, dash="dash"),
                name="Margin f(x)=+1",
                showlegend=True,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Contour(
                x=gx,
                y=gy,
                z=Z,
                contours_coloring="none",
                contours=dict(
                    start=-1,
                    end=-1,
                    size=1,
                    coloring="none",
                ),
                line=dict(color=COLOR_MARGIN, width=1.8, dash="dash"),
                name="Margin f(x)=-1",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    return fig

