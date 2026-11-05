"""
DMSL Interactive Study Sandbox
Main Streamlit Application Entrypoint
Senior Data Presentation Architecture: Clean Dark Aesthetic with Native Contrast
"""

import importlib
import streamlit as st
import utils.plotting
import utils.datasets
import modules.registry
import modules.svm_linear.page
import modules.svm_linear.tiles.hyperplane_algebra
import modules.svm_linear.tiles.perceptron_vs_ls_vs_svm
import modules.svm_linear.tiles.hard_margin_osh
import modules.svm_linear.tiles.soft_margin_slack
import modules.svm_linear.tiles.c_regularization
import modules.svm_linear.tiles.kernel_mapping
import modules.svm_kernels.page
import modules.svm_kernels.tiles.poly_kernel
import modules.svm_kernels.tiles.rbf_kernel
import modules.svm_kernels.tiles.kernel_regression
import modules.svm_kernels.tiles.gram_matrix_mercer
import modules.svm_kernels.tiles.loss_functions_arena
import modules.svm_kernels.tiles.svm_vs_lr_benchmark
import modules.trees_and_boosting.page
import modules.trees_and_boosting.tiles.recursive_partitioning
import modules.trees_and_boosting.tiles.node_impurity
import modules.trees_and_boosting.tiles.cost_complexity_pruning
import modules.trees_and_boosting.tiles.tree_instability
import modules.trees_and_boosting.tiles.adaboost_dynamics
import modules.trees_and_boosting.tiles.adaboost_benchmark

# Hot reload modules on every run to eliminate stale bytecode
importlib.reload(utils.plotting)
importlib.reload(utils.datasets)
importlib.reload(modules.svm_linear.tiles.hyperplane_algebra)
importlib.reload(modules.svm_linear.tiles.perceptron_vs_ls_vs_svm)
importlib.reload(modules.svm_linear.tiles.hard_margin_osh)
importlib.reload(modules.svm_linear.tiles.soft_margin_slack)
importlib.reload(modules.svm_linear.tiles.c_regularization)
importlib.reload(modules.svm_linear.tiles.kernel_mapping)
importlib.reload(modules.svm_linear.page)
importlib.reload(modules.svm_kernels.tiles.poly_kernel)
importlib.reload(modules.svm_kernels.tiles.rbf_kernel)
importlib.reload(modules.svm_kernels.tiles.kernel_regression)
importlib.reload(modules.svm_kernels.tiles.gram_matrix_mercer)
importlib.reload(modules.svm_kernels.tiles.loss_functions_arena)
importlib.reload(modules.svm_kernels.tiles.svm_vs_lr_benchmark)
importlib.reload(modules.svm_kernels.page)
importlib.reload(modules.trees_and_boosting.tiles.recursive_partitioning)
importlib.reload(modules.trees_and_boosting.tiles.node_impurity)
importlib.reload(modules.trees_and_boosting.tiles.cost_complexity_pruning)
importlib.reload(modules.trees_and_boosting.tiles.tree_instability)
importlib.reload(modules.trees_and_boosting.tiles.adaboost_dynamics)
importlib.reload(modules.trees_and_boosting.tiles.adaboost_benchmark)
importlib.reload(modules.trees_and_boosting.page)
importlib.reload(modules.registry)
from modules.registry import get_available_modules

# Streamlit Page Configuration
st.set_page_config(
    page_title="DMSL Study Sandbox",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished dark analytics aesthetic
st.markdown(
    """
    <style>
    /* Base typography and layout refinement */
    .stApp {
        background-color: #0B1120;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }
    
    /* Top title & subheadings */
    h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #F8FAFC !important;
        margin-bottom: 2px !important;
    }
    
    /* Tabs bar: sleek pill style */
    div[data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0F172A;
        padding: 5px;
        border-radius: 10px;
        border: 1px solid #1E293B;
        margin-bottom: 12px;
    }
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 6px 14px !important;
        border-radius: 6px !important;
        color: #94A3B8 !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.15s ease-in-out;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        background-color: #1E293B !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    }
    
    /* Bordered containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px !important;
        border-color: #334155 !important;
        background-color: rgba(30, 41, 59, 0.35) !important;
        margin-bottom: 8px !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E293B 0%, #172033 100%);
        border: 1px solid #334155;
        padding: 8px 12px;
        border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 500;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 700;
        font-size: 1.20rem !important;
    }
    div[data-testid="stMetricDelta"] {
        font-weight: 600;
        font-size: 0.78rem !important;
    }
    
    /* Section dividers */
    hr {
        border-color: #1E293B;
        margin: 18px 0;
    }
    
    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        background-color: rgba(15, 23, 42, 0.4) !important;
    }

    /* Headings inside cards */
    h4 {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
        font-size: 1.0rem !important;
    }
    h5 {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        margin-bottom: 6px !important;
        font-size: 0.90rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    modules = get_available_modules()

    with st.sidebar:
        st.markdown("### 📐 DMSL Sandbox")
        st.caption("Data Mining & Statistical Learning (ISYE-7406)")

        st.markdown("---")
        st.markdown("**Course Modules**")
        module_keys = list(modules.keys())
        module_titles = [modules[k]["title"] for k in module_keys]

        selected_title = st.selectbox(
            "Select Course Module",
            module_titles,
            index=0,
            help="Switch between parsed lecture deck visualizers",
            label_visibility="collapsed",
        )

        selected_key = module_keys[module_titles.index(selected_title)]
        selected_mod = modules[selected_key]

        st.markdown("---")
        st.markdown(f"**Course:** `{selected_mod['course']}`")
        st.markdown(f"**Deck Reference:** `{selected_mod['slides_ref']}`")
        st.markdown(f"**Active Module ID:** `{selected_key}`")

        st.markdown("---")
        st.info(
            "💡 **Extensible Workspace Architecture**\n\n"
            "Drop new slide decks into this workspace. "
            "Antigravity will parse them and add a new interactive page to this sidebar."
        )

    # Render selected module page
    selected_mod["render_fn"]()


if __name__ == "__main__":
    main()
