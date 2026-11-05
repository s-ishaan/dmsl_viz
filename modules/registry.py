"""
Module Registry for DMSL Interactive Visual Sandbox
Dynamically manages and discovers lecture deck visualizers.
"""

from typing import Dict, Any, Callable
from modules.svm_linear import render as render_svm_linear
from modules.svm_kernels.page import render as render_svm_kernels
from modules.trees_and_boosting.page import render as render_trees_and_boosting


# Registered lecture modules
# Future decks dropped into workspace will be added here
REGISTRY: Dict[str, Dict[str, Any]] = {
    "trees_and_boosting": {
        "title": "Decision Trees, Pruning & AdaBoost Ensembles",
        "topic": "Recursive Partitioning, Pruning & Committee Methods",
        "render_fn": render_trees_and_boosting,
    },
    "svm_kernels": {
        "title": "Support Vector Machines (Part 2: Kernels & RKHS)",
        "topic": "Non-linear Boundaries, Mercer RKHS & Loss Landscapes",
        "render_fn": render_svm_kernels,
    },
    "svm_linear": {
        "title": "Support Vector Machines (Part 1: Linear & OSH)",
        "topic": "Maximal Margin Classifiers & Dual Optimization",
        "render_fn": render_svm_linear,
    },
}


def get_available_modules() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of available lecture deck modules."""
    return REGISTRY
