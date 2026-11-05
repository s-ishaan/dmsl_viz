#!/usr/bin/env bash
# Quick launcher for DMSL Interactive Study Sandbox
cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run app.py
