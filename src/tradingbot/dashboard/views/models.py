"""Models page: saved LightGBM model catalog from models/*_meta.json."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render() -> None:
    from tradingbot.ml.trainer import LGBMTrainer

    st.subheader("Model Catalog")
    model_dir = st.sidebar.text_input("Model directory", value="models")

    entries = LGBMTrainer.load_catalog(Path(model_dir))
    if not entries:
        st.info("No saved models found. Train them first: `tradingbot ml-train-all`")
        return

    import pandas as pd

    df = pd.DataFrame(entries)
    # Operator-relevant columns first; whatever else the meta grows stays behind.
    front = [
        c
        for c in (
            "symbol",
            "timeframe",
            "holdout_auc",
            "entry_threshold",
            "exit_threshold",
            "has_calibrator",
            "n_features",
            "trained_at",
        )
        if c in df.columns
    ]
    df = df[front + [c for c in df.columns if c not in front]]
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].round(3)

    st.caption(f"{len(entries)} models in `{model_dir}/`")
    st.dataframe(df, use_container_width=True, hide_index=True)
