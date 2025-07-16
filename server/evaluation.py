"""
Evaluation server logic for forward and backward feature selection.
"""

import asyncio
import warnings
import pandas as pd
import numpy as np
from shiny import reactive, render, ui
from shinywidgets import render_plotly
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

from utils.algorithms import forward_selection_plsda_df, backward_elimination_plsda_df
from utils.data_processing import load_features_data


def setup_evaluation_server(input, output, session):
    """Set up evaluation server logic exactly like original app.py."""
    
    # Load data exactly like original
    df = load_features_data()
    label_col = "Class"
    
    # Extended tasks for async feature selection (exactly like original)
    @reactive.extended_task
    async def forward_res(
        numeric_data: pd.DataFrame,
        target_data: pd.Series,
        init_features: list[str] | None
    ) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            forward_selection_plsda_df,
            numeric_data,
            target_data,
            2,          # n_components
            10,         # plateau_steps
            init_features,
            "accuracy"
        )

    @reactive.extended_task
    async def backward_res(
        numeric_data: pd.DataFrame,
        target_data: pd.Series
    ) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            backward_elimination_plsda_df,
            numeric_data,
            target_data,
            2,          # min_features
            2,          # n_components
            "accuracy"
        )

    # ——————————————————————————————————————
    # Forward Feature Selection (exactly like original)
    # ——————————————————————————————————————
    @reactive.Effect
    @reactive.event(input.run_forward)
    def _start_forward():
        numeric = df.drop(columns=[label_col])
        target  = df[label_col]
        f1 = input.init_feat1().strip()
        f2 = input.init_feat2().strip()
        init = [f1, f2] if (f1 and f2) else None
        forward_res(numeric, target, init)

    @render_plotly
    def forward_perf_plot():
        if input.run_forward() < 1 or forward_res.result() is None:
            fig = go.Figure()
            fig.update_layout(
                title="Waiting to run forward selection…",
                template="ggplot2"
            )
            return fig

        df_steps = forward_res.result()
        fig = go.Figure(go.Scatter(
            x=df_steps['total_features'],
            y=df_steps['accuracy'],
            mode="lines+markers"
        ))
        fig.update_layout(
            title="Forward Selection Accuracy",
            xaxis_title="Number of Features",
            yaxis_title="Accuracy",
            template="ggplot2",
            xaxis_range=[
                df_steps['total_features'].min() - 1,
                df_steps['total_features'].max() + 1
            ],
            height=300,
        )
        return fig

    @render.data_frame
    def forward_log():
        if input.run_forward() < 1 or forward_res.result() is None:
            return pd.DataFrame(columns=[
                'step', 'total_features', 'feature_added', 'accuracy'
            ])
        return forward_res.result()

    @render.ui
    def forward_slider_ui():
        # before any run (or still running), lock at 2
        if input.run_forward() < 1 or forward_res.result() is None:
            return ui.input_slider(
                "forward_step", 
                "Features",
                min=2, max=2, value=2, step=1
            )
        df_steps = forward_res.result()
        counts = df_steps["total_features"].astype(int)
        low, high = counts.min(), counts.max()
        return ui.input_slider(
            "forward_step",
            "Features",
            min=low, max=high,
            value=low, step=1,
        )

    @render_plotly
    def forward_scatter_plot():
        if input.run_forward() < 1 or forward_res.result() is None:
            fig = go.Figure()
            fig.update_layout(
                title="…waiting for forward selection…",
                width=600, height=550
            )
            return fig

        df_steps = forward_res.result()
        selected_n = input.forward_step()
        mask = df_steps["total_features"] == selected_n
        if not mask.any():
            fig = go.Figure()
            fig.update_layout(
                title=f"No record for {selected_n} features.",
                width=600, height=550
            )
            return fig

        idx = df_steps.index[mask][0]
        added_lists = df_steps.loc[:idx, "feature_added"].str.split(",")
        sel_feats = [feat for sub in added_lists for feat in sub]

        X_scaled = StandardScaler().fit_transform(df[sel_feats])
        Y_dummy  = pd.get_dummies(df[label_col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pls = PLSRegression(n_components=2, scale=False).fit(X_scaled, Y_dummy)
        scores = pls.x_scores_

        df_sc = pd.DataFrame(scores, columns=["Component1","Component2"])
        df_sc["Class"] = df[label_col].values

        cmap = {
            cl: c for cl,c in zip(
                sorted(df_sc["Class"].unique()),
                ["#c3121e","#0348a1","#ffb01c","#027608",
                 "#1dace6","#9c5300","#9966cc","#ff4500"]
            )
        }
        fig = px.scatter(
            df_sc,
            x="Component1", y="Component2",
            color="Class",
            template="ggplot2",
            title=f"{selected_n} Features",
            color_discrete_map=cmap
        )
        fig.update_traces(marker=dict(size=26, opacity=0.6))
        fig.update_layout(width=600, height=550)
        return fig

    # ─────────────────────────────────────────────────────────────────────────────
    # Backward Elimination (exactly like original)
    # ─────────────────────────────────────────────────────────────────────────────
    @reactive.Effect
    @reactive.event(input.run_backward)
    def _start_backward():
        numeric = df.drop(columns=[label_col])
        target  = df[label_col]
        backward_res(numeric, target)

    @render_plotly
    def backward_perf_plot():
        if input.run_backward() < 1 or backward_res.result() is None:
            fig = go.Figure()
            fig.update_layout(
                title="Waiting to run backward selection…",
                template="ggplot2"
            )
            return fig

        df_steps = backward_res.result()
        fig = go.Figure(go.Scatter(
            x=df_steps['total_features'],
            y=df_steps['accuracy'],
            mode="lines+markers"
        ))
        low  = df_steps['total_features'].min() - 1
        high = df_steps['total_features'].max() + 1
        fig.update_layout(
            title="Backward Elimination Accuracy",
            xaxis_title="Number of Features",
            yaxis_title="Accuracy",
            template="ggplot2", height=300
        )
        fig.update_xaxes(autorange=False, range=[high, low], tickmode='linear')
        return fig

    @render.data_frame
    def backward_log():
        if input.run_backward() < 1 or backward_res.result() is None:
            return pd.DataFrame(columns=[
                'step', 'accuracy', 'feature_removed', 'total_features'
            ])
        return backward_res.result()
    
    @render.text
    def backward_final_feats():
        if input.run_backward() < 1 or backward_res.result() is None:
            return ""
        # get the elimination log
        df_steps = backward_res.result()
        # start with all features
        current_feats = list(df.drop(columns=[label_col]).columns)
        # remove in order of the recorded removals
        for _, row in df_steps.sort_values("step").iterrows():
            feat = row["feature_removed"]
            if feat:
                current_feats.remove(feat)
        # format as a nice comma-list
        return f"Final {len(current_feats)} features: " + ", ".join(current_feats)

    @render.ui
    def backward_slider_ui():
        if input.run_backward() < 1 or backward_res.result() is None:
            # lock slider at full set size until done
            full_n = len(df.columns) - 1
            return ui.input_slider(
                "backward_step", "Features",
                min=2, max=full_n, value=full_n, step=-1
            )
        df_steps = backward_res.result()
        counts = df_steps['total_features'].tolist()
        low, high = min(counts), max(counts)
        return ui.input_slider(
            "backward_step", "Features",
            min=low, max=high, value=high, step=-1
        )

    @render_plotly
    def backward_scatter_plot():
        if input.run_backward() < 1 or backward_res.result() is None:
            fig = go.Figure()
            fig.update_layout(
                title="…waiting for backward selection…",
                width=600, height=550
            )
            return fig

        df_steps = backward_res.result()
        target_n = input.backward_step()
        current_feats = list(df.drop(columns=[label_col]).columns)

        for _, row in df_steps.sort_values('step').iterrows():
            if row['feature_removed'] and len(current_feats) > target_n:
                current_feats.remove(row['feature_removed'])
            if len(current_feats) == target_n:
                break

        X = StandardScaler().fit_transform(df[current_feats])
        Y = pd.get_dummies(df[label_col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pls = PLSRegression(n_components=2, scale=False).fit(X, Y)
        scores = pls.x_scores_

        df_sc = pd.DataFrame(scores, columns=["Component1","Component2"])
        df_sc["Class"] = df[label_col].values

        cmap = {
            cl: c for cl,c in zip(
                sorted(df_sc["Class"].unique()),
                ["#c3121e","#0348a1","#ffb01c","#027608",
                 "#1dace6","#9c5300","#9966cc","#ff4500"]
            )
        }
        fig = px.scatter(
            df_sc, x="Component1", y="Component2",
            color="Class",
            template="ggplot2",
            title=f"{len(current_feats)} Features",
            color_discrete_map=cmap
        )
        fig.update_traces(marker=dict(size=26, opacity=0.6))
        fig.update_layout(width=600, height=550)
        return fig 