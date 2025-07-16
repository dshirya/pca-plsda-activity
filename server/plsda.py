"""
PLS-DA server logic for classification analysis and visualization.
"""

import numpy as np
import pandas as pd
import warnings
from shiny import reactive, render, ui
from shinywidgets import render_widget
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

from config.feature_groups import FEATURE_GROUPS_PLSDA
from config.settings import DEFAULT_PLS_FEATURES
from utils.helpers import make_safe_id
from utils.data_processing import load_features_data


def setup_plsda_server(input, output, session):
    """Set up PLS-DA server logic exactly like original app.py."""
    
    # Track previous group states to prevent infinite loops (exactly like original)
    prev_group_states = {
        gid: all(col in DEFAULT_PLS_FEATURES for _, col in info["features"])
        for gid, info in FEATURE_GROUPS_PLSDA.items()
    }

    # Select/Deselect all groups (exactly like original)
    @reactive.Effect
    @reactive.event(input.select_all)
    def plsda_select_all():
        for gid, info in FEATURE_GROUPS_PLSDA.items():
            ui.update_checkbox(f"group_{gid}", value=True, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"feat_{fid}", value=True, session=session)

    @reactive.Effect
    @reactive.event(input.deselect_all)
    def plsda_deselect_all():
        for gid, info in FEATURE_GROUPS_PLSDA.items():
            ui.update_checkbox(f"group_{gid}", value=False, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"feat_{fid}", value=False, session=session)

    # Group toggle effects (exactly like original)
    for gid, info in FEATURE_GROUPS_PLSDA.items():
        @reactive.Effect
        @reactive.event(getattr(input, f"group_{gid}"))
        def _grp_toggle(gid=gid, info=info):
            new_state = getattr(input, f"group_{gid}")()
            old_state = prev_group_states[gid]
            # only run if it really changed
            if new_state != old_state:
                prev_group_states[gid] = new_state
                # propagate to all children
                for _, col in info["features"]:
                    fid = make_safe_id(col)
                    ui.update_checkbox(f"feat_{fid}", value=new_state, session=session)

    # Core PLS-DA reactive calculation (exactly like original)
    @reactive.Calc
    def run_plsda():
        # 1) Gather selected columns
        sel = []
        for info in FEATURE_GROUPS_PLSDA.values():
            for _, col in info["features"]:
                if getattr(input, f"feat_{make_safe_id(col)}")():
                    sel.append(col)
        
        # Load data
        df = load_features_data()
        valid = [c for c in sel if c in df.columns]
        if not valid:
            return None

        if len(valid) < 2:
            return None
        
        # 2) Prepare X, y
        X = df[valid].values
        y = df["Class"].values

        # 3) Standardize the full data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 4) Fit PLS‐DA on full data
            #    One‐hot encode y for PLSRegression
            Y_dummy = pd.get_dummies(y)
            pls = PLSRegression(n_components=2, scale=True)
            pls.fit(X_scaled, Y_dummy)

            # 5) Extract scores (x_scores_) for every sample
            scores = pls.x_scores_

            # 6) Explained variance (approximate)
            total_variance = np.var(X_scaled, axis=0).sum()
            var_comp1 = np.var(scores[:, 0])
            var_comp2 = np.var(scores[:, 1])
            pls1_pct = var_comp1 / total_variance * 100
            pls2_pct = var_comp2 / total_variance * 100

            # 7) Classify in the latent space on full data
            #    (use the PLS predictions and take the argmax)
            y_pred_cont = pls.predict(X_scaled)
            pred_idx = np.argmax(y_pred_cont, axis=1)
            pred_labels = [Y_dummy.columns[i] for i in pred_idx]

            # 8) Compute metrics on the full dataset
            acc = accuracy_score(y, pred_labels)
            f1  = f1_score(y, pred_labels, average="macro")
            sil = silhouette_score(scores, y) if len(np.unique(y)) > 1 else np.nan

        # 9) Fisher Discriminant Ratio
        overall_mean = scores.mean(axis=0)
        between_var = 0.0
        within_var = 0.0
        for cls in np.unique(y):
            cls_scores = scores[y == cls]
            m_cls = cls_scores.mean(axis=0)
            between_var += cls_scores.shape[0] * np.sum((m_cls - overall_mean) ** 2)
            within_var  += np.sum((cls_scores - m_cls) ** 2)
        fisher = between_var / within_var if within_var > 1e-6 else np.nan

        # 10) Pack metrics into a DataFrame
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "F1 Score", "Silhouette", "Fisher Ratio"],
            "Value":  [f"{acc:.3f}",
                       f"{f1:.3f}",
                       f"{sil:.3f}" if not np.isnan(sil) else "N/A",
                       f"{fisher:.3f}" if not np.isnan(fisher) else "N/A"]
        })

        # 11) Top‐5 feature contributions for each component
        W = pls.x_weights_
        w1, w2 = W[:, 0], (W[:, 1] if W.shape[1] > 1 else None)
        top_n = min(10, len(valid))
        idx1 = np.argsort(np.abs(w1))[::-1][:top_n]
        top1 = [(valid[i], w1[i]) for i in idx1]

        if w2 is not None:
            idx2 = np.argsort(np.abs(w2))[::-1][:top_n]
            top2 = [(valid[i], w2[i]) for i in idx2]
        else:
            top2 = []

        contrib_rows = []
        for i in range(max(len(top1), len(top2))):
            f1n, v1 = top1[i] if i < len(top1) else ("", "")
            f2n, v2 = top2[i] if i < len(top2) else ("", "")
            contrib_rows.append([
                f1n, f"{v1:.3f}" if v1 != "" else "",
                f2n, f"{v2:.3f}" if v2 != "" else ""
            ])
        contrib_df = pd.DataFrame(
            contrib_rows,
            columns=["Component 1 feature", "Score", "Component 2 feature", "Score "]
        )

        # 12) Calculate VIP (Variable Importance in Projection) scores
        # VIP formula: sqrt(p * sum(w_jf^2 * SS_f) / sum(SS_f))
        p = len(valid)  # number of variables
        n_comp = min(2, W.shape[1])  # number of components (max 2)
        
        # Calculate sum of squares for each component (approximate from explained variance)
        ss_comp1 = var_comp1 * (X_scaled.shape[0] - 1)  # SS for component 1
        ss_comp2 = var_comp2 * (X_scaled.shape[0] - 1) if n_comp > 1 else 0  # SS for component 2
        total_ss = ss_comp1 + ss_comp2
        
        vip_scores = []
        for j in range(len(valid)):
            w1_sq = W[j, 0] ** 2
            w2_sq = W[j, 1] ** 2 if n_comp > 1 else 0
            
            numerator = w1_sq * ss_comp1 + w2_sq * ss_comp2
            vip_j = np.sqrt(p * numerator / total_ss) if total_ss > 0 else 0
            vip_scores.append(vip_j)
        
        # Create VIP dataframe sorted by VIP scores (descending) - top 10 only
        vip_data = list(zip(valid, vip_scores))
        vip_data.sort(key=lambda x: x[1], reverse=True)
        
        # Take only top 10 VIP scores
        top_vip_data = vip_data[:10]
        
        vip_df = pd.DataFrame(top_vip_data, columns=["Feature", "VIP Score"])
        vip_df["VIP Score"] = vip_df["VIP Score"].map(lambda x: f"{x:.3f}")

        # 13) Return everything for plotting & tables
        return {
            "scores": scores,
            "labels": y,
            "metrics": metrics_df,
            "contrib": contrib_df,
            "vip": vip_df,
            "pls1_pct": pls1_pct,
            "pls2_pct": pls2_pct
        }

    # ——————————————
    # Plot renderer (exactly like original)
    # ——————————————
    @render_widget
    def pls_plot():
        res = run_plsda()
        fig = go.Figure()
        if res is None:
            # no features selected
            fig.update_layout(
                title="PLS-DA Projection (No features selected)",
                template="ggplot2",
                autosize=False,
                width=800,
                height=830
            )
            return fig

        df_sc = pd.DataFrame({
            "LV1": res["scores"][:,0],
            "LV2": res["scores"][:,1],
            "Class": res["labels"]
        })
        cmap = {
            cl: c for cl,c in zip(
                sorted(df_sc["Class"].unique()),
                ["#c3121e","#0348a1","#ffb01c","#027608",
                 "#1dace6","#9c5300","#9966cc","#ff4500"]
            )
        }

        fig = px.scatter(
            df_sc,
            x="LV1", y="LV2", color="Class",
            labels={
                "LV1": f"LV 1 ({res['pls1_pct']:.1f}%)",
                "LV2": f"LV 2 ({res['pls2_pct']:.1f}%)"
            },
            template="ggplot2",
            color_discrete_map=cmap
        )
        fig.update_traces(
            marker=dict(
                size=34, 
                opacity=0.8
            )
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(size=24),
            width=800,
            height=830,
            showlegend=True,

            legend_orientation="h",
            legend_x=0.5,         
            legend_xanchor="center",
            legend_y=1.02,         
            legend_yanchor="bottom",
            legend_title_side="top center"
        )

        fig.update_yaxes(
            scaleanchor="x", 
            scaleratio=1
        )
        fig.update_xaxes(
            constrain="domain"
        )
        return fig

    # ——————————————
    # Metrics table (exactly like original)
    # ——————————————
    @render.data_frame
    def metrics_table():
        res = run_plsda()
        if res is None:
            return pd.DataFrame({"Message":["No features selected."]})
        return res["metrics"]

    # ——————————————
    # Contributions table (exactly like original)
    # ——————————————
    @render.data_frame
    def contrib_table():
        res = run_plsda()
        if res is None:
            return pd.DataFrame()
        return res["contrib"] 

    # ——————————————
    # VIP table
    # ——————————————
    @render.data_frame
    def vip_table():
        res = run_plsda()
        if res is None:
            return pd.DataFrame({"Message": ["No features selected."]})
        return res["vip"]