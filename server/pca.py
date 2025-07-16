"""
PCA server logic for principal component analysis and element mapping.
"""

import numpy as np
import pandas as pd
import warnings
from shiny import reactive, render, ui
from shinywidgets import render_plotly
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

from config.feature_groups import FEATURE_GROUPS_PCA
from config.element_groups import SYMBOL_TO_GROUP
from config.settings import DEFAULT_PCA_FEATURES
from utils.helpers import make_safe_id


def setup_pca_server(input, output, session):
    """Set up PCA server logic exactly like original app.py."""
    
    # Track previous group states to prevent infinite loops (exactly like original)
    prev_group_states_pca = {
        gid: all(col in DEFAULT_PCA_FEATURES for _, col in info["features"])
        for gid, info in FEATURE_GROUPS_PCA.items()
    }

    # Select/Deselect all groups (exactly like original)
    @reactive.Effect
    @reactive.event(input.pca_select_all)
    def pca_select_all():
        for gid, info in FEATURE_GROUPS_PCA.items():
            ui.update_checkbox(f"pca_group_{gid}", value=True, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"pca_feat_{fid}", value=True, session=session)

    @reactive.Effect
    @reactive.event(input.pca_deselect_all)
    def pca_deselect_all():
        for gid, info in FEATURE_GROUPS_PCA.items():
            ui.update_checkbox(f"pca_group_{gid}", value=False, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"pca_feat_{fid}", value=False, session=session)

    # Group toggle effects (exactly like original)
    for gid, info in FEATURE_GROUPS_PCA.items():
        @reactive.Effect
        @reactive.event(getattr(input, f"pca_group_{gid}"))
        def _grp_toggle(gid=gid, info=info):
            new_state = getattr(input, f"pca_group_{gid}")()
            old_state = prev_group_states_pca[gid]
            if new_state != old_state:
                prev_group_states_pca[gid] = new_state
                for _, col in info["features"]:
                    fid = make_safe_id(col)
                    ui.update_checkbox(f"pca_feat_{fid}", value=new_state, session=session)

    # PCA calculation (exactly like original)
    @reactive.Calc
    def pca_res():
        data = pd.read_excel("data/elemental-property-list.xlsx")

        # 1) pull out only the user‐selected columns
        sel = [col
            for _, info in FEATURE_GROUPS_PCA.items()
            for disp, col in info["features"]
            if getattr(input, f"pca_feat_{make_safe_id(col)}")()]
        sel = [c for c in sel if c in data.columns]
        if not sel:
            return None

        # 2) build your DataFrame, drop NaN‐columns *first*, then drop zero‐variance
        Xdf = data[sel].copy()
        Xdf = Xdf.dropna(axis=1, how="any")              # ← drop any column with at least one NaN
        zero_var = Xdf.std(axis=0) == 0
        if zero_var.any():
            Xdf = Xdf.loc[:, ~zero_var]
        if Xdf.shape[1] < 2:
            return None   # not enough valid features

        # 3) scale and PCA as before with warnings suppression
        X = StandardScaler().fit_transform(Xdf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pca = PCA(n_components=2, random_state=42).fit(X)
            pcs = pca.transform(X)

        # 4) Build a DataFrame for plotting
        dfp = pd.DataFrame({
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "Symbol": data["Symbol"].astype(str)
        })

        # 5) Explained‐variance
        pc1_pct = pca.explained_variance_ratio_[0] * 100
        pc2_pct = pca.explained_variance_ratio_[1] * 100

        # 6) Top‐10 loadings
        comps = pca.components_
        def top10(component):
            loadings = list(zip(sel, component))
            loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            return loadings[:10]

        top1 = top10(comps[0])
        top2 = top10(comps[1])

        contrib_df = pd.DataFrame({
            "PC1 Feature": [f for f, _ in top1],
            "PC1 Loading": [f"{v:.3f}" for _, v in top1],
            "PC2 Feature": [f for f, _ in top2],
            "PC2 Loading": [f"{v:.3f}" for _, v in top2],
        })

        return {
            "dfp": dfp,
            "pc1_pct": pc1_pct,
            "pc2_pct": pc2_pct,
            "contrib": contrib_df
        }

    @render_plotly
    def pca_plot():
        res = pca_res()
        if res is None:
            fig = go.Figure()
            fig.update_layout(
                title="No features selected or no data.",
                template="ggplot2",
                width=800,
                height=800,
            )
            return fig

        # 1) assign each point to a group (or "other")
        dfp = res["dfp"].copy()
        dfp["Group"] = dfp["Symbol"].map(SYMBOL_TO_GROUP).fillna("other")

        # 2) scatter with color by Group
        fig = px.scatter(
            dfp,
            x="PC1", y="PC2", text="Symbol",
            color="Group",
            color_discrete_map={
                "alkali_metals": "blue",
                "alkaline_earth_metals": "turquoise",
                "transition_metals": "palegreen",
                "lanthanides": "yellow",
                "actinides": "goldenrod",
                "metalloids": "orange",
                "non_metals": "orangered",
                "halogens": "red",
                "noble_gases": "skyblue",
                "post_transition_metals": "darkgreen",
                "Other": "grey"
            },
            labels={
              "PC1": f"PC 1 ({res['pc1_pct']:.1f}%)",
              "PC2": f"PC 2 ({res['pc2_pct']:.1f}%)"
            },
            template="ggplot2"
        )
        fig.update_traces(marker=dict(
                                    size=24, 
                                    opacity=0.6))
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=False,
            font=dict(size=16),
            autosize=True,
            width=800,
            height=800,
        )
        return fig
        
    @render.data_frame
    def pca_contrib_table():
        res = pca_res()
        if res is None:
            return pd.DataFrame({"Message": ["No PCA results."]})
        return res["contrib"] 