"""
Clustering server logic for structure type visualization.
"""

import numpy as np
import pandas as pd
import re
import warnings
from shiny import reactive, render, ui
from shinywidgets import render_plotly
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

from config.feature_groups import FEATURE_GROUPS_PCA
from utils.helpers import make_safe_id


def setup_clustering_server(input, output, session):
    """Set up clustering server logic exactly like original app.py."""

    # Select/Deselect all groups (exactly like original)
    @reactive.Effect
    @reactive.event(input.clust_select_all)
    def clust_select_all():
        for gid, info in FEATURE_GROUPS_PCA.items():
            ui.update_checkbox(f"clust_group_{gid}", value=True, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"clust_feat_{fid}", value=True, session=session)

    @reactive.Effect
    @reactive.event(input.clust_deselect_all)
    def clust_deselect_all():
        for gid, info in FEATURE_GROUPS_PCA.items():
            ui.update_checkbox(f"clust_group_{gid}", value=False, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"clust_feat_{fid}", value=False, session=session)

    # Group toggle effects (exactly like original)
    for gid, info in FEATURE_GROUPS_PCA.items():
        @reactive.Effect
        @reactive.event(getattr(input, f"clust_group_{gid}"))
        def _grp_toggle(gid=gid, info=info):
            state = getattr(input, f"clust_group_{gid}")()
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"clust_feat_{fid}", value=state, session=session)

    # Clustering calculation (exactly like original)
    @reactive.Calc
    def clust_res():
        data = pd.read_excel("data/elemental-property-list.xlsx")

        sel = [col for _, info in FEATURE_GROUPS_PCA.items()
                for disp, col in info["features"]
                if getattr(input, f"clust_feat_{make_safe_id(col)}")()]
        sel = [c for c in sel if c in data.columns]
        if not sel:
            return None

        # **DROP zero-variance columns** from your PCA matrix
        Xdf = data[sel].copy()
        zero_var = Xdf.std(axis=0) == 0
        if zero_var.any():
            Xdf = Xdf.loc[:, ~zero_var]
        if Xdf.shape[1] < 2:
            return None  # not enough features for 2 components

        # then proceed with warnings suppression
        X = StandardScaler().fit_transform(Xdf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pca = PCA(n_components=2, random_state=42).fit(X)
            pcs = pca.transform(X)
            
        dfp = pd.DataFrame({
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "Symbol": data["Symbol"].astype(str)
        })
        pc1_pct = pca.explained_variance_ratio_[0] * 100
        pc2_pct = pca.explained_variance_ratio_[1] * 100

        # build the top-10 loading table just like in pca_res()
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

        # b) read hard-coded structure file
        struct_df = pd.read_excel("data/pauling-data.xlsx", usecols=["Formula", "Structure type"])
        def split_formula(f):
            return re.findall(r"[A-Z][a-z]?", f)

        links = []
        for _, row in struct_df.iterrows():
            formula = row["Formula"]
            a, b = split_formula(formula)
            try:
                x0, y0 = dfp.loc[dfp.Symbol==a, ["PC1","PC2"]].iloc[0]
                x1, y1 = dfp.loc[dfp.Symbol==b, ["PC1","PC2"]].iloc[0]
            except IndexError:
                continue
            links.append({
                "x": [x0, x1],
                "y": [y0, y1],
                "mid_x": (x0 + x1) / 2,
                "mid_y": (y0 + y1) / 2,
                "struct": row["Structure type"],
                "formula": formula
            })

        return {
            "dfp": dfp,
            "pc1_pct": pc1_pct,
            "pc2_pct": pc2_pct,
            "contribution": contrib_df,
            "links": links
        }

    # render the clustering plot (exactly like original)
    @render_plotly
    def clust_plot():
        res = clust_res()
        if not res:
            fig = go.Figure()
            fig.update_layout(
                title="No data",
                width=800, 
                height=830
            )
            return fig

        dfp = res["dfp"]
        fig = px.scatter(
            dfp, 
            x="PC1", 
            y="PC2", 
            text="Symbol", 
            template="ggplot2", 
            width=800, 
            height=830
        )
        fig.update_traces(
            marker=dict(
                color="green", 
                size=26, 
                opacity=0.6
            )
        )

        # colour map for your three structure types
        cmap = {
            "CsCl": "#c3121e",
            "NaCl": "#0348a1",
            "ZnS":  "#ffb01c",  
        }

        seen = set()
        for link in res["links"]:
            struct = link["struct"]
            col = cmap.get(struct, "#888888")

            # 1) line trace, grouped with its markers
            fig.add_trace(go.Scatter(
                x=link["x"],
                y=link["y"],
                mode="lines",
                line=dict(color=col, width=1),
                opacity=0.4,
                legendgroup=struct,
                showlegend=False,
                hoverinfo="none"
            ))

            # 2) midpoint marker, same group
            show_leg = struct not in seen
            fig.add_trace(go.Scatter(
                x=[link["mid_x"]],
                y=[link["mid_y"]],
                mode="markers",
                marker=dict(color=col, size=10, opacity=0.6),
                name=struct,
                legendgroup=struct,
                showlegend=show_leg,
                hovertext=[link["formula"]],
                hovertemplate="%{hovertext}"
            ))
            seen.add(struct)

        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            font=dict(size=16),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            width=800, height=830
        )
        return fig
    
    @render.data_frame
    def clust_contrib_table():
        res = clust_res()
        if res is None:
            return pd.DataFrame({"Message": ["No PCA results."]})
        return res["contribution"] 