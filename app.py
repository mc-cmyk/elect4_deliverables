"""ReviewLens - Part 3: Streamlit dashboard for Jollibee branch managers.

Run after train_grid.py and extract_complaints.py:
    streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = Path(__file__).parent / "output"
SENTIMENTS = ["Negative", "Neutral", "Positive"]
# Which column drives the overview charts: the star-rating label ("sentiment")
# or the model's reading of the text ("predicted").
SENTIMENT_COL = "sentiment"

# Status colours (red / amber / green) - always shown with a text label.
SENT_COLORS = {"Negative": "#d03b3b", "Neutral": "#fab219", "Positive": "#0ca30c"}
SENT_TINTS = {"Negative": "#f8d7d7", "Neutral": "#fdefc8", "Positive": "#d6f0d6"}
BLUE = "#2a78d6"
BLUE_RAMP = ["#f1f6fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fde6da", "#f6b796", "#eb6834", "#b84a1f"]
MUTED = "#898781"
BASELINE_NOTE = "48.8%"
NOTABLE_LIFT = 1.3

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")


# ------------------------------------------------------------------ data
def files_stamp():
    """Modification times of the output files - re-running the pipeline
    changes this, which invalidates the cached data on the next refresh."""
    return tuple(sorted((p.name, p.stat().st_mtime) for p in OUT.glob("*.csv")))


@st.cache_data
def load(stamp):
    needed = {
        "predictions_test.csv": "python train_grid.py",
        "cv_results.csv": "python train_grid.py",
        "complaint_terms_chain.csv": "python extract_complaints.py",
        "complaint_terms_branch.csv": "python extract_complaints.py",
        "complaint_themes_branch.csv": "python extract_complaints.py",
        "branch_priorities.csv": "python extract_complaints.py",
        "jollibee_branch_summary.csv": None,
    }
    missing = [(f, cmd) for f, cmd in needed.items() if not (OUT / f).exists()]
    if missing:
        return None, missing

    read = lambda f: pd.read_csv(OUT / f, encoding="utf-8-sig")
    reviews = read("predictions_test.csv")
    # Stored in UTC; branch managers think in Philippine time.
    reviews["review_date"] = (pd.to_datetime(reviews["review_date"], utc=True)
                              .dt.tz_convert("Asia/Manila").dt.tz_localize(None))
    reviews["month"] = reviews["review_date"].dt.to_period("M").dt.to_timestamp()
    data = {
        "reviews": reviews,
        "cv": read("cv_results.csv"),
        "terms_chain": read("complaint_terms_chain.csv"),
        "terms_branch": read("complaint_terms_branch.csv"),
        "themes": read("complaint_themes_branch.csv"),
        "priorities": read("branch_priorities.csv"),
        "branches": read("jollibee_branch_summary.csv"),
    }
    return data, []


def style_fig(fig, height):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
                      hoverlabel=dict(font_size=13))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(137,135,129,0.18)", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def bar_height(n_rows):
    return max(260, 26 * n_rows + 80)


data, missing = load(files_stamp())
if data is None:
    st.title("ReviewLens")
    st.error("Some result files are missing. Run these first, then refresh:")
    st.code("\n".join(sorted({cmd for _, cmd in missing if cmd})), language="bash")
    st.stop()

reviews, cv = data["reviews"], data["cv"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Filters")
    all_branches = sorted(reviews["Branch"].unique())
    sel_branches = st.multiselect("Branch", all_branches, placeholder="All branches")
    d_min, d_max = reviews["review_date"].min().date(), reviews["review_date"].max().date()
    date_range = st.date_input("Date range", (d_min, d_max), min_value=d_min,
                               max_value=d_max, format="MM/DD/YYYY")
    sel_sent = st.multiselect("Sentiment", SENTIMENTS, default=SENTIMENTS)

branches_in_view = sel_branches or all_branches
start, end = (date_range if isinstance(date_range, tuple) and len(date_range) == 2
              else (d_min, d_max))
view = reviews[
    reviews["Branch"].isin(branches_in_view)
    & reviews["review_date"].dt.date.between(start, end)
    & reviews[SENTIMENT_COL].isin(sel_sent or SENTIMENTS)
]

st.title("ReviewLens")
st.caption(f"Jollibee Foodpanda reviews · {start:%b %d, %Y} – {end:%b %d, %Y} · "
           f"{'all branches' if not sel_branches else f'{len(sel_branches)} branch(es)'}")

tabs = st.tabs(["Sentiment Overview", "Complaint Summary", "Branch Comparison",
                "Review Table", "Evaluation"])

# ------------------------------------------------------------------ 1. overview
with tabs[0]:
    if view.empty:
        st.info("No reviews match these filters.")
    else:
        neg = view[view[SENTIMENT_COL] == "Negative"]
        top = neg["Branch"].value_counts()
        k1, k2, k3 = st.columns(3)
        k1.metric("Total reviews", f"{len(view):,}")
        k2.metric("Negative reviews", f"{len(neg) / len(view):.1%}")
        k3.metric("Most complained-about branch",
                  top.index[0] if len(top) else "—",
                  f"{top.iloc[0]} negative reviews" if len(top) else None,
                  delta_color="off", delta_arrow="off")

        left, right = st.columns([1, 2])
        with left:
            st.subheader("Overall sentiment")
            counts = view[SENTIMENT_COL].value_counts().reindex(SENTIMENTS).dropna()
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values, hole=0.62, sort=False,
                marker=dict(colors=[SENT_COLORS[s] for s in counts.index],
                            line=dict(color="white", width=2)),
                textinfo="percent", textposition="inside", insidetextorientation="horizontal",
                hovertemplate="%{label}: %{value:,} reviews (%{percent})<extra></extra>"))
            st.plotly_chart(style_fig(fig, 330))

        with right:
            st.subheader("Monthly trend")
            monthly = (view.groupby(["month", SENTIMENT_COL]).size()
                       .rename("reviews").reset_index())
            fig = px.line(monthly, x="month", y="reviews", color=SENTIMENT_COL,
                          color_discrete_map=SENT_COLORS, markers=True,
                          category_orders={SENTIMENT_COL: SENTIMENTS},
                          labels={"month": "", "reviews": "Reviews", SENTIMENT_COL: ""})
            fig.update_traces(line_width=2, marker_size=7,
                              hovertemplate="%{x|%B %Y}: %{y} reviews<extra>%{fullData.name}</extra>")
            fig.update_xaxes(tickformat="%b %Y", dtick="M2")
            st.plotly_chart(style_fig(fig, 330))

        st.subheader("Sentiment by branch")
        st.caption("Branches sorted by share of negative reviews, highest at the top.")
        by_branch = (view.groupby(["Branch", SENTIMENT_COL]).size()
                     .unstack(fill_value=0).reindex(columns=SENTIMENTS, fill_value=0))
        by_branch["neg_pct"] = by_branch["Negative"] / by_branch[SENTIMENTS].sum(axis=1)
        order = by_branch.sort_values("neg_pct").index.tolist()
        long = by_branch.reset_index().melt(id_vars=["Branch", "neg_pct"], value_vars=SENTIMENTS,
                                            var_name="Sentiment", value_name="Reviews")
        fig = px.bar(long, y="Branch", x="Reviews", color="Sentiment", orientation="h",
                     color_discrete_map=SENT_COLORS, category_orders={"Branch": order[::-1],
                                                                      "Sentiment": SENTIMENTS},
                     custom_data=["neg_pct"], labels={"Branch": ""})
        fig.update_traces(marker_line_color="white", marker_line_width=1.5,
                          hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x} reviews"
                                        "<br>Negative share: %{customdata[0]:.0%}<extra></extra>")
        fig.update_yaxes(categoryorder="array", categoryarray=order)
        st.plotly_chart(style_fig(fig, bar_height(len(order))))

# ------------------------------------------------------------------ 2. complaints
with tabs[1]:
    n_complaints = int(data["priorities"]["n_complaints"].sum())
    st.caption(f"Based on all {n_complaints:,} negative reviews from Feb 2024 to Apr 2025. "
               "The date and sentiment filters don't apply on this page; the branch filter does.")

    st.subheader("What customers complain about most")
    terms = data["terms_chain"].head(20).iloc[::-1]
    fig = px.bar(terms, x="n_reviews", y="term", orientation="h",
                 labels={"n_reviews": "Complaints mentioning it", "term": ""},
                 custom_data=["pct_reviews"])
    fig.update_traces(marker_color=BLUE,
                      hovertemplate="<b>%{y}</b><br>%{x} complaints (%{customdata[0]:.1%})<extra></extra>")
    st.plotly_chart(style_fig(fig, bar_height(20)))

    st.subheader("Each branch's top issue")
    st.caption(f"\"Times the chain average\" compares how often an issue comes up at this branch "
               f"with all branches combined. Above {NOTABLE_LIFT}× is worth a look.")
    pri = data["priorities"]
    pri = pri[pri["Branch"].isin(branches_in_view)].copy()
    pri["top_issue_1_rate"] *= 100
    pri["top_issue_2_rate"] *= 100
    pri["share_of_chain_complaints"] *= 100
    st.dataframe(
        pri[["Branch", "n_complaints", "top_issue_1", "top_issue_1_lift", "top_issue_1_rate",
             "top_issue_2", "top_issue_2_lift", "share_of_chain_complaints"]],
        hide_index=True,
        column_config={
            "n_complaints": st.column_config.NumberColumn("Complaints"),
            "top_issue_1": "Top issue",
            "top_issue_1_lift": st.column_config.NumberColumn("Times the chain average", format="%.1f×"),
            "top_issue_1_rate": st.column_config.NumberColumn("% of branch complaints", format="%.0f%%"),
            "top_issue_2": "Second issue",
            "top_issue_2_lift": st.column_config.NumberColumn("Second issue vs chain", format="%.1f×"),
            "share_of_chain_complaints": st.column_config.NumberColumn("Share of all complaints",
                                                                       format="%.1f%%"),
        })

    st.subheader("Common phrases at one branch")
    tb = data["terms_branch"]
    options = [b for b in branches_in_view if b in set(tb["Branch"])]
    if options:
        pick = st.selectbox("Branch", options, label_visibility="collapsed")
        bt = tb[tb["Branch"] == pick].head(15).iloc[::-1]
        if bt.empty:
            st.info("Not enough complaints at this branch to find repeated phrases.")
        else:
            fig = px.bar(bt, x="n_reviews", y="term", orientation="h",
                         labels={"n_reviews": "Complaints mentioning it", "term": ""})
            fig.update_traces(marker_color=BLUE,
                              hovertemplate="<b>%{y}</b><br>%{x} complaints<extra></extra>")
            fig.update_xaxes(tick0=0, dtick=1 if bt["n_reviews"].max() <= 10 else None)
            st.plotly_chart(style_fig(fig, bar_height(len(bt))))
    else:
        st.info("None of the selected branches have complaint records.")

# ------------------------------------------------------------------ 3. comparison
with tabs[2]:
    st.subheader("Complaint issues by branch")
    st.caption("Each cell: share of that branch's complaints mentioning the issue. "
               f"Orange cells come up more than {NOTABLE_LIFT}× as often as the chain average "
               "(with at least 3 complaints). Same scope as the Complaint Summary page.")
    th = data["themes"][data["themes"]["Branch"].isin(branches_in_view)]
    if th.empty:
        st.info("None of the selected branches have complaint records.")
    else:
        theme_order = (data["themes"].groupby("theme_label")["chain_rate"].first()
                       .sort_values(ascending=False).index.tolist())
        branch_order = (th.groupby("Branch")["n_complaints"].first()
                        .sort_values(ascending=False).index.tolist())
        piv = lambda col: th.pivot(index="Branch", columns="theme_label", values=col) \
                            .reindex(index=branch_order, columns=theme_order)
        rate, lift, hits, notable = piv("rate") * 100, piv("lift"), piv("n_hits"), piv("notable")
        n_c = th.groupby("Branch")["n_complaints"].first().reindex(branch_order)
        custom = np.dstack([lift.values, hits.values,
                            np.repeat(n_c.values[:, None], len(theme_order), axis=1)])
        hover = ("<b>%{y}</b> · %{x}<br>%{z:.0f}% of complaints (%{customdata[1]:.0f} of "
                 "%{customdata[2]:.0f})<br>%{customdata[0]:.1f}× the chain average<extra></extra>")
        to_scale = lambda ramp: [[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)]
        zmax = float(np.nanmax(rate.values))
        is_notable = notable.astype(bool).values
        labels = np.vectorize(lambda v: f"{v:.0f}%")(rate.values)
        # Two layers: blue for every cell, orange drawn over the notable ones.
        # Each layer labels only its own cells so the text never doubles up.
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=rate.values, x=theme_order, y=branch_order, customdata=custom,
            colorscale=to_scale(BLUE_RAMP), zmin=0, zmax=zmax, xgap=2, ygap=2,
            text=np.where(is_notable, "", labels), texttemplate="%{text}", hovertemplate=hover,
            colorbar=dict(title="% of complaints", ticksuffix="%", thickness=12)))
        fig.add_trace(go.Heatmap(
            z=np.where(is_notable, rate.values, np.nan), x=theme_order, y=branch_order,
            customdata=custom, colorscale=to_scale(ORANGE_RAMP), zmin=0, zmax=zmax,
            xgap=2, ygap=2, text=np.where(is_notable, labels, ""), texttemplate="%{text}",
            hovertemplate=hover, showscale=False))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(side="top", showgrid=False)
        st.plotly_chart(style_fig(fig, max(320, 28 * len(branch_order) + 90)))

    st.subheader("Negative reviews by branch")
    bs = data["branches"][data["branches"]["Branch"].isin(branches_in_view)].copy()
    chain_neg = data["branches"]["n_negative"].sum() / data["branches"]["n_reviews"].sum()
    st.caption(f"All reviews, not just the test sample. The line marks the chain average "
               f"({chain_neg:.0%}). Branches with fewer than 30 reviews are greyed out.")
    bs = bs.sort_values("pct_negative")
    bs["label"] = np.where(bs["usable"], "Enough reviews", "Fewer than 30 reviews")
    fig = px.bar(bs, x="pct_negative", y="Branch", orientation="h", color="label",
                 color_discrete_map={"Enough reviews": BLUE, "Fewer than 30 reviews": "#c3c2b7"},
                 custom_data=["n_negative", "n_reviews"],
                 labels={"pct_negative": "Share of negative reviews", "Branch": "", "label": ""})
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:.0%} negative "
                                    "(%{customdata[0]} of %{customdata[1]})<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=bs["Branch"].tolist())
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.add_vline(x=chain_neg, line_color=MUTED, line_width=1.5,
                  annotation_text=f"Chain average {chain_neg:.0%}", annotation_position="bottom right",
                  annotation_font_color=MUTED)
    fig.update_layout(showlegend=not bs["usable"].all())
    st.plotly_chart(style_fig(fig, bar_height(len(bs))))

# ------------------------------------------------------------------ 4. reviews
with tabs[3]:
    query = st.text_input("Search reviews", placeholder="e.g. gravy, malamig, late")
    table = view
    if query:
        table = table[table["text_clean"].str.contains(query, case=False, regex=False, na=False)]
    st.caption(f"{len(table):,} reviews · click a column header to sort")
    if not table.empty:
        shown = pd.DataFrame({
            "Branch": table["Branch"],
            "Date": table["review_date"],
            "Rating": table["rating"],
            "Sentiment": table["sentiment"],
            "Predicted": table["predicted"],
            "Tags": [[t for t, on in (("Short review", s), ("Rider issue", r)) if on]
                     for s, r in zip(table["is_short"], table["rider_blamed"])],
            "Review": table["text_clean"],
        }).sort_values("Date", ascending=False)
        paint = lambda v: (f"background-color: {SENT_TINTS[v]}; color: #0b0b0b"
                           if v in SENT_TINTS else "")
        st.dataframe(
            shown.style.map(paint, subset=["Sentiment", "Predicted"]),
            hide_index=True, height=560,
            column_config={
                "Date": st.column_config.DatetimeColumn("Date", format="MMM D, YYYY h:mm A"),
                "Rating": st.column_config.NumberColumn("Rating", format="%d ★"),
                "Sentiment": st.column_config.TextColumn(
                    "Sentiment", help="From the customer's star rating"),
                "Predicted": st.column_config.TextColumn(
                    "Predicted", help="What ReviewLens read from the review text"),
                "Review": st.column_config.TextColumn("Review text"),
                "Tags": st.column_config.ListColumn(
                    "Tags", help="Short review: under 3 words. Rider issue: complaint is "
                                 "about the delivery rider, not the branch."),
            })

# ------------------------------------------------------------------ 5. evaluation
with tabs[4]:
    st.caption("Developer view: how the sentiment model was chosen and how well it does. "
               "Uses the full held-out test set; the sidebar filters don't apply here.")
    best = cv[cv["is_best"]].iloc[0]
    y, p = reviews["sentiment"], reviews["predicted"]
    acc = (y == p).mean()
    majority = (y == y.value_counts().idxmax()).mean()
    per_class = []
    for c in SENTIMENTS:
        tp = ((p == c) & (y == c)).sum()
        prec = tp / max((p == c).sum(), 1)
        rec = tp / max((y == c).sum(), 1)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class.append({"Class": c, "Precision": prec, "Recall": rec, "F1": f1})
    per_class = pd.DataFrame(per_class)

    m1, m2, m3, m4 = st.columns([1.7, 1.2, 1, 1.2])
    m1.metric("Selected model", f"{best['model']}", f"{best['vectorizer']} {best['ngram_range']}",
              delta_color="off", delta_arrow="off")
    m2.metric("CV macro-F1", f"{best['macro_f1_mean']:.3f}", f"± {best['macro_f1_std']:.3f} across folds",
              delta_color="off", delta_arrow="off")
    m3.metric("Test macro-F1", f"{per_class['F1'].mean():.3f}")
    m4.metric("Test accuracy", f"{acc:.1%}", f"{(acc - majority) * 100:+.1f} pts vs majority baseline")
    st.info(f"**Majority-class baseline is {BASELINE_NOTE}** (always predicting Negative). "
            "Macro-F1 is the selection metric because it weights the small Neutral class equally.")

    st.subheader("All 25 configurations (5-fold cross-validation)")
    cols = {"rank": "Rank", "config": "Configuration", "macro_f1_mean": "Macro-F1",
            "macro_f1_std": "± std", "accuracy_mean": "Accuracy", "f1_negative_mean": "F1 Negative",
            "f1_neutral_mean": "F1 Neutral", "f1_positive_mean": "F1 Positive"}
    cv_show = cv[list(cols) + ["is_best"]].rename(columns=cols)
    highlight = lambda row: (["background-color: #cde2fb; color: #0b0b0b; font-weight: 600"]
                             * len(row) if row["is_best"] else [""] * len(row))
    st.dataframe(cv_show.style.apply(highlight, axis=1).format(precision=3),
                 hide_index=True, height=35 * 26 + 3, column_order=list(cols.values()))
    afinn_note = cv.loc[cv["vectorizer"] == "AFINN lexicon", "note"]
    if len(afinn_note):
        st.caption(f"AFINN baseline: {afinn_note.iloc[0]} Many Taglish reviews contain no AFINN "
                   "word at all and score 0, which defaults them to Neutral.")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix (test set)")
        cm = pd.crosstab(y, p).reindex(index=SENTIMENTS, columns=SENTIMENTS, fill_value=0)
        row_pct = cm.div(cm.sum(axis=1), axis=0)
        cell_text = [[f"{n}<br>({r:.0%})" for n, r in zip(nr, rr)]
                     for nr, rr in zip(cm.values, row_pct.values)]
        fig = go.Figure(go.Heatmap(
            z=cm.values, x=[f"Predicted {s}" for s in SENTIMENTS],
            y=[f"Actual {s}" for s in SENTIMENTS], text=cell_text,
            colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)],
            xgap=2, ygap=2, texttemplate="%{text}", textfont_size=14,
            hovertemplate="%{y} → %{x}: %{z} reviews<extra></extra>",
            showscale=False))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(side="top", showgrid=False)
        st.plotly_chart(style_fig(fig, 360))
    with right:
        st.subheader("Per-class precision, recall, F1")
        pc_long = per_class.melt(id_vars="Class", var_name="Metric", value_name="Score")
        fig = px.bar(pc_long, x="Class", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a"],
                     text_auto=".2f", labels={"Class": ""})
        fig.update_traces(marker_line_color="white", marker_line_width=2, textposition="outside",
                          hovertemplate="%{x} %{fullData.name}: %{y:.3f}<extra></extra>")
        fig.update_yaxes(range=[0, 1.08], showgrid=True, gridcolor="rgba(137,135,129,0.18)")
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(style_fig(fig, 360))

    short = reviews["is_short"].astype(bool)
    st.warning(
        f"**Why Neutral scores poorly (F1 {per_class.loc[1, 'F1']:.2f}).** Ratings are U-shaped: "
        "most customers give 1 or 5 stars, so only ~10% of reviews are 3-star Neutral. "
        "A 3-star review's text usually reads as mildly positive or mildly negative "
        "(\"ok naman pero late\"), so the model confuses Neutral with both neighbours - "
        "visible in the middle row and column of the confusion matrix. This is a property "
        "of the labels, not a bug, and it is reported as-is.")
    st.caption(f"Short reviews (under 3 words) are kept: {(y[short] == p[short]).mean():.0%} accuracy "
               f"on {short.sum()} short reviews vs {(y[~short] == p[~short]).mean():.0%} on "
               f"{(~short).sum()} longer ones. Dropping them would remove mostly Positive reviews "
               "and skew the class balance.")
