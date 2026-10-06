import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import student_performance_analysis as spa

INK, INDIGO, AMBER, MIST = "#1B2540", "#3D5AFE", "#F2A900", "#E3E8F2"

st.set_page_config(page_title="Student Performance Analysis", page_icon="📊", layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@500;700&family=Public+Sans:wght@400;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'Public Sans', sans-serif; color: {INK}; }}
    h1, h2, h3 {{ font-family: 'Newsreader', serif; color: {INK}; }}
    .hero {{ border-left: 6px solid {INDIGO}; padding: 4px 0 4px 18px; margin-bottom: 18px; }}
    .hero h1 {{ font-size: 2.3rem; margin: 0; }}
    .hero p {{ margin: 4px 0 0; color: #4A5578; }}
    .kpi {{ background: #fff; border: 1px solid {MIST}; border-radius: 10px; padding: 14px 16px; }}
    .kpi .v {{ font-family: 'Newsreader', serif; font-size: 1.9rem; font-weight: 700; }}
    .kpi .l {{ color: #4A5578; font-size: 0.85rem; }}
    .verdict {{ border-radius: 8px; padding: 10px 14px; margin: 8px 0 16px; font-weight: 600; }}
    .verdict.yes {{ background: #E8EDFF; border-left: 5px solid {INDIGO}; }}
    .verdict.no {{ background: #FFF4D6; border-left: 5px solid {AMBER}; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": MIST, "grid.color": MIST})


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def verdict(reject, text):
    cls = "yes" if reject else "no"
    st.markdown(f'<div class="verdict {cls}">{text}</div>', unsafe_allow_html=True)


# ------------------------------ sidebar ------------------------------
with st.sidebar:
    st.header("Settings")
    benchmark = st.slider("Benchmark mean mark", 40, 90, 60)
    alpha = st.select_slider("Significance level (alpha)", [0.01, 0.05, 0.10], value=0.05)
    distinction = st.slider("Distinction cut-off", 60, 90, 75)
    att_cut = st.slider("Good attendance cut-off (%)", 50, 95, 75)
    upload = st.file_uploader("Or upload your own CSV", type="csv",
                              help="Needs columns: " + ", ".join(spa.COLS))

if upload is not None:
    df = pd.read_csv(upload)
    missing = [c for c in spa.COLS if c not in df.columns]
    if missing:
        st.error(f"Your CSV is missing these columns: {', '.join(missing)}. Using sample data instead.")
        df = spa.generate_data()
else:
    df = spa.generate_data()

st.markdown(
    '<div class="hero"><h1>Do study habits decide final marks?</h1>'
    "<p>A statistics and probability analysis of student performance: study hours, "
    "attendance, assignments and sleep.</p></div>",
    unsafe_allow_html=True,
)

ov = spa.data_overview(df)
c1, c2, c3, c4 = st.columns(4)
kpi(c1, ov["rows"], "Students")
kpi(c2, f"{df[spa.TARGET].mean():.1f}", "Average final mark")
kpi(c3, f"{(df[spa.TARGET] >= distinction).mean():.0%}", f"Scored {distinction}+")
kpi(c4, f"{df['study_hours'].corr(df[spa.TARGET]):.2f}", "Study hours vs marks (r)")
st.write("")

tabs = st.tabs(["Data", "Descriptive", "Probability", "Outliers", "Correlation",
                "Hypothesis tests", "Recommendations"])

# ------------------------------ data ------------------------------
with tabs[0]:
    st.subheader("Data understanding")
    st.dataframe(df, width='stretch', height=300)
    st.caption(f"{ov['rows']} rows, {ov['columns']} columns, {ov['missing_values']} missing "
               f"values, {ov['duplicates']} duplicate rows. Sample data is synthetic and seeded.")

# ------------------------------ descriptive ------------------------------
with tabs[1]:
    st.subheader("Descriptive statistics")
    st.dataframe(spa.descriptive_stats(df), width='stretch')
    col = st.selectbox("Column to plot", spa.COLS, index=4)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sns.histplot(df[col], kde=True, color=INDIGO, ax=ax)
    ax.axvline(df[col].mean(), color=AMBER, lw=2, label="mean")
    ax.axvline(df[col].median(), color=INK, lw=2, ls="--", label="median")
    ax.legend()
    st.pyplot(fig)

# ------------------------------ probability ------------------------------
with tabs[2]:
    prob = spa.probabilities(df, distinction=distinction, att_cut=att_cut)
    left, right = st.columns(2)
    with left:
        st.subheader("Probability")
        st.dataframe(pd.Series(prob["simple"], name="probability").round(3).to_frame(),
                     width='stretch')
    with right:
        st.subheader("Conditional probability")
        st.dataframe(pd.Series(prob["conditional"], name="probability").round(3).to_frame(),
                     width='stretch')
    ind = prob["independence"]
    verdict(not ind["independent?"],
            f"P(A and B) = {ind['P(distinction AND good attendance)']:.3f} vs "
            f"P(A) x P(B) = {ind['P(distinction) x P(good attendance)']:.3f}. "
            + ("They are close, so distinction and good attendance look independent."
               if ind["independent?"] else
               "They differ, so scoring a distinction and attending well are not independent."))

# ------------------------------ outliers ------------------------------
with tabs[3]:
    st.subheader("Outliers with the IQR method")
    fences, rows = spa.iqr_outliers(df)
    st.dataframe(fences, width='stretch')
    fig, axes = plt.subplots(1, len(spa.COLS), figsize=(12, 3.2))
    for ax, c in zip(axes, spa.COLS):
        sns.boxplot(y=df[c], ax=ax, color=MIST, flierprops={"markerfacecolor": AMBER,
                                                          "markeredgecolor": AMBER})
        ax.set_ylabel("")
        ax.set_title(c.replace("_", " "), fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(f"**{len(rows)} students** are outliers in at least one column:")
    st.dataframe(rows, width='stretch')

# ------------------------------ correlation ------------------------------
with tabs[4]:
    st.subheader("Correlation")
    matrix, tbl = spa.correlation(df)
    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, cbar=False)
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
    with right:
        st.dataframe(tbl.style.format({"pearson_r": "{:.3f}", "p_value": "{:.4f}"}),
                     width='stretch')
        reg = spa.regression(df)
        st.caption(f"Regression (all four factors together): R² = {reg['r2']}. "
                   "Coefficients are marks gained per one-unit increase, others held fixed.")
        st.dataframe(reg["coefficients"].rename("coefficient").to_frame(), width='stretch')
    feat = st.selectbox("Scatter plot", spa.FEATURES)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sns.regplot(data=df, x=feat, y=spa.TARGET, ax=ax, scatter_kws={"alpha": 0.5, "color": INDIGO},
                line_kws={"color": AMBER})
    st.pyplot(fig)

# ------------------------------ hypothesis ------------------------------
with tabs[5]:
    st.subheader("Hypothesis testing")
    res = spa.hypothesis_tests(df, benchmark, alpha, att_cut)
    a = res["one_sample"]
    st.markdown(f"**1. One-sample t-test.** H0: {a['H0']}. H1: {a['H1']}.")
    st.write(f"Sample mean {a['sample_mean']:.2f}, t = {a['t_stat']:.2f}, {spa.fmt_p(a['p_value'])}")
    verdict(a["reject_H0"], f"Reject H0: the mean mark differs from {benchmark}." if a["reject_H0"]
            else f"Fail to reject H0: no evidence the mean differs from {benchmark}.")
    b = res["two_sample"]
    st.markdown(f"**2. Welch two-sample t-test.** H0: {b['H0']}. H1: {b['H1']}.")
    st.write(f"Mean {b['mean_high_attendance']:.2f} vs {b['mean_low_attendance']:.2f}, "
             f"t = {b['t_stat']:.2f}, {spa.fmt_p(b['p_value'])}")
    verdict(b["reject_H0"], "Reject H0: attendance groups score differently." if b["reject_H0"]
            else "Fail to reject H0: no evidence the groups differ.")
    c = res["correlation"]
    st.markdown(f"**3. Correlation test.** H0: {c['H0']}. H1: {c['H1']}.")
    st.write(f"r = {c['r']:.3f}, {spa.fmt_p(c['p_value'])}")
    verdict(c["reject_H0"], "Reject H0: study hours and marks are linearly related." if c["reject_H0"]
            else "Fail to reject H0: no evidence of a linear relationship.")

# ------------------------------ recommendations ------------------------------
with tabs[6]:
    st.subheader("Business recommendations")
    for i, rec in enumerate(spa.recommendations(df, benchmark, alpha), 1):
        st.markdown(f"{i}. {rec}")
