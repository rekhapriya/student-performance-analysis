"""
Student Performance Analysis - Statistics & Probability Assignment
-------------------------------------------------------------------
A college wants to know whether study hours, attendance, assignment scores
and sleep hours influence students' final marks.

Flow: Data Understanding -> Descriptive Statistics -> Probability ->
      Conditional Probability -> IQR/Outliers -> Correlation ->
      Hypothesis Testing -> Business Recommendations

Run:  python student_performance_analysis.py
"""

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = ["study_hours", "attendance", "assignment_score", "sleep_hours"]
TARGET = "final_marks"
COLS = FEATURES + [TARGET]


# ----------------------------------------------------------------------
# 1. DATA UNDERSTANDING
# ----------------------------------------------------------------------
def generate_data(n=300, seed=42):
    """Create a realistic synthetic student dataset (seeded, so results repeat)."""
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(4.0, 1.5, n), 0.5, 10)
    attendance = np.clip(rng.normal(78, 12, n), 35, 100)
    assignment = np.clip(rng.normal(72, 12, n) + 1.5 * (study - 4), 20, 100)
    sleep = np.clip(rng.normal(6.8, 1.1, n), 3.5, 9.5)

    marks = (
        12
        + 4.0 * study
        + 0.25 * attendance
        + 0.25 * assignment
        + 1.2 * (sleep - 7)
        + rng.normal(0, 6, n)
    )
    df = pd.DataFrame(
        {
            "study_hours": study.round(1),
            "attendance": attendance.round(1),
            "assignment_score": assignment.round(1),
            "sleep_hours": sleep.round(1),
            "final_marks": np.clip(marks, 0, 100).round(1),
        }
    )
    # A few unusual students so outlier detection has something real to find
    df.loc[[10, 77, 150], "final_marks"] = [18.0, 22.0, 15.0]
    df.loc[[40, 200], "study_hours"] = [9.8, 9.5]
    df.insert(0, "student_id", [f"S{i:03d}" for i in range(1, n + 1)])
    return df


def data_overview(df):
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


# ----------------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# ----------------------------------------------------------------------
def descriptive_stats(df):
    out = pd.DataFrame(
        {
            "mean": df[COLS].mean(),
            "median": df[COLS].median(),
            "mode": df[COLS].mode().iloc[0],
            "std": df[COLS].std(),
            "variance": df[COLS].var(),
            "min": df[COLS].min(),
            "max": df[COLS].max(),
            "skewness": df[COLS].skew(),
            "kurtosis": df[COLS].kurt(),
        }
    )
    return out.round(2)


# ----------------------------------------------------------------------
# 3 & 4. PROBABILITY AND CONDITIONAL PROBABILITY
# ----------------------------------------------------------------------
def probabilities(df, distinction=75, pass_mark=40, att_cut=75, study_cut=5, sleep_cut=6):
    n = len(df)
    high = df[TARGET] >= distinction
    good_att = df["attendance"] >= att_cut
    long_study = df["study_hours"] >= study_cut
    good_sleep = df["sleep_hours"] >= sleep_cut

    def cond(event, given):
        return float(event[given].mean()) if given.sum() else float("nan")

    p_high, p_att = high.mean(), good_att.mean()
    simple = {
        f"P(final >= {distinction})": p_high,
        f"P(final < {pass_mark}) - fail": float((df[TARGET] < pass_mark).mean()),
        f"P(attendance >= {att_cut}%)": p_att,
        f"P(study >= {study_cut} hrs)": float(long_study.mean()),
        f"P(sleep >= {sleep_cut} hrs)": float(good_sleep.mean()),
    }
    conditional = {
        f"P(final >= {distinction} | attendance >= {att_cut}%)": cond(high, good_att),
        f"P(final >= {distinction} | attendance < {att_cut}%)": cond(high, ~good_att),
        f"P(final >= {distinction} | study >= {study_cut} hrs)": cond(high, long_study),
        f"P(final >= {distinction} | study < {study_cut} hrs)": cond(high, ~long_study),
        f"P(final >= {distinction} | sleep >= {sleep_cut} hrs)": cond(high, good_sleep),
        f"P(final >= {distinction} | sleep < {sleep_cut} hrs)": cond(high, ~good_sleep),
    }
    # Independence check: if independent, P(A and B) == P(A) * P(B)
    joint = float((high & good_att).mean())
    independence = {
        "P(distinction AND good attendance)": joint,
        "P(distinction) x P(good attendance)": float(p_high * p_att),
        "independent?": bool(abs(joint - p_high * p_att) < 0.01),
    }
    return {"simple": simple, "conditional": conditional, "independence": independence, "n": n}


# ----------------------------------------------------------------------
# 5. OUTLIERS (IQR METHOD)
# ----------------------------------------------------------------------
def iqr_outliers(df, k=1.5):
    rows, flagged = [], pd.Series(False, index=df.index)
    for col in COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (df[col] < lo) | (df[col] > hi)
        flagged |= mask
        rows.append(
            {"column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
             "lower_fence": lo, "upper_fence": hi, "outliers": int(mask.sum())}
        )
    return pd.DataFrame(rows).round(2).set_index("column"), df[flagged]


# ----------------------------------------------------------------------
# 6. CORRELATION
# ----------------------------------------------------------------------
def correlation(df):
    matrix = df[COLS].corr()
    rows = []
    for col in FEATURES:
        r, p = stats.pearsonr(df[col], df[TARGET])
        rows.append({"feature": col, "pearson_r": r, "p_value": p,
                     "strength": strength_label(r)})
    return matrix.round(3), pd.DataFrame(rows).set_index("feature")


def fmt_p(p):
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


def strength_label(r):
    a = abs(r)
    word = "very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6 else "strong"
    return f"{word} {'positive' if r > 0 else 'negative'}"


def regression(df):
    """Multiple linear regression with NumPy: how much does each factor add?"""
    X = np.column_stack([np.ones(len(df))] + [df[c] for c in FEATURES])
    y = df[TARGET].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    coefs = pd.Series(beta[1:], index=FEATURES).round(3)
    return {"intercept": round(float(beta[0]), 3), "coefficients": coefs, "r2": round(float(r2), 3)}


# ----------------------------------------------------------------------
# 7. HYPOTHESIS TESTING
# ----------------------------------------------------------------------
def hypothesis_tests(df, benchmark=60, alpha=0.05, att_cut=75):
    results = {}

    # Test 1: one-sample t-test - does the average mark differ from the benchmark?
    t, p = stats.ttest_1samp(df[TARGET], benchmark)
    results["one_sample"] = {
        "H0": f"Mean final marks = {benchmark}",
        "H1": f"Mean final marks != {benchmark}",
        "sample_mean": df[TARGET].mean(), "t_stat": t, "p_value": p,
        "reject_H0": bool(p < alpha),
    }

    # Test 2: two-sample (Welch) t-test - do high-attendance students score differently?
    hi = df.loc[df["attendance"] >= att_cut, TARGET]
    lo = df.loc[df["attendance"] < att_cut, TARGET]
    t2, p2 = stats.ttest_ind(hi, lo, equal_var=False)
    results["two_sample"] = {
        "H0": f"Mean marks equal for attendance >= {att_cut}% and < {att_cut}%",
        "H1": "Mean marks differ between the two groups",
        "mean_high_attendance": hi.mean(), "mean_low_attendance": lo.mean(),
        "t_stat": t2, "p_value": p2, "reject_H0": bool(p2 < alpha),
    }

    # Test 3: is the correlation between study hours and marks real or chance?
    r, p3 = stats.pearsonr(df["study_hours"], df[TARGET])
    results["correlation"] = {
        "H0": "No linear relationship between study hours and final marks (r = 0)",
        "H1": "A linear relationship exists (r != 0)",
        "r": r, "p_value": p3, "reject_H0": bool(p3 < alpha),
    }
    results["alpha"] = alpha
    return results


# ----------------------------------------------------------------------
# 8. BUSINESS RECOMMENDATIONS (built from the results above)
# ----------------------------------------------------------------------
def recommendations(df, benchmark=60, alpha=0.05):
    _, corr_tbl = correlation(df)
    prob = probabilities(df)
    tests = hypothesis_tests(df, benchmark, alpha)
    reg = regression(df)
    cond = prob["conditional"]
    keys = list(cond.keys())

    strongest = corr_tbl["pearson_r"].abs().idxmax()
    recs = [
        f"Focus first on {strongest.replace('_', ' ')} - it has the strongest link with final "
        f"marks (r = {corr_tbl.loc[strongest, 'pearson_r']:.2f}).",
        f"Attendance matters: the chance of scoring 75+ is {cond[keys[0]]:.0%} for students "
        f"with 75%+ attendance versus {cond[keys[1]]:.0%} for the rest. Set up early alerts "
        f"for students whose attendance drops below 75%.",
        f"Study time pays off: {cond[keys[2]]:.0%} of students studying 5+ hours a day reach "
        f"75+, against {cond[keys[3]]:.0%} of those studying less. Offer guided study "
        f"sessions and time-management workshops.",
        f"Sleep is a quieter factor (each extra hour is linked to about "
        f"{reg['coefficients']['sleep_hours']:.1f} marks, holding the others fixed). Include "
        f"sleep and wellbeing tips in exam-season advice.",
        "Review the IQR outliers one by one: students with very low marks may need "
        "counselling or learning support, not just more study time.",
        f"The four factors together explain about {reg['r2']:.0%} of the variation in marks, "
        f"so also collect data on teaching quality, prior grades and stress for a fuller picture.",
    ]
    verdict = tests["one_sample"]
    recs.insert(
        0,
        f"The average mark ({verdict['sample_mean']:.1f}) "
        f"{'differs significantly from' if verdict['reject_H0'] else 'is not significantly different from'} "
        f"the benchmark of {benchmark} ({fmt_p(verdict['p_value'])}).",
    )
    return recs


# ----------------------------------------------------------------------
# RUN EVERYTHING
# ----------------------------------------------------------------------
def main():
    pd.set_option("display.width", 120)
    df = generate_data()

    print("\n=== 1. DATA UNDERSTANDING ===")
    print(df.head())
    print(data_overview(df))

    print("\n=== 2. DESCRIPTIVE STATISTICS ===")
    print(descriptive_stats(df))

    prob = probabilities(df)
    print("\n=== 3. PROBABILITY ===")
    for k, v in prob["simple"].items():
        print(f"{k}: {v:.3f}")

    print("\n=== 4. CONDITIONAL PROBABILITY ===")
    for k, v in prob["conditional"].items():
        print(f"{k}: {v:.3f}")
    print(prob["independence"])

    fences, outlier_rows = iqr_outliers(df)
    print("\n=== 5. OUTLIERS (IQR) ===")
    print(fences)
    print(f"\nStudents flagged in at least one column: {len(outlier_rows)}")
    print(outlier_rows)

    matrix, corr_tbl = correlation(df)
    print("\n=== 6. CORRELATION ===")
    print(matrix)
    print(corr_tbl.round(4))
    print("\nRegression:", regression(df))

    print("\n=== 7. HYPOTHESIS TESTING (alpha = 0.05) ===")
    for name, res in hypothesis_tests(df).items():
        if isinstance(res, dict):
            print(f"\n{name}:")
            for k, v in res.items():
                print(f"  {k}: {fmt_p(v) if k == 'p_value' else round(v, 4) if isinstance(v, float) else v}")

    print("\n=== 8. BUSINESS RECOMMENDATIONS ===")
    for i, rec in enumerate(recommendations(df), 1):
        print(f"{i}. {rec}")


if __name__ == "__main__":
    main()
