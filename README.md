# Student Performance Analysis

A statistics and probability project: do **study hours, attendance, assignment scores and sleep** influence students' final marks?

Built with Python, Pandas, NumPy, SciPy, Seaborn and Streamlit.

## What it covers

| Step | What I did |
|---|---|
| Data understanding | 300 synthetic students (seeded), shape, missing values, duplicates |
| Descriptive statistics | Mean, median, mode, std, variance, skewness, kurtosis |
| Probability | P(distinction), P(good attendance), independence check |
| Conditional probability | P(distinction given attendance, study hours or sleep) |
| Outliers | IQR fences and box plots |
| Correlation | Pearson r with p-values, heatmap, multiple regression (R²) |
| Hypothesis testing | One-sample t-test vs benchmark, Welch t-test (attendance groups), correlation test |
| Recommendations | Generated from the results, not hard-coded |

## Run it

```bash
pip install -r requirements.txt
python student_performance_analysis.py   # full report in the terminal
streamlit run app.py                     # interactive dashboard
```

You can also upload your own CSV in the dashboard. It needs these columns: `study_hours, attendance, assignment_score, sleep_hours, final_marks`.

## Key findings (sample data, seed 42)

- Study hours have the strongest link with final marks (r ≈ 0.53).
- Students with 75%+ attendance are far more likely to score 75+ than those below 75%.
- The average mark differs significantly from the benchmark of 60 (one-sample t-test, p < 0.0001).
- The four factors together explain about 45% of the variation in marks.

The data is synthetic, so the numbers show the method, not real students.

## Interview prep

- **Why Welch's t-test?** It does not assume equal variances in the two groups.
- **What does a p-value mean?** The probability of seeing a result at least this extreme if H0 were true. It is not the probability that H0 is true.
- **Correlation vs causation?** Study hours and marks move together, but this data cannot prove one causes the other.
- **Why IQR for outliers?** It does not depend on the mean or standard deviation, which outliers distort.
- **Independence check:** if A and B are independent, P(A and B) = P(A) x P(B).
- **Why seed the random generator?** So results are reproducible.
