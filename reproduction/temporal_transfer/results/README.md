# Short-record donor transfer

Feature windows are 1981 onward (3yr: 1981-1983, 5yr: 1981-1985, 10yr: 1981-1990, 5yr_shifted: 1986-1990); future labels are 2001-2010. The donor/calibration/test split is a deterministic 60/20/20 partition of unique physical `site_id` values. A record enters a retrieval window when at least 68% of expected daily discharge observations are finite. Flow, precipitation and temperature coverage in the evaluation period is a prespecified held-out-label availability gate for the evaluation cohort; Donor labels in the evaluation period supply analog predictions and label-normalizing IQRs, and calibration labels in the evaluation period select mixture weights. test-target labels in the evaluation period are used only for the availability gate and final scoring. Early discharge supplies similarity features and donor-distance scaling.

The early-feature cohort and final evaluable cohort are 2279→2273 (3 yr), 2325→2319 (5 yr), 2422→2416 (10 yr) and 2510→2504 (shifted 5 yr). The final cohort is the reported estimand because all four future labels are required.

Distances are annual log1p mean-normalized FDC, monthly conditional W2 (12 x 32 quantiles), and monthly allocation. Distances are normalized by donor-pool medians. Candidate weights mix each distance with annual FDC and are selected from [0.0, 0.25, 0.5, 0.75, 1.0] using calibration labels only. Predictions are medians of K=10 donors. `summary.json` includes 5,000-replicate physical-site bootstrap intervals, and `per_target_errors_<window>.csv` contains target-level donor-IQR-normalized errors for inspection.

The W2–allocation paired interval is recomputed directly from per-target aggregate error differences with seed 20263107 and 5,000 replicates.
