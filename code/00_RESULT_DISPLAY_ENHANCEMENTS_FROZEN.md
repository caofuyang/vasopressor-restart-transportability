# Result-display enhancement rules (frozen before calculation)

Status: descriptive/post-audit presentation enhancement; it does not alter the frozen endpoint, cohort, partitions, features, hyperparameters, or primary model.

1. Calibration display uses ten equal-frequency groups in the eICU locked-evaluation set. Source-only and target-native full models are shown against the identity line. Risk distributions use fixed 0.05-wide bins from 0 to 1.
2. The fixed-risk descriptive comparator predicts the restart-first prevalence observed in the eICU adaptation-development partition for every locked-evaluation record. It is not selected against locked-evaluation performance and is not a fitted clinical model.
3. Paired differences use the same 1,000 hospital-cluster bootstrap draws and original eICU bootstrap seed. Positive AUROC/AUPRC differences favor the named first model; positive Brier improvement means lower Brier score for the named first model.
4. Direct comparisons include target-native full versus source-only, target-native full versus harmonized last-values, and target-native dynamic-vitals versus harmonized last-values. All results are reported regardless of direction.
5. Hospital heterogeneity is reported descriptively using aggregate distributions across contributing hospitals. No hospital is selected, excluded, or identified by its database identifier on the basis of performance.
6. Patient-level predictions remain restricted and are not included in the submission or shareable aggregate package.
