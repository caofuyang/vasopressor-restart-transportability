# Post-hoc reporting rules frozen before new expanded calculations

Date frozen: 2026-09-09  
Purpose: enrich reporting without changing any endpoint rule, partition, feature set, model, fitted parameter, or primary analysis.

## General

- All analyses below are explicitly post hoc and exploratory or sensitivity analyses.
- Previously fitted predictions are used without refitting or tuning.
- Negative or null findings are retained.
- No hospital, subgroup, threshold, feature set, or comparison will be selected according to observed performance.
- Patient-level inputs and predictions remain in the restricted local output directory. Only aggregate results leave that directory.

## Same-record MIMIC label sensitivity

- Use the intersection of the original MIMIC temporal prediction file and the final scope-corrected cohort.
- Restrict to records included in the final corrected primary temporal-validation population.
- Keep the original dynamic multinomial logistic restart probability fixed.
- Compare restart-first binary metrics against the originally generated outcome and the final corrected outcome on exactly the same records.
- Use 2,000 patient-cluster bootstrap resamples with seed 20260825. Each sampled patient contributes all eligible records.
- Report AUROC, AUPRC, Brier score, event rate, and paired corrected-minus-original metric differences.
- This analysis measures label sensitivity under fixed old predictions. It does not isolate the effect of endpoint correction on a newly fitted model, and neither label is treated as a clinical gold standard.

## Hospital heterogeneity

- Include all 37 hospitals in the frozen eICU locked-evaluation partition.
- Public results use deterministic anonymized hospital labels; the database hospital identifiers remain restricted.
- Report N, restart-first events, event rate, source-model and target-native mean predicted risk, Brier score, and observed-minus-predicted risk gap for every hospital.
- Report hospital-specific AUROC only when both the event and non-event counts are at least 10. Otherwise mark AUROC as not estimable under the reporting rule.
- Hospitals are ordered by sample size, not by AUROC or a favorable model result. No hospital-level performance ranking is interpreted.
- Quantify concentration using the share of records and events in the largest and five largest hospitals, the range of leave-one-hospital-out overall metrics, and each hospital's contribution to aggregate excess squared error. These are descriptive checks of whether overall transport results are dominated by a few hospitals; they are not causal analyses.

## Observability-selection differences

- Compare the final primary-analysis population with all excluded or indeterminate fixed candidates separately in MIMIC-IV and eICU.
- Continuous baseline variables use the absolute standardized mean difference based on complete values and pooled standard deviations. Binary variables use the standard difference in proportions. Missingness is reported separately and is not imputed for this comparison.
- The common high-priority baseline variables are age, female sex, episode duration, last and 6-hour mean MAP, last and 6-hour mean heart rate, last respiratory rate, last temperature, last and 6-hour mean norepinephrine-equivalent dose, number of drugs, and modified shock index when available.
- Post-index administrative observation time is summarized separately, capped at 24 hours, and is never treated as a prediction feature. For MIMIC-IV this is hospital-discharge coverage from the index; for eICU it is the corrected continuous administrative observation interval.
- Unknown outcomes remain unknown. No inverse-probability weighting or outcome imputation is introduced.

## Stopping rule

The additions stop when they answer label sensitivity, model updating, hospital concentration, and observability selection. No new algorithm family, threshold optimization, or performance-selected subgroup will be added.
