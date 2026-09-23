# Numerical reproduction and historical differences

Completed 10–11 September 2026; release 11 September 2026. Two separate authorized raw-data runs included MIMIC reconstruction, 12 source fits and 2,000-replicate patient bootstrap, and eICU reconstruction, all three existing event variants, refits/recalibration and 1,000-replicate hospital bootstrap. The previously reported overlap sensitivity was independently refitted twice. Private execution logs, matrices and binaries remain locally retained.

| Verification | Executed comparison | Result |
|---|---|---|
| MIMIC reconstruction and retraining | 233 field-level checks | All matched |
| eICU reconstruction and retraining | 178 field-level checks | All matched |
| Existing overlap sensitivity | 17 field-level checks | All matched |
| Independent JSON prediction versus fitted estimators | 36 current fitted/recalibration specifications | All matched within 1e-10 |
| Actual pre-audit historical models on original derived inputs | 8 checks across 4 models | Matched within 1e-10; not historical raw retraining |
| Historical MIMIC report versus corrected results | 630 numeric fields | 464 changed |
| Historical eICU report versus corrected results | 2,424 numeric fields | 1,512 changed |

The first three rows total 428 field checks, not models or participants. Identity and label fields require exact agreement; numerical prediction/point tolerances are 1e-10 and bootstrap-field tolerance is 1e-8. Per-field differences and judgements are in the comparison CSVs. Fixed-model prediction is separate from retraining. Hashes identify files and are not substitutes for these numerical tests.

## Corrections and impact

The original tied-record selection and training-row order were not deterministic. The author authorized retaining existing source priorities and drug deduplication, resolving remaining ties by original CSV record order, and sorting training records by stable keys. Clinical thresholds, endpoints, partitions, feature sets and hyperparameters were not retuned. A bounded scanner assigns original ordinals before filtering and avoids the earlier full-table memory failure. Failed attempts and original models remain retained.

Terminal infusion ties included conflicting Stopped and FinishedRunning statuses. Applying the unchanged explicit-stopped predicate changed three candidate memberships: two removed and one added. Candidates changed from 11,940 to 11,939. The primary cohort remains 9,279 records (8,590 development; 689 temporal validation), with one death-first development record removed and another added. Common-record labels and partitions did not change. eICU retains 2,104 locked-evaluation records: 734 restart-first, 56 administrative-death-proxy-first and 1,314 no-recorded-event.

Current primary MIMIC dynamic logistic restart AUROC is 0.695012. Current eICU source AUROC is 0.504509, Brier 0.519695 and mean predicted risk 0.825217. Dose/agent-feature ablation gives AUROC 0.504347 and Brier 0.237782: probability error improves without restored discrimination. Target-native refitting gives AUROC 0.632450 and Brier 0.215574. All affected estimates and intervals were updated rather than forced to match historical numbers.

The former manuscript's -5.675 “update slope” was an evaluation calibration slope, not the fitted recalibration coefficient. The corrected fitted update coefficient is -0.0081726686, while evaluation calibration slope is -5.732373. The manuscript now distinguishes them. This is an interpretation correction, not model retuning.

Historical benchmark rows remain archival. The same-record label analysis applies genuine archived parameters to deterministic features, giving AUROC 0.703625 on 689 common records. Both label versions have 320 restart events and identical metrics in this comparison. This does not mean historical-model predictions on the original features were unchanged.

A final portability repair replaced stage 17's hard-coded temporary drive with the private output directory. Actual stage-17 re-execution matched all 11,939 rows and 51 columns exactly. Unused ancestor-directory variables and personal-path defaults were also removed. The convenience wrapper is a new entry point to already-executed modules; independent runs used the stage commands. Another operating system or external machine was not independently tested.

## Evidence

- MIMIC_INDEPENDENT_RUN_COMPARISON.csv, EICU_INDEPENDENT_RUN_COMPARISON.csv, OVERLAP_INDEPENDENT_RUN_COMPARISON.csv: actual independent-run checks.
- CURRENT_FIXED_MODEL_JSON_REPLAY.csv: current fixed-model probability checks.
- MIMIC_REPORTED_VS_DETERMINISTIC_RESULTS.csv and EICU_REPORTED_VS_DETERMINISTIC_RESULTS.csv: original value—reproduced value—difference—judgement.
- MIMIC_PRIMARY_COHORT_VERSION_COMPARISON.json and MIMIC_TERMINAL_TIE_MEMBERSHIP_EXPLANATION.csv: aggregate membership changes without identifiers.
- MIMIC_DETERMINISTIC_VS_HISTORICAL_FEATURE_AUDIT.csv: historical feature differences.
- HISTORICAL_PRE_AUDIT_FIXED_PREDICTION_REPLAY.csv: separately scoped historical-model verification.
- current_aggregates/: current manuscript, supplement and figure results.

Post hoc status, observation selection, the administrative death proxy, cross-partition patients and conditional bootstrap uncertainty remain limitations. Computational agreement does not establish clinical utility, causality or decision thresholds. No patient-level data, predictions or joblib objects are included publicly.
