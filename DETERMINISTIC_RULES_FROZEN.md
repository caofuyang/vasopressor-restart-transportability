# Deterministic implementation correction — 2026-09-10

Authorized by the author before this run. Historical models, parameters and results are preserved separately; they are not renamed as corrected models.

1. Original CSV data-record ordinal (header excluded; 1-based) resolves otherwise tied records, after existing explicit clinical source priorities. The ordinal is assigned before filtering or joins using a serial CSV reader.
2. SQL and numerical libraries execute with one thread; insertion order is preserved where configured. No rounding is introduced to force agreement.
3. Model input frames are sorted by subject_id, hadm_id, stay_id (MIMIC) and patienthealthsystemstayid (eICU; the analysis record is a hospital stay), with stable sorting, before fitting and evaluation. Stable record keys are checked for uniqueness. Existing partition membership, endpoint definitions, feature sets, hyperparameters, random seeds and bootstrap cluster units remain unchanged.
4. Numerical verification uses absolute tolerance 1e-10 for probabilities/point metrics and 1e-8 for bootstrap result fields, with exact cohort membership and labels. Deviations are reported, never hidden by changing tolerance after results.
5. New results are provisional until a second independent execution verifies them. Fixed-model prediction replay and retraining replay are separate checks.

Implementation note: raw extraction disables global insertion-order buffering to stay within memory, while assigning the ordinal inside the serial CSV reader and explicitly ordering the filtered extraction by that ordinal. This changes execution strategy only. Two failed extraction attempts are retained in private execution logs.

Large MIMIC chart/laboratory CSVs are now scanned in bounded Arrow batches with explicit numeric/timestamp types. Record ordinals are assigned before any item/admission filter. Existing SQL applies the original time windows and cleaning rules to these filtered raw tables. This reader substitution is subject to the same feature-level comparison; it is not treated as validated merely because the code executes.
