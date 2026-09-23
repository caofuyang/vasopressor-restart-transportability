# Vasopressor cessation and transportability reproduction package

Release 11 September 2026; deterministic implementation version 10 September 2026. Two independent raw reconstructions and refits were completed. See REPRODUCTION_REPORT.md and numerical comparison CSVs for evidence. Agreement between corrected runs does not mean the historical results were reproduced unchanged.

## Data and access

Obtain authorized access to MIMIC-IV 3.1 and eICU-CRD 2.0 separately. This package does not include their patient data. Supply the root containing `icu/` and `hosp/` for MIMIC, and the directory containing `patient.csv.gz`, `infusionDrug.csv.gz`, `vitalPeriodic.csv.gz`, `vitalAperiodic.csv.gz`, and `carePlanEOL.csv.gz` for eICU. All intermediates, fitted binary artifacts and private logs are written to a new user-selected run directory. Never distribute that run directory.

## Execution order

Use Python 3.12.14 and install `requirements.txt` in a dedicated environment. ANALYSIS_ENVIRONMENT.json records Windows 11 and numerical versions; other platforms have not been independently tested. Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before launching individual stage commands. The complete `run_all.py` wrapper sets these automatically before subprocesses. The following placeholders must be replaced with authorized data and new private output paths:

```text
python code/run_mimic_raw_stages.py --raw MIMIC_RAW_ROOT --run-dir PRIVATE_MIMIC_RUN_A
python code/train_mimic.py --run-dir PRIVATE_MIMIC_RUN_A
python code/run_mimic_raw_stages.py --raw MIMIC_RAW_ROOT --run-dir PRIVATE_MIMIC_RUN_B
python code/train_mimic.py --run-dir PRIVATE_MIMIC_RUN_B
python code/compare_independent_runs.py --first PRIVATE_MIMIC_RUN_A --second PRIVATE_MIMIC_RUN_B --output AGGREGATE_MIMIC_COMPARISON.csv
python code/run_eicu.py --raw EICU_RAW_ROOT --run-dir PRIVATE_EICU_RUN_A --parameters PRIVATE_MIMIC_RUN_A/models/parameters
python code/export_eicu.py --run-dir PRIVATE_EICU_RUN_A --source-parameters PRIVATE_MIMIC_RUN_A/models/parameters
```

Repeat the eICU commands in a second independent directory and compare with `compare_eicu_runs.py`. Complete execution, including all existing sensitivity analyses, aggregate tables and six figures, is available through:

```text
python -m pip install -r requirements.txt
python code/run_all.py --mimic-raw MIMIC_RAW_ROOT --eicu-raw EICU_RAW_ROOT --output PRIVATE_RUN_A
python code/run_all.py --mimic-raw MIMIC_RAW_ROOT --eicu-raw EICU_RAW_ROOT --output PRIVATE_RUN_B
python code/compare_independent_runs.py --first PRIVATE_RUN_A/mimic --second PRIVATE_RUN_B/mimic --output MIMIC_COMPARISON.csv
python code/compare_eicu_runs.py --first PRIVATE_RUN_A/eicu --second PRIVATE_RUN_B/eicu --source-parameters PRIVATE_RUN_A/mimic/models/parameters --output EICU_COMPARISON.csv
python code/compare_overlap_runs.py --first PRIVATE_RUN_A/overlap --second PRIVATE_RUN_B/overlap --eicu-run PRIVATE_RUN_A/eicu --source PRIVATE_RUN_A/mimic/models/parameters/mimic_primary__dynamic_multinomial_logistic.json --output OVERLAP_COMPARISON.csv
```

The wrapper assembles the same stage commands used in the completed independent runs; it was added after those runs. Evidence comes from actual stage executions, not a claim that the later wrapper was used for both runs. New output directories are required; failures preserve private logs and do not overwrite history. Each MIMIC run streams 432,997,491 chartevents records. Allow sufficient disk space for raw-derived intermediates and temporary files. Do not redistribute private run directories.

`historical_parameters/` contains the actual previous fitted specifications, explicitly retained as historical. The training command reads their feature definitions and architecture/hyperparameters to construct **unfitted** estimators. It learns all preprocessing statistics and coefficients anew from the regenerated development data. Newly fitted readable specifications go to each private run's `models/parameters/`; eICU specifications go to its `parameters/`.

## Verification scope

- Exact patient-record membership, endpoint labels and frozen partitions are compared privately; only aggregate discrepancy counts are released.
- Fixed-model replay compares executable JSON predictions against fitted estimator predictions on the same evaluation records.
- Retraining replay compares separately fitted models from independent raw reconstructions, not just hashes, imported modules, or serialized binaries.
- Historical-versus-corrected result differences must remain visible even when independent runs of the corrected implementation agree.
- Bootstrap intervals remain conditional on fitted models. Passing computational reproducibility checks does not establish clinical usefulness or causal effects.

## Current parameters and independent prediction

`parameters/` supplies 36 current fitted/recalibration JSON files, including 12 MIMIC, 21 eICU and 3 from the already reported overlap-exclusion sensitivity. CURRENT_MODEL_INVENTORY.csv maps roles and training code. JSON contains ordered inputs, categories, imputation statistics, scaling, coefficients/intercepts, class order, or full boosting trees and baseline predictions. Recalibration files contain probability clipping and fitted intercept/slope. `historical_pre_audit_parameters/` contains four genuine earlier fitted models, separate from the 33 pre-deterministic snapshots. None is renamed as current.

```text
python code/predict_from_json.py --input PRIVATE_FEATURES.parquet --model parameters/mimic_primary__dynamic_multinomial_logistic.json --output PRIVATE_PROBABILITIES.csv
python code/predict_from_json.py --input PRIVATE_EICU_FEATURES.parquet --model parameters/mimic_primary__dynamic_multinomial_logistic.json --recalibration parameters/primary_traceable__primary_logistic.json --output PRIVATE_UPDATED_PROBABILITIES.csv
```

CSV input is also supported. Input rows must retain the pipeline's raw feature values, units, encodings and missing values; do not standardize them manually. Output follows input row order and must remain private. `portable_predict.py` uses numpy/pandas without loading fitted joblib or sklearn estimators. Class order is explicit in `estimator.classes`; class 1 means restart-first. The recalibration update coefficient is distinct from evaluation calibration slope. RANDOM_SEED_LOCATIONS.csv and JSON estimator settings retain original seeds, including null random_state where specified; no seeds were selected to obtain desired metrics.

## Raw-to-result code map

| Process | Executable source |
|---|---|
| MIMIC drug identification, episodes, eligibility and unit conversion | `code/mimic/stage01.py` through `stage06.py`; original SQL rules and norepinephrine-equivalent calculations |
| MIMIC time alignment and features | `stage07.py` and `stream_raw.py`; raw vitals/laboratories, source priorities and original time windows; see FEATURE_DICTIONARY.csv |
| MIMIC observation, endpoints and frozen partitions | stages 11, 12, 13, 17, 18, 19_post_audit and 20_scope_corrected; stage runner defines order |
| Source training/evaluation | `train_mimic.py`, `mimic_evaluation.py`, `model_from_json.py`; 12 fits and 2,000 patient-bootstrap replicates |
| eICU drugs, stop/restart and features | `code/eicu/eicu_stage1b_explicit_stop_cohort.py`, `eicu_stage2_external_features.py`; patient, infusionDrug, vitalPeriodic, vitalAperiodic and carePlanEOL |
| eICU observation, labels, frozen hospital partitions and evaluation | `code/eicu_correction/01_correct_cohort.py`, `03_evaluate.py`, frozen manifests and `export_eicu.py`; three existing variants and 1,000 hospital-bootstrap replicates |
| Existing cross-partition-patient exclusion sensitivity | `replay_existing_overlap_sensitivity.py`; original 43-patient exclusion, fixed hospital assignments and existing fitting specifications |
| Tables and figures | `build_descriptive_aggregates.py`, `build_result_displays.py`, `build_clinical_results.py`, `prepare_historical_comparison_inputs.py`, `replay_existing_posthoc.py`, `publication_figures.py` |

No unprovided personal script or pre-existing derived matrix is required by the current raw-data pipeline. `historical_reference_aggregates/` provides explicitly archival benchmark rows. Same-record label comparisons apply genuine archived parameters to deterministic features, then hold probabilities fixed between labels; they do not assert unchanged historical features. `current_aggregates/` contains current manuscript results. Data access requires PhysioNet credentialing: https://physionet.org/content/mimiciv/3.1/ and https://physionet.org/content/eicu-crd/2.0/ . This repository is released under the MIT License (see the LICENSE file). To cite this software, see CITATION.cff; a citable archive with a persistent DOI is deposited on Zenodo and linked from this repository's Releases. The associated manuscript is: "Recorded vasopressor restart after cessation: timing, endpoint ascertainment, and cross-database model transportability in MIMIC-IV and eICU".
