from __future__ import annotations
import os
import json
from pathlib import Path
import joblib
from model_from_json import source_bundle, FrozenJSONModel
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
OUT = Path(os.environ['AGGREGATE_OUTPUT'])
OUT.mkdir(parents=True, exist_ok=True)
MIMIC_COHORT = Path(os.environ['MIMIC_RUN']) / 'scope_corrected/restricted/stage18_post_audit_scope_corrected_cohort.parquet'
MIMIC_FEATURES = Path(os.environ['MIMIC_RUN']) / 'derived/vasopressor_liberation_features_stage7.parquet'
MIMIC_MODELS = Path(os.environ['MIMIC_RUN']) / 'models/parameters' / 'mimic_primary__dynamic_multinomial_logistic.json'
MIMIC_POINTS = Path(os.environ['MIMIC_RUN']) / 'models/results/mimic_primary_main_temporal_results.csv'
MIMIC_CI = Path(os.environ['MIMIC_RUN']) / 'models/results/mimic_primary_main_patient_bootstrap_ci.csv'
EICU_LABELS = Path(os.environ['EICU_RUN_ROOT']) / 'corrected/private/corrected_labels.parquet'
EICU_FEATURES = Path(os.environ['EICU_RUN_ROOT']) / 'derived/stage2b_explicit_stop_feature_matrix.parquet'
EICU_MODELS = Path(os.environ['EICU_RUN_ROOT']) / 'parameters/primary_traceable__target_native_refit.json'
EICU_POINTS = Path(os.environ['EICU_RUN_ROOT']) / 'corrected/results/primary_traceable_performance.csv'
EICU_CI = Path(os.environ['EICU_RUN_ROOT']) / 'corrected/results/primary_traceable_hospital_CI.csv'

def metrics(y, p):
    return {'auroc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p), 'brier': brier_score_loss(y, p)}

def q(v):
    x = pd.to_numeric(v, errors='coerce').dropna()
    if not len(x):
        return ('Not available', 0)
    a, b, c = x.quantile([0.25, 0.5, 0.75])
    return (f'{b:.2f} [{a:.2f}, {c:.2f}]', len(x))

def add_cont(rows, database, group, frame, var, label):
    value, nonmissing = q(frame[var]) if var in frame else ('Not available', 0)
    n = len(frame)
    rows.append(dict(database=database, group=group, variable=label, level='Median [Q1, Q3]', value=value, numerator='', denominator=nonmissing, missing_n=n - nonmissing if var in frame else n, missing_pct=(n - nonmissing) / n * 100 if n else np.nan, source_variable=var if var in frame else 'not available'))

def add_cat(rows, database, group, frame, var, label):
    n = len(frame)
    if var not in frame:
        rows.append(dict(database=database, group=group, variable=label, level='Not available', value='Not available', numerator='', denominator=n, missing_n=n, missing_pct=100.0 if n else np.nan, source_variable='not available'))
        return
    s = frame[var]
    missing = int(s.isna().sum())
    for level, count in s.dropna().astype(str).value_counts().sort_index().items():
        rows.append(dict(database=database, group=group, variable=label, level=level, value=f'{count}/{n} ({count / n * 100:.1f}%)', numerator=count, denominator=n, missing_n=missing, missing_pct=missing / n * 100 if n else np.nan, source_variable=var))

def characteristics(database, group, frame, outcome_col):
    rows = []
    for var, label in [('age', 'Age, years'), ('duration_hours', 'Vasopressor episode duration, h'), ('map_last', 'Last MAP, mmHg'), ('map_mean_1h', 'Mean MAP, prior 1 h, mmHg'), ('map_mean_6h', 'Mean MAP, prior 6 h, mmHg'), ('heart_rate_last', 'Last heart rate, beats/min'), ('heart_rate_mean_6h', 'Mean heart rate, prior 6 h, beats/min'), ('respiratory_rate_last', 'Last respiratory rate, breaths/min'), ('temperature_c_last', 'Last temperature, C'), ('nee_last_5min', 'Last NEE'), ('nee_mean_6h', 'Mean NEE, prior 6 h'), ('modified_shock_index_last', 'Modified shock index'), ('distinct_drugs', 'Distinct vasopressors in episode'), ('active_agents_last_5min', 'Active vasopressors at last grid point')]:
        add_cont(rows, database, group, frame, var, label)
    for var, label in [('gender', 'Sex'), ('race', 'Race'), ('admission_type', 'Admission type'), ('terminal_drug', 'Terminal vasopressor'), (outcome_col, 'Competing outcome')]:
        add_cat(rows, database, group, frame, var, label)
    return rows

def selection_rows(database, all_frame, included_col, status_col):
    rows = []
    subsets = {'included_primary': all_frame[all_frame[included_col].astype(bool)], 'excluded_or_indeterminate': all_frame[~all_frame[included_col].astype(bool)]}
    for group, frame in subsets.items():
        rows.extend(characteristics(database, group, frame, status_col))
    return rows

def main():
    mc = pd.read_parquet(MIMIC_COHORT)
    mf = pd.read_parquet(MIMIC_FEATURES)
    el = pd.read_parquet(EICU_LABELS)
    ef = pd.read_parquet(EICU_FEATURES)
    el = el[el.variant.eq('primary_traceable')].copy()
    assert not mc.stay_id.duplicated().any()
    assert not el.patienthealthsystemstayid.duplicated().any()
    assert not ef.patienthealthsystemstayid.duplicated().any()
    assert not mf.stay_id.duplicated().any()
    m = mc.merge(mf, on='stay_id', how='left', validate='one_to_one', suffixes=('', '_feature'), indicator=True)
    e = el.merge(ef, on='patienthealthsystemstayid', how='left', validate='one_to_one', suffixes=('', '_feature'), indicator=True)
    assert m._merge.eq('both').all()
    assert e._merge.eq('both').all()
    m_in = m[m.primary_analysis_include.eq(1)].copy()
    e_in = e[e.included].copy()
    group_map = {'development': 'MIMIC-IV development', 'temporal_validation': 'MIMIC-IV temporal validation'}
    egroup_map = {'adaptation_development': 'eICU development', 'recalibration': 'eICU recalibration', 'locked_evaluation': 'eICU locked evaluation'}
    rows = []
    for split, label in group_map.items():
        rows += characteristics('MIMIC-IV', label, m_in[m_in.corrected_analysis_split.eq(split)], 'corrected_status')
    for split, label in egroup_map.items():
        rows += characteristics('eICU', label, e_in[e_in.partition.eq(split)], 'status')
    chars = pd.DataFrame(rows)
    chars.to_csv(OUT / 'audit_corrected_characteristics_long.csv', index=False, encoding='utf-8-sig')
    selection = pd.DataFrame(selection_rows('MIMIC-IV', m, 'primary_analysis_include', 'corrected_status') + selection_rows('eICU', e, 'included', 'status'))
    selection.to_csv(OUT / 'included_vs_excluded_characteristics_long.csv', index=False, encoding='utf-8-sig')
    cohort = []
    for split, label in group_map.items():
        z = m_in[m_in.corrected_analysis_split.eq(split)]
        cohort.append(dict(database='MIMIC-IV', group=label, records=len(z), patients=z.subject_id.nunique(), admissions=z.hadm_id.nunique(), icu_stays=z.stay_id.nunique(), restart_events=int(z.corrected_outcome_code.eq(1).sum()), death_events=int(z.corrected_outcome_code.eq(2).sum()), hospitals='Not available'))
    for split, label in egroup_map.items():
        z = e_in[e_in.partition.eq(split)]
        cohort.append(dict(database='eICU', group=label, records=len(z), patients=z.uniquepid.nunique(), admissions=z.patienthealthsystemstayid.nunique(), icu_stays='Not available', restart_events=int(z.new_code.eq(1).sum()), death_events=int(z.new_code.eq(2).sum()), hospitals=z.hospitalid.nunique()))
    cohort = pd.DataFrame(cohort)
    cohort.to_csv(OUT / 'cohort_and_outcome_summary.csv', index=False, encoding='utf-8-sig')
    mb = source_bundle(MIMIC_MODELS.parent, 'primary')
    mt = m_in[m_in.corrected_analysis_split.eq('temporal_validation')]
    mm = mb['models']['dynamic_multinomial_logistic']
    fs = mb['features']['dynamic_multinomial_logistic']
    mp = mm.predict_proba(mt[fs])[:, list(mm.classes_).index(1)]
    my = mt.corrected_outcome_code.eq(1).astype(int).to_numpy()
    mimic_verified = metrics(my, mp)
    eb = {'target_native_refit': FrozenJSONModel(EICU_MODELS)}
    ee = e_in[e_in.partition.eq('locked_evaluation')]
    target = eb['target_native_refit']
    tp = target.predict_proba(ee[list(target.feature_names_in_)])[:, list(target.classes_).index(1)]
    ey = ee.new_code.eq(1).astype(int).to_numpy()
    target_verified = metrics(ey, tp)
    source = mm
    sp = source.predict_proba(ee[fs])[:, list(source.classes_).index(1)]
    source_verified = metrics(ey, sp)
    mp_saved = pd.read_csv(MIMIC_POINTS)
    ep_saved = pd.read_csv(EICU_POINTS)
    mci = pd.read_csv(MIMIC_CI)
    eci = pd.read_csv(EICU_CI)
    mci_target = mci[mci.model.eq('dynamic_multinomial_logistic') & mci.outcome.eq('restart_first')]
    eci_target = eci[eci.model.isin(['corrected_primary_source_raw', 'target_native_refit']) & eci.metric.eq('auroc')]
    overlap = el.groupby('uniquepid' if 'uniquepid' in el else 'patienthealthsystemstayid').partition.nunique()
    cross = int(e.groupby('uniquepid').partition.nunique().gt(1).sum())
    verification = {'input_paths': {'mimic_cohort': str(MIMIC_COHORT), 'mimic_features': str(MIMIC_FEATURES), 'eicu_labels': str(EICU_LABELS), 'eicu_features': str(EICU_FEATURES)}, 'join_QA': {'mimic_rows': len(m), 'mimic_unmatched': int(m._merge.ne('both').sum()), 'eicu_rows': len(e), 'eicu_unmatched': int(e._merge.ne('both').sum()), 'mimic_duplicate_keys': int(m.stay_id.duplicated().sum()), 'eicu_duplicate_keys': int(e.patienthealthsystemstayid.duplicated().sum())}, 'mimic': {'included': len(m_in), 'development': int(m_in.corrected_analysis_split.eq('development').sum()), 'temporal': len(mt), 'temporal_restart': int(my.sum()), 'independent_metrics': mimic_verified, 'saved_target_row': mp_saved[mp_saved.model.eq('dynamic_multinomial_logistic') & mp_saved.outcome.eq('restart_first')].to_dict('records'), 'bootstrap_CI': mci_target.to_dict('records'), 'bootstrap_unit': 'subject_id', 'attempted_replicates': 2000}, 'eicu': {'original_candidates': len(ef), 'included': len(e_in), 'partitions': e_in.partition.value_counts().to_dict(), 'locked_restart': int(ey.sum()), 'locked_hospitals': int(ee.hospitalid.nunique()), 'cross_partition_patients': cross, 'independent_source_metrics': source_verified, 'independent_target_metrics': target_verified, 'saved_rows': ep_saved[ep_saved.model.isin(['corrected_primary_source_raw', 'target_native_refit'])].to_dict('records'), 'bootstrap_CI_auroc': eci_target.to_dict('records'), 'bootstrap_unit': 'hospitalid', 'attempted_replicates': 1000}}
    (OUT / 'independent_result_verification.json').write_text(json.dumps(verification, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(json.dumps(verification, indent=2, ensure_ascii=False, default=str))
if __name__ == '__main__':
    main()
