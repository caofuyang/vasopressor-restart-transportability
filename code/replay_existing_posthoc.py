"""Post-hoc aggregate enhancements using frozen labels, partitions, and predictions.

No model is fitted or tuned in this script. Patient-level inputs and predictions are
read only; only aggregate tables and figures are written outside ``restricted``.
"""
from __future__ import annotations
import hashlib
import os
from model_from_json import source_bundle as load_source_bundle, FrozenJSONModel
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ['POSTHOC_OUTPUT'])
RESTRICTED = OUT / 'restricted'
MIMIC_FINAL = Path(os.environ['MIMIC_RUN']) / 'scope_corrected/restricted/stage18_post_audit_scope_corrected_cohort.parquet'
MIMIC_FEATURES = Path(os.environ['MIMIC_RUN']) / 'derived/vasopressor_liberation_features_stage7.parquet'
MIMIC_OLD_PRED = Path(os.environ['HISTORY_REPLAY']) / 'legacy_fixed_predictions.parquet'
MIMIC_OLD_PERF = HERE.parent / 'historical_reference_aggregates/mimic_pre_audit_reported_performance.csv'
MIMIC_NEW_PERF = Path(os.environ['MIMIC_RUN']) / 'models/results/mimic_primary_main_temporal_results.csv'
EICU_LABELS = Path(os.environ['EICU_RUN_ROOT']) / 'corrected/private/corrected_labels.parquet'
EICU_FEATURES = Path(os.environ['EICU_RUN_ROOT']) / 'derived/stage2b_explicit_stop_feature_matrix.parquet'
EICU_SOURCE_MODEL = Path(os.environ['MIMIC_RUN']) / 'models/parameters/mimic_primary__dynamic_multinomial_logistic.json'
EICU_TARGET_MODELS = Path(os.environ['EICU_RUN_ROOT']) / 'parameters/primary_traceable__target_native_refit.json'
EICU_OLD_NEW = Path(os.environ['HISTORY_REPLAY']) / 'transport_before_after.csv'
LOCKED_PRED = Path(os.environ['DISPLAY_OUTPUT']) / 'restricted/locked_predictions.parquet'
RULES = HERE / '00_POSTHOC_REPORTING_RULES_FROZEN.md'
SEED_MIMIC = 20260825
BOOTSTRAPS_MIMIC = 2000

def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {'auroc': float(roc_auc_score(y, p)), 'auprc': float(average_precision_score(y, p)), 'brier': float(brier_score_loss(y, p)), 'event_rate': float(np.mean(y)), 'mean_predicted_risk': float(np.mean(p))}

def broad_old_mimic(x: str) -> str:
    return {'durable_liberation_alive': 'no_recorded_event', 'restart_first': 'restart_first', 'death_first': 'death_first'}.get(str(x), 'other')

def broad_new_mimic(x: str) -> str:
    if x == 'durable_liberation_alive_confirmed':
        return 'no_recorded_event'
    if x == 'restart_first':
        return 'restart_first'
    if x == 'death_first':
        return 'death_first'
    return 'excluded_or_indeterminate'

def broad_old_eicu(x: str) -> str:
    if x == 'durable_liberation_alive':
        return 'no_recorded_event'
    if x == 'restart_first':
        return 'restart_first'
    if x == 'death_first':
        return 'death_first'
    return 'other'

def broad_new_eicu(x: str) -> str:
    if x == 'no_recorded_restart_or_death_proxy_under_observation':
        return 'no_recorded_event'
    if x == 'recorded_restart_first':
        return 'restart_first'
    if x == 'recorded_death_proxy_first':
        return 'death_first'
    if x == 'tie_restart_and_administrative_death_proxy':
        return 'tie'
    return 'excluded_or_indeterminate'

def smd_cont(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors='coerce').dropna()
    b = pd.to_numeric(b, errors='coerce').dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan

def smd_binary(a: pd.Series, b: pd.Series) -> float:
    pa, pb = (float(a.mean()), float(b.mean()))
    pooled = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return float((pa - pb) / pooled) if pooled > 0 else np.nan

def selection_table(database: str, data: pd.DataFrame, include_col: str) -> pd.DataFrame:
    included = data[data[include_col].astype(bool)].copy()
    excluded = data[~data[include_col].astype(bool)].copy()
    specs = [('age', 'Age, years'), ('duration_hours', 'Vasopressor episode duration, h'), ('map_last', 'Last MAP, mmHg'), ('map_mean_6h', 'Mean MAP, prior 6 h, mmHg'), ('heart_rate_last', 'Last heart rate, beats/min'), ('heart_rate_mean_6h', 'Mean heart rate, prior 6 h, beats/min'), ('respiratory_rate_last', 'Last respiratory rate, breaths/min'), ('temperature_c_last', 'Last temperature, C'), ('nee_last_5min', 'Last norepinephrine-equivalent dose'), ('nee_mean_6h', 'Mean norepinephrine-equivalent dose, prior 6 h'), ('distinct_drugs', 'Distinct vasopressors in episode'), ('modified_shock_index_last', 'Modified shock index')]
    rows = []
    for col, label in specs:
        if col not in data:
            continue
        ai = pd.to_numeric(included[col], errors='coerce')
        ae = pd.to_numeric(excluded[col], errors='coerce')
        rows.append({'database': database, 'variable': label, 'type': 'continuous', 'included_n': len(included), 'excluded_n': len(excluded), 'included_nonmissing': int(ai.notna().sum()), 'excluded_nonmissing': int(ae.notna().sum()), 'included_mean': ai.mean(), 'excluded_mean': ae.mean(), 'included_sd': ai.std(ddof=1), 'excluded_sd': ae.std(ddof=1), 'standardized_difference': smd_cont(ai, ae), 'included_missing_pct': ai.isna().mean() * 100, 'excluded_missing_pct': ae.isna().mean() * 100})
    gi = included['gender'].astype(str).str.upper().isin(['F', 'FEMALE']).astype(float)
    ge = excluded['gender'].astype(str).str.upper().isin(['F', 'FEMALE']).astype(float)
    rows.append({'database': database, 'variable': 'Female sex', 'type': 'binary', 'included_n': len(included), 'excluded_n': len(excluded), 'included_nonmissing': int(included.gender.notna().sum()), 'excluded_nonmissing': int(excluded.gender.notna().sum()), 'included_mean': gi.mean(), 'excluded_mean': ge.mean(), 'included_sd': gi.std(ddof=1), 'excluded_sd': ge.std(ddof=1), 'standardized_difference': smd_binary(gi, ge), 'included_missing_pct': included.gender.isna().mean() * 100, 'excluded_missing_pct': excluded.gender.isna().mean() * 100})
    return pd.DataFrame(rows)

def qsummary(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors='coerce').dropna()
    return {'n_nonmissing': int(len(x)), 'median': float(x.median()) if len(x) else np.nan, 'q1': float(x.quantile(0.25)) if len(x) else np.nan, 'q3': float(x.quantile(0.75)) if len(x) else np.nan, 'minimum': float(x.min()) if len(x) else np.nan, 'maximum': float(x.max()) if len(x) else np.nan}

def predict(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(frame[features])[:, list(model.classes_).index(1)]

def anonymized_hospital(hospitalid: int) -> str:
    token = hashlib.sha256(f'JIC-20260909-{int(hospitalid)}'.encode()).hexdigest()[:6].upper()
    return f'H-{token}'

def main() -> None:
    required = [MIMIC_FINAL, MIMIC_FEATURES, MIMIC_OLD_PRED, MIMIC_OLD_PERF, MIMIC_NEW_PERF, EICU_LABELS, EICU_FEATURES, EICU_SOURCE_MODEL, EICU_TARGET_MODELS, EICU_OLD_NEW, LOCKED_PRED, RULES]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    RESTRICTED.mkdir()
    mimic = pd.read_parquet(MIMIC_FINAL)
    mf = pd.read_parquet(MIMIC_FEATURES)
    if mimic.stay_id.duplicated().any() or mf.stay_id.duplicated().any():
        raise AssertionError('MIMIC grain')
    m = mimic.merge(mf, on='stay_id', how='left', validate='one_to_one', suffixes=('', '_feature'), indicator=True)
    if not m._merge.eq('both').all():
        raise AssertionError('MIMIC feature join')
    labels = pd.read_parquet(EICU_LABELS)
    labels = labels[labels.variant.eq('primary_traceable')].copy()
    ef = pd.read_parquet(EICU_FEATURES)
    if labels.patienthealthsystemstayid.duplicated().any() or ef.patienthealthsystemstayid.duplicated().any():
        raise AssertionError('eICU grain')
    e = labels.merge(ef, on='patienthealthsystemstayid', how='left', validate='one_to_one', suffixes=('', '_feature'), indicator=True)
    if not e._merge.eq('both').all():
        raise AssertionError('eICU feature join')
    m['original_category'] = m.competing_outcome.map(broad_old_mimic)
    m['final_category'] = m.corrected_status.map(broad_new_mimic)
    mt = m.groupby(['competing_outcome', 'corrected_status'], dropna=False).size().reset_index(name='records')
    mt.to_csv(OUT / 'mimic_endpoint_transition_detailed.csv', index=False)
    e['original_category'] = e.old_outcome.map(broad_old_eicu)
    e['final_category'] = e.status.map(broad_new_eicu)
    et = e.groupby(['old_outcome', 'status', 'included'], dropna=False).size().reset_index(name='records')
    et.to_csv(OUT / 'eicu_endpoint_transition_detailed.csv', index=False)
    compact_rows = []
    for database, frame, include_col in [('MIMIC-IV', m, 'primary_analysis_include'), ('eICU', e, 'included')]:
        inc = frame[frame[include_col].astype(bool)]
        compact_rows.append({'database': database, 'fixed_candidates': len(frame), 'primary_analysis': len(inc), 'excluded_or_indeterminate': len(frame) - len(inc), 'original_no_recorded_event': int((frame.original_category == 'no_recorded_event').sum()), 'original_restart_first': int((frame.original_category == 'restart_first').sum()), 'original_death_first': int((frame.original_category == 'death_first').sum()), 'final_no_recorded_event': int((inc.final_category == 'no_recorded_event').sum()), 'final_restart_first': int((inc.final_category == 'restart_first').sum()), 'final_death_first': int((inc.final_category == 'death_first').sum()), 'final_tie': int((inc.final_category == 'tie').sum()), 'broad_analysis_category_changed': int((frame.original_category != frame.final_category).sum())})
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(OUT / 'endpoint_correction_impact_compact.csv', index=False)
    mo = pd.read_csv(MIMIC_OLD_PERF)
    mn = pd.read_csv(MIMIC_NEW_PERF)
    mo = mo[(mo.model == 'dynamic_multinomial_logistic') & (mo.outcome == 'restart_first')].iloc[0]
    mn = mn[(mn.model == 'dynamic_multinomial_logistic') & (mn.outcome == 'restart_first')].iloc[0]
    eo = pd.read_csv(EICU_OLD_NEW)
    er = eo[(eo.variant == 'primary_traceable') & (eo.model == 'corrected_primary_source_raw')].iloc[0]
    perf = pd.DataFrame([{'database': 'MIMIC-IV', 'version': 'historical original analysis (archived)', 'n': int(mo.test_n), 'events': int(mo.test_events), 'event_rate': mo.prevalence, 'auroc': mo.auroc, 'auprc': mo.auprc, 'brier': mo.brier_ovr, 'comparison_scope': 'Different analysis population, endpoint implementation, and fitted model'}, {'database': 'MIMIC-IV', 'version': 'final corrected analysis', 'n': int(mn.test_n), 'events': int(mn.test_events), 'event_rate': mn.prevalence, 'auroc': mn.auroc, 'auprc': mn.auprc, 'brier': mn.brier, 'comparison_scope': 'Different analysis population, endpoint implementation, and fitted model'}, {'database': 'eICU', 'version': 'historical original analysis (archived)', 'n': int(er.n_old), 'events': int(er.events_old), 'event_rate': er.prevalence_old, 'auroc': er.auroc_old, 'auprc': er.auprc_old, 'brier': er.brier_old, 'comparison_scope': 'Different analysis population and eICU labels; source model also changed after MIMIC correction'}, {'database': 'eICU', 'version': 'final corrected analysis', 'n': int(er.n_new), 'events': int(er.events_new), 'event_rate': er.prevalence_new, 'auroc': er.auroc_new, 'auprc': er.auprc_new, 'brier': er.brier_new, 'comparison_scope': 'Different analysis population and eICU labels; source model also changed after MIMIC correction'}])
    perf.to_csv(OUT / 'original_vs_corrected_performance_descriptive.csv', index=False)
    oldp = pd.read_parquet(MIMIC_OLD_PRED)
    cur = m[(m.primary_analysis_include == 1) & (m.corrected_analysis_split == 'temporal_validation')][['subject_id', 'hadm_id', 'stay_id', 'corrected_status']]
    same = cur.merge(oldp, on=['subject_id', 'hadm_id', 'stay_id'], how='inner', validate='one_to_one')
    if len(same) != len(cur):
        raise AssertionError(f'MIMIC same-record coverage {len(same)}/{len(cur)}')
    p = same['dynamic_multinomial_logistic__p_restart_first'].to_numpy(float)
    y_old = (same.observed_outcome.to_numpy() == 1).astype(int)
    y_new = (same.corrected_status.to_numpy() == 'restart_first').astype(int)
    point_old, point_new = (metrics(y_old, p), metrics(y_new, p))
    groups = {sid: np.flatnonzero(same.subject_id.to_numpy() == sid) for sid in same.subject_id.unique()}
    ids = np.array(list(groups))
    rng = np.random.default_rng(SEED_MIMIC)
    boot = {k: [] for k in ['auroc', 'auprc', 'brier', 'event_rate']}
    failed = 0
    for _ in range(BOOTSTRAPS_MIMIC):
        ix = np.concatenate([groups[sid] for sid in rng.choice(ids, len(ids), replace=True)])
        yo, yn, pp = (y_old[ix], y_new[ix], p[ix])
        if len(np.unique(yo)) < 2 or len(np.unique(yn)) < 2:
            failed += 1
            continue
        a, b = (metrics(yo, pp), metrics(yn, pp))
        for key in boot:
            boot[key].append(b[key] - a[key])
    rows = []
    for label, vals in [('original_label', point_old), ('corrected_label', point_new)]:
        rows.append({'label_version': label, 'n': len(same), 'patients': same.subject_id.nunique(), 'events': int((y_old if label == 'original_label' else y_new).sum()), **vals})
    pd.DataFrame(rows).to_csv(OUT / 'mimic_same_record_fixed_prediction_metrics.csv', index=False)
    delta_rows = []
    for key, values in boot.items():
        v = np.asarray(values)
        delta_rows.append({'metric': key, 'corrected_minus_original': point_new[key] - point_old[key], 'ci_low': np.quantile(v, 0.025), 'ci_high': np.quantile(v, 0.975), 'valid_replicates': len(v), 'failed_replicates': failed})
    pd.DataFrame(delta_rows).to_csv(OUT / 'mimic_same_record_fixed_prediction_paired_differences.csv', index=False)
    same.assign(original_restart_label=y_old, corrected_restart_label=y_new)[['observed_outcome', 'corrected_status', 'original_restart_label', 'corrected_restart_label']].groupby(['observed_outcome', 'corrected_status', 'original_restart_label', 'corrected_restart_label']).size().reset_index(name='records').to_csv(OUT / 'mimic_same_record_label_transition.csv', index=False)
    selected = e[e.included.astype(bool)].copy()
    evaluation = selected[selected.partition == 'locked_evaluation'].copy().reset_index(drop=True)
    y = (evaluation.new_code.to_numpy() == 1).astype(int)
    source_bundle = load_source_bundle(EICU_SOURCE_MODEL.parent,'primary')
    source_model = source_bundle['models']['dynamic_multinomial_logistic']
    source_features = source_bundle['features']['dynamic_multinomial_logistic']
    target_bundle = {'target_native_refit':FrozenJSONModel(EICU_TARGET_MODELS)}
    target_model = target_bundle['target_native_refit']
    target_features = list(target_model.feature_names_in_)
    p_source = predict(source_model, evaluation, source_features)
    p_target = predict(target_model, evaluation, target_features)
    saved = pd.read_parquet(LOCKED_PRED)
    if len(saved) != len(evaluation) or not np.allclose(saved.source_only, p_source) or (not np.allclose(saved.target_native_full, p_target)):
        raise AssertionError('Locked prediction alignment did not reproduce the saved predictions')
    hospital_rows = []
    for hid, g in evaluation.assign(y=y, p_source=p_source, p_target=p_target).groupby('hospitalid', sort=True):
        yy = g.y.to_numpy(int)
        ps = g.p_source.to_numpy(float)
        pt = g.p_target.to_numpy(float)
        events = int(yy.sum())
        non_events = len(yy) - events
        row = {'hospital_label': anonymized_hospital(hid), 'n': len(g), 'events': events, 'non_events': non_events, 'event_rate': yy.mean(), 'source_mean_risk': ps.mean(), 'target_mean_risk': pt.mean(), 'source_brier': brier_score_loss(yy, ps), 'target_brier': brier_score_loss(yy, pt), 'source_observed_minus_predicted': yy.mean() - ps.mean(), 'target_observed_minus_predicted': yy.mean() - pt.mean(), 'source_excess_squared_error_sum': float(np.sum((yy - ps) ** 2 - (yy - pt) ** 2)), 'auroc_reporting_eligible': events >= 10 and non_events >= 10}
        row['source_auroc'] = roc_auc_score(yy, ps) if row['auroc_reporting_eligible'] else np.nan
        row['target_auroc'] = roc_auc_score(yy, pt) if row['auroc_reporting_eligible'] else np.nan
        hospital_rows.append(row)
    hospitals = pd.DataFrame(hospital_rows).sort_values(['n', 'hospital_label'], ascending=[False, True]).reset_index(drop=True)
    hospitals.to_csv(OUT / 'eicu_hospital_heterogeneity_anonymized.csv', index=False)
    evaluation.assign(y=y, p_source=p_source, p_target=p_target).to_parquet(RESTRICTED / 'locked_predictions_with_hospitalid.parquet', index=False)
    total_n = len(evaluation)
    total_events = int(y.sum())
    largest = hospitals.iloc[0]
    top5 = hospitals.head(5)
    total_positive_excess = hospitals.source_excess_squared_error_sum.clip(lower=0).sum()
    concentration = {'hospitals': len(hospitals), 'locked_n': total_n, 'events': total_events, 'largest_hospital_n': int(largest.n), 'largest_hospital_record_share': float(largest.n / total_n), 'largest_hospital_event_share': float(largest.events / total_events), 'five_largest_record_share': float(top5.n.sum() / total_n), 'five_largest_event_share': float(top5.events.sum() / total_events), 'hospitals_auroc_eligible': int(hospitals.auroc_reporting_eligible.sum()), 'largest_positive_excess_error_contribution_share': float(hospitals.source_excess_squared_error_sum.clip(lower=0).max() / total_positive_excess), 'five_largest_positive_excess_error_contribution_share': float(hospitals.nlargest(5, 'source_excess_squared_error_sum').source_excess_squared_error_sum.clip(lower=0).sum() / total_positive_excess)}
    loo = []
    arr = evaluation.assign(y=y, p_source=p_source, p_target=p_target)
    for hid in sorted(arr.hospitalid.unique()):
        g = arr[arr.hospitalid != hid]
        for model, col in [('source_only', 'p_source'), ('target_native', 'p_target')]:
            mm = metrics(g.y.to_numpy(int), g[col].to_numpy(float))
            loo.append({'omitted_hospital_label': anonymized_hospital(hid), 'model': model, **mm})
    loo = pd.DataFrame(loo)
    loo.to_csv(OUT / 'eicu_leave_one_hospital_out_metrics.csv', index=False)
    for model in ['source_only', 'target_native']:
        x = loo[loo.model == model]
        for metric in ['auroc', 'auprc', 'brier', 'mean_predicted_risk']:
            concentration[f'{model}_loo_{metric}_min'] = float(x[metric].min())
            concentration[f'{model}_loo_{metric}_max'] = float(x[metric].max())
    (OUT / 'eicu_hospital_concentration_summary.json').write_text(json.dumps(concentration, indent=2), encoding='utf-8')
    msel = selection_table('MIMIC-IV', m, 'primary_analysis_include')
    esel = selection_table('eICU', e, 'included')
    selection = pd.concat([msel, esel], ignore_index=True)
    selection['absolute_standardized_difference'] = selection.standardized_difference.abs()
    selection.to_csv(OUT / 'observability_selection_standardized_differences.csv', index=False)
    m['postindex_administrative_observation_hours_capped24'] = ((pd.to_datetime(m.raw_dischtime) - pd.to_datetime(m.episode_end)).dt.total_seconds() / 3600).clip(lower=0, upper=24)
    e['postindex_administrative_observation_hours_capped24'] = ((pd.to_numeric(e.observation_end, errors='coerce') - pd.to_numeric(e.origin, errors='coerce')) / 60).clip(lower=0, upper=24)
    obs = []
    for database, frame, inc in [('MIMIC-IV', m, 'primary_analysis_include'), ('eICU', e, 'included')]:
        for label, sub in [('included_primary', frame[frame[inc].astype(bool)]), ('excluded_or_indeterminate', frame[~frame[inc].astype(bool)])]:
            obs.append({'database': database, 'group': label, 'n': len(sub), **qsummary(sub.postindex_administrative_observation_hours_capped24)})
    pd.DataFrame(obs).to_csv(OUT / 'postindex_observation_time_summary.csv', index=False)
    m.groupby('corrected_status').size().reset_index(name='records').to_csv(OUT / 'mimic_exclusion_reason_counts.csv', index=False)
    e.groupby(['included', 'status']).size().reset_index(name='records').to_csv(OUT / 'eicu_exclusion_reason_counts.csv', index=False)
    plot = hospitals.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 9.2), sharey=True, gridspec_kw={'width_ratios': [1.0, 1.0, 1.45]}, constrained_layout=True)
    axes[0].barh(y_pos, plot.n, color='#4C78A8', edgecolor='#243B53', linewidth=0.4)
    axes[0].set_xlabel('Records')
    axes[0].set_title('A  Hospital sample size')
    axes[0].set_yticks(y_pos, plot.hospital_label, fontsize=6)
    axes[0].grid(axis='x', alpha=0.18)
    axes[1].scatter(plot.event_rate, y_pos, s=18, color='#D28E2D', edgecolor='#6B4B16', linewidth=0.4)
    axes[1].axvline(y.mean(), color='#555555', ls='--', lw=1, label='Overall event rate')
    axes[1].set_xlim(-0.03, 1.03)
    axes[1].set_xlabel('Restart-first proportion')
    axes[1].set_title('B  Observed event rate')
    axes[1].grid(axis='x', alpha=0.18)
    axes[2].axvline(0, color='#555555', lw=1)
    axes[2].scatter(plot.source_observed_minus_predicted, y_pos, s=18, color='#C44E52', label='MIMIC source-only', edgecolor='#6C2528', linewidth=0.4)
    axes[2].scatter(plot.target_observed_minus_predicted, y_pos, s=18, facecolor='white', edgecolor='#4C72B0', label='eICU target-native', linewidth=1)
    axes[2].set_xlim(-1.03, 1.03)
    axes[2].set_xlabel('Observed minus mean predicted risk')
    axes[2].set_title('C  Hospital risk gap')
    axes[2].grid(axis='x', alpha=0.18)
    axes[2].legend(frameon=False, fontsize=7, loc='lower right')
    fig.suptitle('Exploratory hospital heterogeneity in the eICU locked-evaluation set', fontsize=11)
    for ext in ['pdf', 'png', 'tiff']:
        fig.savefig(OUT / f'Figure_4_eICU_hospital_heterogeneity.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    qa = {'rules_file': str(RULES), 'rules_sha256': hashlib.sha256(RULES.read_bytes()).hexdigest(), 'mimic_rows': len(m), 'mimic_unique_stays': int(m.stay_id.nunique()), 'mimic_feature_join_complete': True, 'eicu_rows': len(e), 'eicu_unique_admissions': int(e.patienthealthsystemstayid.nunique()), 'eicu_feature_join_complete': True, 'mimic_same_record_n': len(same), 'mimic_same_record_patients': int(same.subject_id.nunique()), 'mimic_same_record_label_changes': int(np.sum(y_old != y_new)), 'mimic_bootstrap_attempted': BOOTSTRAPS_MIMIC, 'mimic_bootstrap_failed': failed, 'mimic_bootstrap_unit': 'subject_id', 'eicu_locked_n': len(evaluation), 'eicu_locked_events': int(y.sum()), 'eicu_locked_hospitals': int(evaluation.hospitalid.nunique()), 'saved_locked_predictions_exactly_reproduced': True, 'hospital_auroc_min_events_and_nonevents': 10, 'patient_level_output_restricted': True, 'patient_identifiers_in_public_outputs': False, 'models_refit_or_tuned': False, 'partitions_or_labels_changed': False, 'all_checks_passed': True}
    (OUT / 'enhancement_QA.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(OUT), 'compact': compact.to_dict(orient='records'), 'same_record': rows, 'concentration': concentration, 'qa': qa}, indent=2, default=str))
if __name__ == '__main__':
    main()
