"""Build aggregate clinical-result enhancements without refitting any model.

Patient-level inputs are read locally. Only aggregate CSV/JSON and figures are
written to the public output directory.
"""
from __future__ import annotations
import hashlib
import os
from model_from_json import source_bundle as load_source_bundle, ablation_bundle as load_ablation_bundle
import json
import warnings
from pathlib import Path
import duckdb
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ['CLINICAL_OUTPUT'])
MIMIC_COHORT = Path(os.environ['MIMIC_RUN']) / 'scope_corrected/restricted/stage18_post_audit_scope_corrected_cohort.parquet'
MIMIC_FEATURES = Path(os.environ['MIMIC_RUN']) / 'derived/vasopressor_liberation_features_stage7.parquet'
EICU_LABELS = Path(os.environ['EICU_RUN_ROOT']) / 'corrected/private/corrected_labels.parquet'
EICU_FEATURES = Path(os.environ['EICU_RUN_ROOT']) / 'derived/stage2b_explicit_stop_feature_matrix.parquet'
EICU_RESULTS = Path(os.environ['EICU_RUN_ROOT']) / 'corrected/results'
SOURCE_MODELS = Path(os.environ['REPRO_PARAMETERS']) / 'mimic_primary__dynamic_multinomial_logistic.json'
ABLATION_MODELS = Path(os.environ['REPRO_PARAMETERS']) / 'ablation__source__corrected_primary__harmonized_dynamic_vitals.json'
RULES = HERE / '00_CLINICAL_RESULTS_ENHANCEMENT_RULES_FROZEN.md'
FEATURE_META = {'age': ('Age', 'years', 'At admission/index record'), 'duration_hours': ('Qualifying vasopressor episode duration', 'hours', 'Episode start to reconstructed cessation'), 'distinct_drugs': ('Distinct vasopressors in qualifying episode', 'count', 'Qualifying episode'), 'map_last': ('Last MAP', 'mm Hg', 'Latest recorded value before cessation'), 'heart_rate_last': ('Last heart rate', 'beats/min', 'Latest recorded value before cessation'), 'nee_last_5min': ('Last norepinephrine-equivalent dose', 'micrograms/kg/min', 'Last 5 min before cessation'), 'active_agents_last_5min': ('Active vasopressors', 'count', 'Last 5 min before cessation'), 'map_mean_1h': ('Mean MAP', 'mm Hg', '1 h before cessation'), 'map_mean_6h': ('Mean MAP', 'mm Hg', '6 h before cessation'), 'heart_rate_mean_1h': ('Mean heart rate', 'beats/min', '1 h before cessation'), 'heart_rate_mean_6h': ('Mean heart rate', 'beats/min', '6 h before cessation'), 'nee_mean_1h': ('Mean norepinephrine-equivalent dose', 'micrograms/kg/min', '1 h before cessation'), 'nee_mean_6h': ('Mean norepinephrine-equivalent dose', 'micrograms/kg/min', '6 h before cessation'), 'maximum_active_agents_6h': ('Maximum simultaneous vasopressors', 'count', '6 h before cessation')}
FEATURES = list(FEATURE_META)
OUTCOME_ORDER = ['no_recorded_event', 'restart_first', 'death_first']
OUTCOME_LABELS = {'no_recorded_event': 'No recorded event under complete observation', 'restart_first': 'Restart first', 'death_first': 'Death first / administrative death proxy first'}
BINS = [0.0, 1.0, 6.0, 12.0, 24.0]
BIN_LABELS = ['(0,1]', '(1,6]', '(6,12]', '(12,24]']

def sql_path(path: Path) -> str:
    return str(path).replace('\\', '/').replace("'", "''")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def smd(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors='coerce').dropna().astype(float)
    b = pd.to_numeric(b, errors='coerce').dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = [MIMIC_COHORT, MIMIC_FEATURES, EICU_LABELS, EICU_FEATURES, SOURCE_MODELS, ABLATION_MODELS, RULES]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    con = duckdb.connect(':memory:')
    con.execute('SET threads=4')
    mc, mf = (sql_path(MIMIC_COHORT), sql_path(MIMIC_FEATURES))
    el, ef = (sql_path(EICU_LABELS), sql_path(EICU_FEATURES))
    mimic_feature_dupes = con.execute(f"SELECT count(*) FROM (SELECT subject_id,hadm_id,stay_id,episode_id,count(*) n FROM read_parquet('{mf}') GROUP BY ALL HAVING n>1)").fetchone()[0]
    eicu_feature_dupes = con.execute(f"SELECT count(*) FROM (SELECT patienthealthsystemstayid,count(*) n FROM read_parquet('{ef}') GROUP BY 1 HAVING n>1)").fetchone()[0]
    if mimic_feature_dupes or eicu_feature_dupes:
        raise AssertionError(f'Feature key duplicates: MIMIC={mimic_feature_dupes}, eICU={eicu_feature_dupes}')
    feature_sql = ',\n'.join((f'f.{name}' for name in FEATURES))
    mimic = con.execute(f"\n        SELECT c.subject_id, c.hadm_id, c.stay_id, c.episode_id,\n               CASE c.corrected_status\n                 WHEN 'durable_liberation_alive_confirmed' THEN 'no_recorded_event'\n                 WHEN 'restart_first' THEN 'restart_first'\n                 WHEN 'death_first' THEN 'death_first' END AS outcome_group,\n               date_diff('second', c.episode_end, c.scoped_restart_time)/3600.0 AS restart_time_hours,\n               {feature_sql}\n        FROM read_parquet('{mc}') c\n        LEFT JOIN read_parquet('{mf}') f\n          USING(subject_id,hadm_id,stay_id,episode_id)\n        WHERE c.primary_analysis_include=1\n        ").df()
    eicu = con.execute(f"\n        SELECT l.patienthealthsystemstayid,\n               CASE l.status\n                 WHEN 'no_recorded_restart_or_death_proxy_under_observation' THEN 'no_recorded_event'\n                 WHEN 'recorded_restart_first' THEN 'restart_first'\n                 WHEN 'recorded_death_proxy_first' THEN 'death_first' END AS outcome_group,\n               (l.restart-l.origin)/60.0 AS restart_time_hours,\n               {feature_sql}\n        FROM read_parquet('{el}') l\n        LEFT JOIN read_parquet('{ef}') f USING(patienthealthsystemstayid)\n        WHERE l.variant='primary_traceable' AND l.included\n        ").df()
    con.close()
    qa = {'mimic_rows': int(len(mimic)), 'mimic_unique_analysis_keys': int(mimic[['subject_id', 'hadm_id', 'stay_id', 'episode_id']].drop_duplicates().shape[0]), 'mimic_feature_key_duplicates': int(mimic_feature_dupes), 'mimic_rows_with_all_features_missing': int(mimic[FEATURES].isna().all(axis=1).sum()), 'eicu_rows': int(len(eicu)), 'eicu_unique_admissions': int(eicu.patienthealthsystemstayid.nunique()), 'eicu_feature_key_duplicates': int(eicu_feature_dupes), 'eicu_rows_with_all_features_missing': int(eicu[FEATURES].isna().all(axis=1).sum())}
    assert qa['mimic_rows'] == qa['mimic_unique_analysis_keys']
    assert qa['eicu_rows'] == qa['eicu_unique_admissions'] == 6779
    assert qa['mimic_rows_with_all_features_missing'] == 0
    assert qa['eicu_rows_with_all_features_missing'] == 0
    return (mimic, eicu, qa)

def build_time_results(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summaries, bins, qa = ([], [], {})
    for database, frame in frames.items():
        restarts = frame.loc[frame.outcome_group.eq('restart_first')].copy()
        time = pd.to_numeric(restarts.restart_time_hours, errors='coerce')
        valid = time.notna() & np.isfinite(time) & time.gt(0) & time.le(24)
        x = time.loc[valid].astype(float)
        summaries.append({'database': database, 'restart_first_total': int(len(restarts)), 'valid_time_n': int(valid.sum()), 'missing_or_invalid_time_n': int((~valid).sum()), 'median_hours': float(x.median()) if len(x) else np.nan, 'q1_hours': float(x.quantile(0.25)) if len(x) else np.nan, 'q3_hours': float(x.quantile(0.75)) if len(x) else np.nan, 'minimum_hours': float(x.min()) if len(x) else np.nan, 'maximum_hours': float(x.max()) if len(x) else np.nan})
        cats = pd.cut(x, bins=BINS, labels=BIN_LABELS, right=True, include_lowest=False)
        counts = cats.value_counts(sort=False)
        for label in BIN_LABELS:
            n = int(counts.get(label, 0))
            bins.append({'database': database, 'interval_hours': label, 'events': n, 'percent_of_all_restart_first': 100.0 * n / len(restarts) if len(restarts) else np.nan, 'percent_of_valid_restart_times': 100.0 * n / len(x) if len(x) else np.nan, 'restart_first_denominator': int(len(restarts)), 'valid_time_denominator': int(len(x))})
        interval_sum = int(counts.sum())
        qa[database] = {'restart_first_total': int(len(restarts)), 'valid_time_n': int(valid.sum()), 'missing_or_invalid_time_n': int((~valid).sum()), 'interval_event_sum': interval_sum, 'interval_sum_equals_valid_time_n': interval_sum == int(valid.sum()), 'valid_plus_missing_equals_restart_total': int(valid.sum()) + int((~valid).sum()) == len(restarts)}
        if not qa[database]['interval_sum_equals_valid_time_n'] or not qa[database]['valid_plus_missing_equals_restart_total']:
            raise AssertionError(f'Restart-time denominator failure: {database}')
    return (pd.DataFrame(summaries), pd.DataFrame(bins), qa)

def build_characteristics(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, smds = ([], [])
    for database, frame in frames.items():
        if set(frame.outcome_group.dropna()) != set(OUTCOME_ORDER):
            raise AssertionError(f'Unexpected outcome groups in {database}')
        for variable in FEATURES:
            label, unit, window = FEATURE_META[variable]
            for outcome in OUTCOME_ORDER:
                group = frame.loc[frame.outcome_group.eq(outcome), variable]
                values = pd.to_numeric(group, errors='coerce')
                observed = values.dropna().astype(float)
                rows.append({'database': database, 'variable': variable, 'variable_label': label, 'unit': unit, 'window': window, 'outcome_group': outcome, 'outcome_label': OUTCOME_LABELS[outcome], 'group_n': int(len(group)), 'observed_n': int(observed.size), 'missing_n': int(values.isna().sum()), 'missing_percent': 100.0 * values.isna().mean(), 'median': float(observed.median()) if len(observed) else np.nan, 'q1': float(observed.quantile(0.25)) if len(observed) else np.nan, 'q3': float(observed.quantile(0.75)) if len(observed) else np.nan, 'mean': float(observed.mean()) if len(observed) else np.nan, 'sd': float(observed.std(ddof=1)) if len(observed) > 1 else np.nan})
            for first, second in [('restart_first', 'no_recorded_event'), ('death_first', 'no_recorded_event'), ('restart_first', 'death_first')]:
                a = frame.loc[frame.outcome_group.eq(first), variable]
                b = frame.loc[frame.outcome_group.eq(second), variable]
                smds.append({'database': database, 'variable': variable, 'variable_label': label, 'unit': unit, 'window': window, 'comparison': f'{first}_minus_{second}', 'first_group': first, 'second_group': second, 'first_group_n': int((frame.outcome_group == first).sum()), 'second_group_n': int((frame.outcome_group == second).sum()), 'first_observed_n': int(pd.to_numeric(a, errors='coerce').notna().sum()), 'second_observed_n': int(pd.to_numeric(b, errors='coerce').notna().sum()), 'standardized_difference': smd(a, b)})
    long = pd.DataFrame(rows)
    smd_frame = pd.DataFrame(smds)
    compact_vars = ['age', 'duration_hours', 'map_last', 'heart_rate_last', 'nee_last_5min', 'active_agents_last_5min', 'map_mean_6h']
    compact = long[long.variable.isin(compact_vars)].copy()
    compact['display'] = compact.apply(lambda r: ('NE' if pd.isna(r['median']) else f"{r['median']:.2f} [{r['q1']:.2f}, {r['q3']:.2f}]") + f"; missing {int(r['missing_n'])}/{int(r['group_n'])} ({r['missing_percent']:.1f}%)", axis=1)
    compact = compact[['database', 'variable', 'variable_label', 'unit', 'window', 'outcome_group', 'group_n', 'display']]
    return (long, smd_frame, compact)

def build_ablation_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    performance = pd.read_csv(EICU_RESULTS / 'all_performance.csv')
    ci = pd.read_csv(EICU_RESULTS / 'all_hospital_CI.csv')
    paired = pd.read_csv(EICU_RESULTS / 'all_paired_differences.csv')
    full_name = 'corrected_primary_source_raw'
    ablation_name = 'source_trained__corrected_primary__harmonized_dynamic_vitals'
    point = performance[(performance.variant == 'primary_traceable') & performance.model.isin([full_name, ablation_name])].copy()
    cis = ci[(ci.variant == 'primary_traceable') & ci.model.isin([full_name, ablation_name])].copy()
    pdiff = paired[(paired.variant == 'primary_traceable') & (paired.model == ablation_name) & (paired.reference == full_name)].copy()
    if len(point) != 2 or set(pdiff.metric) != {'auroc', 'auprc', 'brier', 'calibration_intercept', 'calibration_slope', 'mean_predicted_risk'}:
        raise AssertionError('Latest ablation results could not be uniquely identified')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        source_bundle = load_source_bundle(SOURCE_MODELS.parent, 'primary')
        ablation_bundle = load_ablation_bundle(ABLATION_MODELS.parent)
    full_features = list(source_bundle['features']['dynamic_multinomial_logistic'])
    ablation_features = list(ablation_bundle['source__corrected_primary__harmonized_dynamic_vitals'].feature_names_in_)
    feature_rows = []
    for feature in full_features:
        feature_rows.append({'feature': feature, 'in_full_source_model': True, 'in_dynamic_vitals_ablation': feature in ablation_features, 'difference': 'retained' if feature in ablation_features else 'omitted from ablation'})
    for feature in ablation_features:
        if feature not in full_features:
            feature_rows.append({'feature': feature, 'in_full_source_model': False, 'in_dynamic_vitals_ablation': True, 'difference': 'added in ablation'})
    feature_compare = pd.DataFrame(feature_rows)
    omitted = feature_compare.loc[feature_compare.difference.eq('omitted from ablation'), 'feature'].tolist()
    added = feature_compare.loc[feature_compare.difference.eq('added in ablation'), 'feature'].tolist()
    expected_omitted = {'nee_last_5min', 'active_agents_last_5min', 'nee_mean_1h', 'nee_max_1h', 'nee_mean_6h', 'nee_max_6h', 'nee_slope_toward_stop_6h', 'maximum_active_agents_6h', 'zero_nee_grid_pct_6h', 'nee_last_to_max_ratio'}
    feature_qa = {'full_feature_count': len(full_features), 'ablation_feature_count': len(ablation_features), 'omitted_features': omitted, 'added_features': added, 'only_difference_is_omission_of_dose_and_active_agent_derived_features': set(omitted) == expected_omitted and (not added)}
    if not feature_qa['only_difference_is_omission_of_dose_and_active_agent_derived_features']:
        raise AssertionError('Ablation feature-set contrast differed from the documented specification')
    return (point, cis, pdiff, feature_compare, feature_qa)

def make_time_figure(summary: pd.DataFrame, bins: pd.DataFrame) -> None:
    colors = ['#244E78', '#4C78A8']
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharey=True)
    for ax, database, color in zip(axes, ['MIMIC-IV', 'eICU'], colors):
        d = bins[bins.database == database]
        s = summary[summary.database == database].iloc[0]
        bars = ax.bar(d.interval_hours, d.percent_of_all_restart_first, color=color, edgecolor='white')
        for bar, n in zip(bars, d.events):
            height = bar.get_height()
            if height >= 8:
                ax.text(bar.get_x() + bar.get_width() / 2, height - 1.6, f'{int(n)}', ha='center', va='top', fontsize=9, color='white', fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.8, f'{int(n)}', ha='center', va='bottom', fontsize=9)
        ax.set_title(database, fontweight='bold')
        ax.set_xlabel('Hours from reconstructed cessation to first recorded restart')
        ax.grid(axis='y', alpha=0.25)
        ax.text(0.02, 0.98, f'Median {s.median_hours:.1f} h [Q1 {s.q1_hours:.1f}, Q3 {s.q3_hours:.1f}]\nValid {int(s.valid_time_n)}/{int(s.restart_first_total)}; missing/invalid {int(s.missing_or_invalid_time_n)}', transform=ax.transAxes, ha='left', va='top', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#BBBBBB', alpha=0.95))
    axes[0].set_ylabel('Percentage of all restart-first records')
    fig.suptitle('Timing of first recorded restart among restart-first records', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.01, 'Conditional event-time distribution; not cumulative incidence or risk among all cessation records.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    for ext in ('pdf', 'png', 'tiff'):
        fig.savefig(OUT / f'Figure_5_restart_time_distribution.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

def make_ablation_figure(point: pd.DataFrame, cis: pd.DataFrame, paired: pd.DataFrame) -> None:
    names = ['Full source', 'Dose/agent-free\ndynamic-vitals']
    models = ['corrected_primary_source_raw', 'source_trained__corrected_primary__harmonized_dynamic_vitals']
    colors = ['#A44A3F', '#3B7A57']
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2))
    metrics = [('mean_predicted_risk', 'Mean predicted risk', True), ('brier', 'Brier score', False), ('auroc', 'AUROC', False)]
    for ax, (metric, title, event_line) in zip(axes, metrics):
        vals, lows, highs = ([], [], [])
        for model in models:
            row = cis[(cis.model == model) & (cis.metric == metric)].iloc[0]
            vals.append(row.estimate)
            lows.append(row.ci_low)
            highs.append(row.ci_high)
        yerr = np.array([np.array(vals) - np.array(lows), np.array(highs) - np.array(vals)])
        ax.bar(names, vals, color=colors, width=0.65, yerr=yerr, capsize=4)
        if event_line:
            ax.axhline(0.3488593155893536, color='#222222', linestyle='--', linewidth=1.2, label='Observed event rate')
            ax.legend(frameon=False, fontsize=8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, min(1.0, max(highs) * 1.18))
        ax.grid(axis='y', alpha=0.25)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.025, f'{v:.3f}', ha='center', fontsize=9)
    diff = {r.metric: r for _, r in paired.iterrows()}
    fig.suptitle('Full MIMIC-IV source model versus ablation omitting dose and active-agent summaries', fontsize=11.5, fontweight='bold')
    fig.text(0.5, 0.01, f"Paired hospital bootstrap: Brier difference {diff['brier'].estimate:.3f} [{diff['brier'].ci_low:.3f}, {diff['brier'].ci_high:.3f}]; AUROC difference {diff['auroc'].estimate:.3f} [{diff['auroc'].ci_low:.3f}, {diff['auroc'].ci_high:.3f}].", ha='center', fontsize=8.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.91])
    for ext in ('pdf', 'png', 'tiff'):
        fig.savefig(OUT / f'Figure_6_source_ablation_contrast.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite: {OUT}')
    OUT.mkdir(parents=True)
    mimic, eicu, join_qa = load_inputs()
    frames = {'MIMIC-IV': mimic, 'eICU': eicu}
    time_summary, time_bins, time_qa = build_time_results(frames)
    characteristics, smds, compact = build_characteristics(frames)
    point, cis, paired, feature_compare, feature_qa = build_ablation_results()
    time_summary.to_csv(OUT / 'restart_time_summary.csv', index=False)
    time_bins.to_csv(OUT / 'restart_time_fixed_intervals.csv', index=False)
    characteristics.to_csv(OUT / 'outcome_clinical_characteristics_long.csv', index=False)
    smds.to_csv(OUT / 'outcome_clinical_standardized_differences.csv', index=False)
    compact.to_csv(OUT / 'main_clinical_characteristics_compact.csv', index=False)
    point.to_csv(OUT / 'source_ablation_performance_points.csv', index=False)
    cis.to_csv(OUT / 'source_ablation_hospital_cluster_CI.csv', index=False)
    paired.to_csv(OUT / 'source_ablation_paired_differences.csv', index=False)
    feature_compare.to_csv(OUT / 'source_ablation_exact_feature_comparison.csv', index=False)
    make_time_figure(time_summary, time_bins)
    make_ablation_figure(point, cis, paired)
    input_hashes = {str(path): sha256(path) for path in [MIMIC_COHORT, MIMIC_FEATURES, EICU_LABELS, EICU_FEATURES, SOURCE_MODELS, ABLATION_MODELS, RULES]}
    outcome_counts = {database: {k: int(v) for k, v in frame.outcome_group.value_counts().to_dict().items()} for database, frame in frames.items()}
    qa = {'analysis_scope': 'post hoc descriptive/exploratory; no model fitting or endpoint changes', 'input_hashes': input_hashes, 'join_QA': join_qa, 'outcome_counts': outcome_counts, 'restart_time_denominator_QA': time_qa, 'ablation_feature_QA': feature_qa, 'patient_level_outputs_written': False, 'models_refit_or_tuned': False, 'endpoint_rules_or_partitions_changed': False, 'all_checks_passed': all((x['interval_sum_equals_valid_time_n'] and x['valid_plus_missing_equals_restart_total'] for x in time_qa.values()))}
    (OUT / 'clinical_enhancement_QA.json').write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT / 'RESTART_TIME_DENOMINATOR_CHECK.json').write_text(json.dumps(time_qa, indent=2, ensure_ascii=False), encoding='utf-8')
    print(time_summary.to_string(index=False))
    print('\nFixed intervals\n', time_bins.to_string(index=False))
    print('\nLargest absolute SMDs\n', smds.assign(abs_smd=smds.standardized_difference.abs()).sort_values('abs_smd', ascending=False).head(20).to_string(index=False))
    print('\nAblation feature QA\n', json.dumps(feature_qa, indent=2))
    print(f'\nWrote aggregate outputs to {OUT}')
if __name__ == '__main__':
    main()
