"""Create aggregate-only eICU calibration, comparison, and heterogeneity displays."""
from __future__ import annotations
import os
import json
from pathlib import Path
import joblib
from model_from_json import source_bundle as load_source_bundle, FrozenJSONModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
RUN = Path(os.environ['EICU_RUN_ROOT']) / 'corrected'
OUT = Path(os.environ['DISPLAY_OUTPUT'])
LABELS = Path(os.environ['EICU_RUN_ROOT']) / 'corrected/private/corrected_labels.parquet'
FEATURES = Path(os.environ['EICU_RUN_ROOT']) / 'derived/stage2b_explicit_stop_feature_matrix.parquet'
SOURCE = Path(os.environ['MIMIC_RUN']) / 'models/parameters' / 'mimic_primary__dynamic_multinomial_logistic.json'
TARGET = Path(os.environ['EICU_RUN_ROOT']) / 'parameters/primary_traceable__target_native_refit.json'
RULES = Path(__file__).resolve().parent / '00_RESULT_DISPLAY_ENHANCEMENTS_FROZEN.md'
SEED = 20260831
BOOTSTRAPS = 1000

def predict(model, frame, features):
    return model.predict_proba(frame[features])[:, list(model.classes_).index(1)]

def metrics(y, p):
    return {'auroc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p), 'brier': brier_score_loss(y, p)}

def main() -> None:
    for path in (LABELS, FEATURES, SOURCE, TARGET, RULES):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUT.exists():
        raise SystemExit(f'Refusing to overwrite: {OUT}')
    OUT.mkdir(parents=True)
    restricted = OUT / 'restricted'
    restricted.mkdir()
    labels = pd.read_parquet(LABELS)
    selected = labels[(labels.variant == 'primary_traceable') & labels.included]
    features = pd.read_parquet(FEATURES)
    data = selected.merge(features, on='patienthealthsystemstayid', validate='one_to_one')
    train = data[data.partition == 'adaptation_development'].copy()
    evaluation = data[data.partition == 'locked_evaluation'].copy().reset_index(drop=True)
    y = (evaluation.new_code.to_numpy() == 1).astype(int)
    source_bundle = load_source_bundle(SOURCE.parent, 'primary')
    source_model = source_bundle['models']['dynamic_multinomial_logistic']
    source_features = source_bundle['features']['dynamic_multinomial_logistic']
    targets = {name: FrozenJSONModel(TARGET.parent / ('primary_traceable__'+name+'.json')) for name in ['target_native_refit','target_native__harmonized_last_values','target_native__harmonized_dynamic_vitals']}
    models = {'source_only': (source_model, source_features), 'target_native_full': (targets['target_native_refit'], list(targets['target_native_refit'].feature_names_in_)), 'target_native_last_values': (targets['target_native__harmonized_last_values'], list(targets['target_native__harmonized_last_values'].feature_names_in_)), 'target_native_dynamic_vitals': (targets['target_native__harmonized_dynamic_vitals'], list(targets['target_native__harmonized_dynamic_vitals'].feature_names_in_))}
    predictions = {name: predict(model, evaluation, fs) for name, (model, fs) in models.items()}
    fixed_risk = float((train.new_code.to_numpy() == 1).mean())
    predictions['fixed_adaptation_prevalence'] = np.full(len(evaluation), fixed_risk)
    point_rows = []
    for name, p in predictions.items():
        row = {'model': name, 'n': len(y), 'events': int(y.sum()), 'event_rate': float(y.mean()), 'mean_predicted_risk': float(p.mean())}
        if np.ptp(p) == 0:
            row.update({'auroc': 0.5, 'auprc': float(y.mean()), 'brier': brier_score_loss(y, p)})
        else:
            row.update(metrics(y, p))
        point_rows.append(row)
    pd.DataFrame(point_rows).to_csv(OUT / 'model_and_fixed_risk_descriptive_results.csv', index=False)
    hospitals = np.sort(evaluation.hospitalid.unique())
    groups = {h: np.flatnonzero(evaluation.hospitalid.to_numpy() == h) for h in hospitals}
    rng = np.random.default_rng(SEED)
    comparisons = [('target_native_full', 'source_only'), ('target_native_full', 'target_native_last_values'), ('target_native_dynamic_vitals', 'target_native_last_values')]
    boot = {pair: {m: [] for m in ('auroc', 'auprc', 'brier_improvement')} for pair in comparisons}
    invalid = {pair: 0 for pair in comparisons}
    for _ in range(BOOTSTRAPS):
        ix = np.concatenate([groups[h] for h in rng.choice(hospitals, len(hospitals), replace=True)])
        yy = y[ix]
        if len(np.unique(yy)) < 2:
            for pair in comparisons:
                invalid[pair] += 1
            continue
        for pair in comparisons:
            a = metrics(yy, predictions[pair[0]][ix])
            b = metrics(yy, predictions[pair[1]][ix])
            boot[pair]['auroc'].append(a['auroc'] - b['auroc'])
            boot[pair]['auprc'].append(a['auprc'] - b['auprc'])
            boot[pair]['brier_improvement'].append(b['brier'] - a['brier'])
    delta_rows = []
    for pair in comparisons:
        a = metrics(y, predictions[pair[0]])
        b = metrics(y, predictions[pair[1]])
        points = {'auroc': a['auroc'] - b['auroc'], 'auprc': a['auprc'] - b['auprc'], 'brier_improvement': b['brier'] - a['brier']}
        for metric, estimate in points.items():
            values = np.asarray(boot[pair][metric])
            delta_rows.append({'first_model': pair[0], 'reference_model': pair[1], 'metric': metric, 'estimate': estimate, 'ci_low': np.quantile(values, 0.025), 'ci_high': np.quantile(values, 0.975), 'valid_replicates': len(values), 'failed_single_class_replicates': invalid[pair]})
    pd.DataFrame(delta_rows).to_csv(OUT / 'direct_paired_model_comparisons.csv', index=False)
    calibration_rows = []
    for name in ('source_only', 'target_native_full'):
        temp = pd.DataFrame({'observed': y, 'predicted': predictions[name]})
        temp['group'] = pd.qcut(temp.predicted.rank(method='first'), 10, labels=False)
        agg = temp.groupby('group').agg(n=('observed', 'size'), events=('observed', 'sum'), mean_predicted=('predicted', 'mean'), observed_rate=('observed', 'mean')).reset_index()
        agg.insert(0, 'model', name)
        calibration_rows.append(agg)
    calibration = pd.concat(calibration_rows, ignore_index=True)
    calibration.to_csv(OUT / 'calibration_groups.csv', index=False)
    bins = np.linspace(0, 1, 21)
    hist_rows = []
    for name in ('source_only', 'target_native_full'):
        for outcome in (0, 1):
            counts, edges = np.histogram(predictions[name][y == outcome], bins=bins)
            for left, right, count in zip(edges[:-1], edges[1:], counts):
                hist_rows.append({'model': name, 'restart_first': outcome, 'bin_left': left, 'bin_right': right, 'count': int(count)})
    pd.DataFrame(hist_rows).to_csv(OUT / 'risk_distribution_histogram.csv', index=False)
    hospital_rows = []
    for h, ix in groups.items():
        yy = y[ix]
        row = {'hospitalid': h, 'n': len(ix), 'events': int(yy.sum()), 'event_rate': float(yy.mean()), 'source_mean_risk': float(predictions['source_only'][ix].mean()), 'target_mean_risk': float(predictions['target_native_full'][ix].mean())}
        if len(np.unique(yy)) == 2:
            row['source_auroc'] = roc_auc_score(yy, predictions['source_only'][ix])
            row['target_auroc'] = roc_auc_score(yy, predictions['target_native_full'][ix])
        hospital_rows.append(row)
    hospital = pd.DataFrame(hospital_rows)
    hospital.to_parquet(restricted / 'hospital_level_aggregates_with_database_ids.parquet', index=False)
    summary_rows = []
    for column in ['n', 'events', 'event_rate', 'source_mean_risk', 'target_mean_risk', 'source_auroc', 'target_auroc']:
        values = hospital[column].dropna()
        summary_rows.append({'measure': column, 'hospitals_with_value': len(values), 'minimum': values.min(), 'q1': values.quantile(0.25), 'median': values.median(), 'q3': values.quantile(0.75), 'maximum': values.max()})
    pd.DataFrame(summary_rows).to_csv(OUT / 'hospital_heterogeneity_summary.csv', index=False)
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color='#777777', linewidth=1, linestyle='--', label='Ideal')
    colors = {'source_only': '#C44E52', 'target_native_full': '#4C72B0'}
    labels_map = {'source_only': 'MIMIC source-only', 'target_native_full': 'eICU target-native'}
    for name in colors:
        g = calibration[calibration.model == name]
        ax.plot(g.mean_predicted, g.observed_rate, marker='o', markersize=3.5, linewidth=1.4, color=colors[name], label=labels_map[name])
    ax.set(xlabel='Mean predicted probability', ylabel='Observed restart-first proportion', xlim=(0, 1), ylim=(0, 1), title='A  Calibration by risk group')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.15)
    ax = axes[1]
    ax.hist(predictions['source_only'], bins=bins, alpha=0.55, color=colors['source_only'], label=labels_map['source_only'])
    ax.hist(predictions['target_native_full'], bins=bins, alpha=0.55, color=colors['target_native_full'], label=labels_map['target_native_full'])
    ax.set(xlabel='Predicted probability', ylabel='Records', xlim=(0, 1), title='B  Prediction distributions')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=0.15)
    for ext in ('pdf', 'png', 'tiff'):
        fig.savefig(OUT / f'Figure_3_calibration_and_risk_distribution.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'observed_restart_first': y, **predictions}).to_parquet(restricted / 'locked_predictions.parquet', index=False)
    qa = {'locked_n': len(y), 'locked_events': int(y.sum()), 'hospitals': len(hospitals), 'adaptation_n': len(train), 'adaptation_restart_first_rate_used_for_fixed_comparator': fixed_risk, 'bootstrap_replicates': BOOTSTRAPS, 'bootstrap_unit': 'hospital', 'seed': SEED, 'patient_level_outputs_restricted': True, 'patient_level_rows_in_public_csv': 0}
    (OUT / 'enhancement_QA.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))
if __name__ == '__main__':
    main()
