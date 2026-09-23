from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
SEED = 20260825
BOOTSTRAPS = 2000
THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
CLASS_NAMES = {0: 'durable_liberation_alive', 1: 'restart_first', 2: 'death_first'}
COMPARISONS = [('dynamic_multinomial_logistic', 'clinical_multinomial_logistic'), ('dynamic_multiclass_hgb', 'dynamic_multinomial_logistic'), ('laboratory_extended_multiclass_hgb', 'dynamic_multiclass_hgb')]

def sql_path(path: Path) -> str:
    return str(path).replace('\\', '/').replace("'", "''")

def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {'auroc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p), 'brier': float(np.mean((y - p) ** 2))}

def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(p, 1e-06, 1 - 1e-06)
    logit_p = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    fit = LogisticRegression(C=1000000.0, solver='lbfgs', max_iter=2000)
    fit.fit(logit_p, y)
    return (float(fit.intercept_[0]), float(fit.coef_[0, 0]))

def patient_bootstrap_indices(frame: pd.DataFrame) -> list[np.ndarray]:
    groups = {subject: indices.to_numpy() for subject, indices in frame.groupby('subject_id').groups.items()}
    subjects = np.asarray(list(groups))
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        samples.append(np.concatenate([groups[subject] for subject in sampled]))
    return samples

def fit_and_evaluate(name: str, data: pd.DataFrame, model_templates: dict, feature_map: dict[str, list[str]]) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    development = data[data.corrected_analysis_split == 'development'].copy()
    validation = data[data.corrected_analysis_split == 'temporal_validation'].copy()
    development = development.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    y_train = development.corrected_outcome_code.astype(int)
    y_test = validation.corrected_outcome_code.astype(int).to_numpy()
    boot_indices = patient_bootstrap_indices(validation)
    fitted = {}
    probabilities = {}
    result_rows = []
    for model_name, template in model_templates.items():
        model = clone(template)
        features = feature_map[model_name]
        model.fit(development[features], y_train)
        raw = model.predict_proba(validation[features])
        classes = list(model.classes_)
        probability = np.zeros((len(validation), 3), dtype=float)
        for position, class_value in enumerate(classes):
            probability[:, int(class_value)] = raw[:, position]
        fitted[model_name] = model
        probabilities[model_name] = probability
        multiclass_loss = log_loss(y_test, probability, labels=[0, 1, 2])
        one_hot = np.eye(3)[y_test]
        multiclass_brier = float(np.mean(np.sum((one_hot - probability) ** 2, axis=1)))
        for code, class_name in CLASS_NAMES.items():
            observed = (y_test == code).astype(int)
            predicted = probability[:, code]
            metric = binary_metrics(observed, predicted)
            intercept, slope = calibration(observed, predicted)
            result_rows.append({'analysis': name, 'model': model_name, 'outcome': class_name, 'training_n': len(development), 'training_subjects': development.subject_id.nunique(), 'test_n': len(validation), 'test_subjects': validation.subject_id.nunique(), 'test_events': int(observed.sum()), 'prevalence': float(observed.mean()), **metric, 'calibration_intercept': intercept, 'calibration_slope': slope, 'multiclass_log_loss': multiclass_loss, 'multiclass_brier': multiclass_brier})
    results = pd.DataFrame(result_rows)
    ci_rows = []
    for model_name, probability in probabilities.items():
        for code, class_name in CLASS_NAMES.items():
            observed = (y_test == code).astype(int)
            point = binary_metrics(observed, probability[:, code])
            values = {metric: [] for metric in ('auroc', 'auprc', 'brier')}
            for indices in boot_indices:
                metric = binary_metrics(observed[indices], probability[indices, code])
                for metric_name in values:
                    values[metric_name].append(metric[metric_name])
            for metric_name, estimate in point.items():
                distribution = np.asarray(values[metric_name])
                ci_rows.append({'analysis': name, 'model': model_name, 'outcome': class_name, 'metric': metric_name, 'estimate': estimate, 'ci_low': np.quantile(distribution, 0.025), 'ci_high': np.quantile(distribution, 0.975)})
    cis = pd.DataFrame(ci_rows)
    restart_observed = (y_test == 1).astype(int)
    difference_rows = []
    for candidate, reference in COMPARISONS:
        candidate_p = probabilities[candidate][:, 1]
        reference_p = probabilities[reference][:, 1]
        candidate_point = binary_metrics(restart_observed, candidate_p)
        reference_point = binary_metrics(restart_observed, reference_p)
        boot_values = {metric: [] for metric in ('auroc', 'auprc', 'brier_improvement')}
        for indices in boot_indices:
            c = binary_metrics(restart_observed[indices], candidate_p[indices])
            r = binary_metrics(restart_observed[indices], reference_p[indices])
            boot_values['auroc'].append(c['auroc'] - r['auroc'])
            boot_values['auprc'].append(c['auprc'] - r['auprc'])
            boot_values['brier_improvement'].append(r['brier'] - c['brier'])
        points = {'auroc': candidate_point['auroc'] - reference_point['auroc'], 'auprc': candidate_point['auprc'] - reference_point['auprc'], 'brier_improvement': reference_point['brier'] - candidate_point['brier']}
        for metric_name, estimate in points.items():
            distribution = np.asarray(boot_values[metric_name])
            difference_rows.append({'analysis': name, 'comparison': f'{candidate} minus {reference}', 'metric': metric_name, 'estimate': estimate, 'ci_low': np.quantile(distribution, 0.025), 'ci_high': np.quantile(distribution, 0.975)})
    differences = pd.DataFrame(difference_rows)
    dca_rows = []
    n = len(validation)
    prevalence = restart_observed.mean()
    for threshold in THRESHOLDS:
        weight = threshold / (1 - threshold)
        dca_rows.extend([{'analysis': name, 'threshold': threshold, 'model': 'treat_all', 'net_benefit': prevalence - (1 - prevalence) * weight}, {'analysis': name, 'threshold': threshold, 'model': 'treat_none', 'net_benefit': 0.0}])
        for model_name, probability in probabilities.items():
            predicted = probability[:, 1]
            treated = predicted >= threshold
            tp = np.sum(treated & (restart_observed == 1))
            fp = np.sum(treated & (restart_observed == 0))
            dca_rows.append({'analysis': name, 'threshold': threshold, 'model': model_name, 'net_benefit': tp / n - fp / n * weight})
    dca = pd.DataFrame(dca_rows)
    return (fitted, results, cis, differences, dca)
