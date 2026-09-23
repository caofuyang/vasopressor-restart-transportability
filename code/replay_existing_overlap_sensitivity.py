import os
from model_from_json import source_bundle
from export_parameters import export,safe
from portable_predict import predict_proba,recalibrate
"""Post hoc eICU sensitivity excluding patients appearing in >1 hospital partition.

This does not alter the frozen primary analysis. Models are refit only within the
same fixed hospital partitions after removing all records from overlapping patients.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
OUT = Path(os.environ['OVERLAP_OUTPUT'])
LAB = Path(os.environ['EICU_RUN_ROOT']) / 'corrected/private/corrected_labels.parquet'
FEAT = Path(os.environ['EICU_RUN_ROOT']) / 'derived/stage2b_explicit_stop_feature_matrix.parquet'
MIMIC = Path(os.environ['REPRO_PARAMETERS']) / 'mimic_primary__dynamic_multinomial_logistic.json'

def logit(p):
    p = np.clip(p, 1e-07, 1 - 1e-07)
    return np.log(p / (1 - p))

def met(y, p):
    return {'auroc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p), 'brier': brier_score_loss(y, p), 'mean_predicted_risk': p.mean()}

def pred(model, x, fs):
    return model.predict_proba(x[fs])[:, list(model.classes_).index(1)]

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    l = pd.read_parquet(LAB)
    l = l[l.variant == 'primary_traceable']
    f = pd.read_parquet(FEAT)
    d = l.merge(f, on='patienthealthsystemstayid', validate='one_to_one')
    parts = d.groupby('uniquepid').partition.nunique()
    overlapping = set(parts[parts > 1].index)
    primary = d[d.included].copy()
    removed = primary[primary.uniquepid.isin(overlapping)].groupby('partition').agg(records=('uniquepid', 'size'), patients=('uniquepid', 'nunique'), restart_events=('new_code', lambda x: (x == 1).sum())).reset_index()
    d = primary[~primary.uniquepid.isin(overlapping)].copy().sort_values('patienthealthsystemstayid',kind='stable').reset_index(drop=True)
    tr = d[d.partition == 'adaptation_development']
    rc = d[d.partition == 'recalibration']
    ev = d[d.partition == 'locked_evaluation']
    bundle = source_bundle(MIMIC.parent,'primary')
    source = bundle['models']['dynamic_multinomial_logistic']
    fs = bundle['features']['dynamic_multinomial_logistic']
    y = ev.new_code.eq(1).astype(int).to_numpy()
    yr = rc.new_code.eq(1).astype(int).to_numpy()
    raw = pred(source, ev, fs)
    rp = pred(source, rc, fs)
    fun = lambda b: float(np.logaddexp(0, logit(rp) + b).sum() - yr @ (logit(rp) + b))
    intercept = minimize_scalar(fun, bounds=(-10, 10), method='bounded').x
    recal_intercept = expit(logit(raw) + intercept)
    lr = LogisticRegression(C=1000000.0, max_iter=5000).fit(logit(rp).reshape(-1, 1), yr)
    recal_full = lr.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
    target = clone(source).fit(tr[fs], tr.new_code.astype(int))
    targetp = pred(target, ev, fs)
    preds = {'source_raw': raw, 'source_intercept_recalibrated': recal_intercept, 'source_logistic_recalibrated': recal_full, 'target_native_refit': targetp}
    parameter_dir=OUT/'parameters';parameter_dir.mkdir()
    specs={'target_native_refit':export(target),'source_intercept_recalibrated':{'kind':'recalibration','intercept':float(intercept),'slope':1.0,'clip':1e-7,'base_parameter_file':MIMIC.name},'source_logistic_recalibrated':{'kind':'recalibration','intercept':float(lr.intercept_[0]),'slope':float(lr.coef_[0,0]),'clip':1e-7,'base_parameter_file':MIMIC.name,'parameters':safe(lr.get_params()),'classes':safe(lr.classes_)}}
    checks=[]
    for name,spec in specs.items():
        (parameter_dir/(name+'.json')).write_text(json.dumps(safe(spec),indent=2,allow_nan=False))
        rebuilt=recalibrate(raw,spec) if spec.get('kind')=='recalibration' else predict_proba(ev,spec)[:,1]
        delta=float(np.max(np.abs(rebuilt-preds[name])))
        checks.append({'model':name,'max_absolute_prediction_difference':delta,'tolerance':1e-10,'judgement':'match' if delta<=1e-10 else 'DIFFERENCE'})
    pd.DataFrame(checks).to_csv(OUT/'fixed_model_json_replay.csv',index=False)
    assert all(c['judgement']=='match' for c in checks)
    points = []
    for name, p in preds.items():
        points.append({'analysis': 'post_hoc_exclude_cross_partition_patients', 'model': name, 'n': len(ev), 'patients': ev.uniquepid.nunique(), 'hospitals': ev.hospitalid.nunique(), 'events': int(y.sum()), **met(y, p)})
    pd.DataFrame(points).to_csv(OUT / 'posthoc_overlap_exclusion_sensitivity.csv', index=False, encoding='utf-8-sig')
    hospitals = np.sort(ev.hospitalid.unique())
    groups = {h: np.flatnonzero(ev.hospitalid.to_numpy() == h) for h in hospitals}
    rng = np.random.default_rng(20260907)
    vals = {k: {m: [] for m in ['auroc', 'auprc', 'brier']} for k in preds}
    for _ in range(1000):
        ix = np.concatenate([groups[h] for h in rng.choice(hospitals, len(hospitals), replace=True)])
        yy = y[ix]
        if len(np.unique(yy)) < 2:
            continue
        for name, p in preds.items():
            z = met(yy, p[ix])
            for m in vals[name]:
                vals[name][m].append(z[m])
    ci = []
    for row in points:
        for m, v in vals[row['model']].items():
            ci.append({'analysis': row['analysis'], 'model': row['model'], 'metric': m, 'estimate': row[m], 'ci_low': np.quantile(v, 0.025), 'ci_high': np.quantile(v, 0.975), 'valid_replicates': len(v), 'attempted_replicates': 1000, 'resampling_unit': 'hospitalid'})
    pd.DataFrame(ci).to_csv(OUT / 'posthoc_overlap_exclusion_hospital_bootstrap_CI.csv', index=False, encoding='utf-8-sig')
    qa = {'overlapping_patients': len(overlapping), 'removed_by_partition': removed.to_dict('records'), 'remaining': d.partition.value_counts().to_dict(), 'locked_hospitals': int(ev.hospitalid.nunique()), 'rule': 'remove every record of any uniquepid appearing in more than one frozen hospital partition; retain hospital assignments; refit using original specifications', 'status': 'reproduction of the existing post hoc sensitivity; frozen primary analysis unchanged'}
    (OUT / 'posthoc_overlap_exclusion_QA.json').write_text(json.dumps(qa, indent=2, default=str), encoding='utf-8')
    print(json.dumps(qa, indent=2))
if __name__ == '__main__':
    main()
