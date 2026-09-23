from pathlib import Path
import json, hashlib, sys, platform, importlib.metadata
import joblib, numpy as np

def safe(v):
    if isinstance(v, np.ndarray):
        return safe(v.tolist())
    if isinstance(v, np.generic):
        return safe(v.item())
    if isinstance(v, float) and (not np.isfinite(v)):
        return None
    if isinstance(v, dict):
        return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [safe(x) for x in v]
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    return repr(v)

def export(pipe):
    pre = pipe.named_steps['preprocess']
    blocks = []
    for name, t, cols in pre.transformers_:
        imp = t.named_steps['imputer']
        b = {'columns': list(cols), 'imputation_strategy': imp.strategy, 'imputation_statistics': safe(imp.statistics_), 'imputer_parameters': safe(imp.get_params())}
        if name == 'numeric':
            b.update(kind='numeric', retained_columns=np.flatnonzero(~np.isnan(imp.statistics_)).tolist(), indicator_columns=imp.indicator_.features_.tolist() if imp.add_indicator else [], mean=None, scale=None)
            if 'scaler' in t.named_steps:
                sc = t.named_steps['scaler']
                b.update(mean=safe(sc.mean_), scale=safe(sc.scale_), variance=safe(sc.var_), samples_seen=safe(sc.n_samples_seen_))
        else:
            oh = t.named_steps['onehot']
            b.update(kind='categorical', categories=safe(oh.categories_), handle_unknown=oh.handle_unknown, drop_indices=[None] * len(cols) if oh.drop_idx_ is None else safe(oh.drop_idx_))
        blocks.append(b)
    model = pipe.named_steps['model']
    m = {'parameters': safe(model.get_params()), 'classes': safe(model.classes_)}
    if hasattr(model, 'coef_'):
        m.update(kind='logistic', coefficients=safe(model.coef_), intercepts=safe(model.intercept_), iterations=safe(model.n_iter_))
    else:
        trees = []
        for iteration in model._predictors:
            ts = []
            for tree in iteration:
                assert not tree.nodes['is_categorical'].any()
                ts.append({'nodes': [{k: safe(row[k]) for k in tree.nodes.dtype.names} for row in tree.nodes]})
            trees.append(ts)
        m.update(kind='hist_gradient_boosting', baseline_prediction=safe(model._baseline_prediction.reshape(-1)), trees=trees, iterations=model.n_iter_)
    return {'format_version': 1, 'ordered_input_features': list(pipe.feature_names_in_), 'ordered_transformed_features': list(pre.get_feature_names_out()), 'preprocessing': blocks, 'estimator': m, 'class_meanings': {'0': 'no recorded event under specified observation', '1': 'restart first', '2': 'death first / administrative death proxy first'}}
