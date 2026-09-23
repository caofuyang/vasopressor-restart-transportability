"""Predict using readable fitted parameters; no sklearn or joblib dependency.

Input columns must retain the raw feature representation and missing values.
This module exports no patient information. Callers control private output paths.
"""
import numpy as np
import pandas as pd

def transform(frame, spec):
    pieces=[]
    for block in spec['preprocessing']:
        x=frame[block['columns']].to_numpy(copy=True)
        if block['kind']=='numeric':
            x[pd.isna(x)]=np.nan
            x=x.astype(float);missing=np.isnan(x)
            stats=np.array(block['imputation_statistics'],dtype=float)
            x=np.where(missing,stats,x)
            x=x[:,block['retained_columns']]
            if block['indicator_columns']:
                x=np.column_stack([x,missing[:,block['indicator_columns']].astype(float)])
            if block.get('scale') is not None:
                x=(x-np.array(block['mean']))/np.array(block['scale'])
        else:
            missing=pd.isna(x)
            for j,stat in enumerate(block['imputation_statistics']):x[missing[:,j],j]=stat
            encoded=[]
            for j,cats in enumerate(block['categories']):
                for k,cat in enumerate(cats):
                    if block['drop_indices'][j]==k:continue
                    encoded.append((x[:,j]==cat).astype(float))
            x=np.column_stack(encoded)
        pieces.append(x)
    return np.column_stack(pieces)

def predict_proba(frame,spec):
    x=transform(frame,spec);model=spec['estimator']
    if model['kind']=='logistic':
        score=x@np.array(model['coefficients']).T+np.array(model['intercepts'])
    elif model['kind']=='hist_gradient_boosting':
        score=np.tile(model['baseline_prediction'],(len(x),1)).astype(float)
        for iteration in model['trees']:
            for k,tree in enumerate(iteration):
                nodes=tree['nodes']
                for row in range(len(x)):
                    i=0
                    while not nodes[i]['is_leaf']:
                        node=nodes[i];v=x[row,node['feature_idx']]
                        left=bool(node['missing_go_to_left']) if np.isnan(v) else v<=node['num_threshold']
                        i=node['left'] if left else node['right']
                    score[row,k]+=nodes[i]['value']
    else:raise ValueError(model['kind'])
    if score.shape[1]==1:
        p=1/(1+np.exp(-np.clip(score[:,0],-709,709)));return np.column_stack([1-p,p])
    score-=score.max(axis=1,keepdims=True);p=np.exp(score);return p/p.sum(axis=1,keepdims=True)

def recalibrate(p,spec):
    p=np.clip(p,spec['clip'],1-spec['clip'])
    z=spec['intercept']+spec['slope']*np.log(p/(1-p))
    return 1/(1+np.exp(-np.clip(z,-709,709)))
