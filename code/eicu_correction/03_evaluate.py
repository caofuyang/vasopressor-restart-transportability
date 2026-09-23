"""Fixed-source eICU evaluation and target updating with hospital-cluster CIs."""
import json
from datetime import datetime,timezone
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from common import *
from model_from_json import source_bundle,ablation_bundle

SEED=20260831
B=1000
METRICS=['auroc','auprc','brier','calibration_intercept','calibration_slope','mean_predicted_risk']
def logit(p):
    p=np.clip(p,1e-7,1-1e-7)
    return np.log(p/(1-p))
def cal(y,p):
    """Two-parameter penalized logistic calibration via damped Newton.
    Same slope penalty 1/C=1e-6 as historical C=1e6 sklearn model.
    Return NA for absent outcome classes, separation/failed convergence.
    """
    if len(np.unique(y))<2: return np.nan,np.nan
    x=logit(p); X=np.column_stack([np.ones(len(y)),x])
    if np.ptp(x)<1e-12: return np.nan,np.nan
    beta=np.array([float(logit(np.array([y.mean()]))[0]),0.])
    def loss(b): return np.logaddexp(0,X@b).sum()-y@(X@b)+.5e-6*b[1]**2
    for _ in range(60):
        z=X@beta; pr=expit(z); w=pr*(1-pr)
        g=X.T@(pr-y)+np.array([0,1e-6*beta[1]])
        if np.max(np.abs(g))<1e-6: return tuple(beta)
        H=(X.T*w)@X+np.diag([1e-10,1e-6])
        try: step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError: return np.nan,np.nan
        scale=1.; old=loss(beta)
        while scale>1e-8 and loss(beta-scale*step)>old: scale*=.5
        beta-=scale*step
        if not np.isfinite(beta).all() or np.max(abs(beta))>1e4: return np.nan,np.nan
    return np.nan,np.nan
def metrics(y,p):
    c0,c1=cal(y,p)
    return dict(auroc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
       auprc=average_precision_score(y,p) if y.sum()>0 else np.nan,
       brier=brier_score_loss(y,p),calibration_intercept=c0,calibration_slope=c1,
       mean_predicted_risk=float(p.mean()))
def predict(model,frame,features):
    return model.predict_proba(frame[features])[:,list(model.classes_).index(1)]

def main():
    check_rules()
    assert json.loads((RESULTS/'pre_performance_QA.json').read_text())['all_passed']
    write_json(RESULTS/'performance_start.json',{'utc':datetime.now(timezone.utc).isoformat(),'rules_sha256':RULE_SHA,'bootstrap':{'unit':'hospital','replicates':B,'seed':SEED,'conditional_on_fitted_models':True}})
    labs=pd.read_parquet(PRIVATE/'corrected_labels.parquet')
    f=pd.read_parquet(DERIVED/'stage2b_explicit_stop_feature_matrix.parquet')
    bundles={name:source_bundle(PARAMETERS,name) for name in ['primary','strict']}
    ablations=ablation_bundle(PARAMETERS)
    features=bundles['primary']['features']['dynamic_multinomial_logistic']
    base_model=bundles['primary']['models']['dynamic_multinomial_logistic']
    allpoints=[]; allcis=[]; alldeltas=[]; fitting=[]; qc=[]
    for variant in VARIANTS:
        print('START',variant,flush=True)
        selected=labs[(labs.variant==variant)&labs.included].copy()
        data=selected.merge(f,on='patienthealthsystemstayid',validate='one_to_one')
        data=data.sort_values('patienthealthsystemstayid',kind='stable').reset_index(drop=True)
        assert (data.origin==data.episode_end_minute).all()
        frames={p:data[data.partition==p].copy() for p in ['adaptation_development','recalibration','locked_evaluation']}
        train,recal,evaluation=(frames[p] for p in frames)
        y=(evaluation.new_code.to_numpy()==1).astype(int)
        yr=(recal.new_code.to_numpy()==1).astype(int)
        preds={}; fitted={}; missing={}
        for name,bundle in bundles.items():
            model=bundle['models']['dynamic_multinomial_logistic']
            preds[f'corrected_{name}_source_raw']=predict(model,evaluation,features)
            rp=predict(model,recal,features)
            lp=logit(rp)
            if len(np.unique(yr))<2:
                missing[f'corrected_{name}_source_intercept_recalibrated']='recalibration lacks both binary classes'
                missing[f'corrected_{name}_source_logistic_recalibrated']='recalibration lacks both binary classes'
                continue
            fun=lambda b: float(np.logaddexp(0,lp+b).sum()-yr@(lp+b))
            opt=minimize_scalar(fun,bounds=(-10,10),method='bounded')
            assert opt.success
            full=LogisticRegression(C=1e6,max_iter=5000).fit(lp.reshape(-1,1),yr)
            preds[f'corrected_{name}_source_intercept_recalibrated']=expit(logit(preds[f'corrected_{name}_source_raw'])+opt.x)
            preds[f'corrected_{name}_source_logistic_recalibrated']=full.predict_proba(logit(preds[f'corrected_{name}_source_raw']).reshape(-1,1))[:,1]
            fitted[name+'_intercept']=float(opt.x); fitted[name+'_logistic']=full
        for name,model in [('target_native_refit',base_model)]+[(f'target_native__{fs}',ablations[f'source__corrected_primary__{fs}']) for fs in ['harmonized_last_values','harmonized_dynamic_vitals']]:
            fs=list(model.feature_names_in_)
            if set(train.new_code.unique())!={0,1,2}:
                missing[name]='adaptation lacks one or more of the three outcome classes'; continue
            target=clone(model)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                target.fit(train[fs],train.new_code.astype(int))
            fitting.append({'variant':variant,'model':name,'n_fit':len(train),'partition':'adaptation_development','warnings':' | '.join(str(w.message) for w in caught)})
            preds[name]=predict(target,evaluation,fs); fitted[name]=target
            assert repr(target.get_params())==repr(clone(model).get_params())
        for name,model in ablations.items():
            if not name.startswith('source__'): continue
            preds[name.replace('source__','source_trained__',1)]=predict(model,evaluation,list(model.feature_names_in_))
        points=[]
        for name,p in preds.items():
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            row=dict(variant=variant,model=name,n=len(y),hospitals=evaluation.hospitalid.nunique(),events=int(y.sum()),prevalence=float(y.mean()),status='estimated',**metrics(y,p))
            points.append(row)
            # Independent sklearn calibration spot-check at point estimates.
            sk=LogisticRegression(C=1e6,max_iter=5000,tol=1e-10).fit(logit(p).reshape(-1,1),y)
            err=max(abs(row['calibration_intercept']-sk.intercept_[0]),abs(row['calibration_slope']-sk.coef_[0,0]))
            qc.append(dict(variant=variant,model=name,calibration_max_abs_difference=float(err)))
            assert err<.02 or not np.isfinite(err)
        for name,reason in missing.items():
            points.append(dict(variant=variant,model=name,n=len(y),hospitals=evaluation.hospitalid.nunique(),events=int(y.sum()),prevalence=float(y.mean()),status=reason,**{m:np.nan for m in METRICS}))
        # Shared cluster draw for every model and all paired comparisons.
        hospitals=np.sort(evaluation.hospitalid.unique())
        groups={h:np.flatnonzero(evaluation.hospitalid.to_numpy()==h) for h in hospitals}
        rng=np.random.default_rng(SEED)
        boot={name:[] for name in preds}
        for b in range(B):
            ix=np.concatenate([groups[h] for h in rng.choice(hospitals,len(hospitals),replace=True)])
            yy=y[ix]
            for name,p in preds.items(): boot[name].append(metrics(yy,p[ix]))
            if (b+1)%200==0: print(variant,'hospital bootstrap',b+1,'/',B,flush=True)
        cis=[]; delta=[]
        base='corrected_primary_source_raw'
        baseboot=pd.DataFrame(boot[base])
        for row in points:
            name=row['model']
            if name not in boot: continue
            bt=pd.DataFrame(boot[name])
            for metric in METRICS:
                values=bt[metric].dropna()
                cis.append(dict(variant=variant,model=name,metric=metric,estimate=row[metric],ci_low=values.quantile(.025),ci_high=values.quantile(.975),valid_replicates=len(values),attempted_replicates=B))
                if name!=base:
                    vals=(bt[metric]-baseboot[metric]).dropna()
                    baseline=next(x[metric] for x in points if x['model']==base)
                    delta.append(dict(variant=variant,model=name,reference=base,metric=metric,estimate=row[metric]-baseline,ci_low=vals.quantile(.025),ci_high=vals.quantile(.975),valid_replicates=len(vals)))
        csv(f'{variant}_performance.csv',pd.DataFrame(points))
        csv(f'{variant}_hospital_CI.csv',pd.DataFrame(cis))
        csv(f'{variant}_paired_differences.csv',pd.DataFrame(delta))
        joblib.dump(fitted,PRIVATE/f'{variant}_fitted_target_models.joblib')
        allpoints+=points; allcis+=cis; alldeltas+=delta
        print('FINISHED',variant,'n_eval',len(y),flush=True)
    points=pd.DataFrame(allpoints)
    csv('all_performance.csv',points); csv('all_hospital_CI.csv',pd.DataFrame(allcis)); csv('all_paired_differences.csv',pd.DataFrame(alldeltas))
    csv('fit_log.csv',pd.DataFrame(fitting)); csv('calibration_independent_QA.csv',pd.DataFrame(qc))
    # Historical before/after tables belong to provenance, not this independent execution.
    assert snapshot()==json.loads((RESULTS/'input_snapshot_before.json').read_text())
    write_json(RESULTS/'performance_complete.json',{'utc':datetime.now(timezone.utc).isoformat(),'variants':list(VARIANTS),'rules_sha256':RULE_SHA,'inputs_unchanged':True,'models_not_selected_by_performance':True})
    print('ALL FIXED ANALYSES COMPLETED',flush=True)
if __name__=='__main__': main()
