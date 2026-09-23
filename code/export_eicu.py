"""Export newly fitted target/recalibration models and actually replay their predictions."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from export_parameters import export,safe
from portable_predict import predict_proba,recalibrate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--source-parameters',type=Path,required=True);args=ap.parse_args()
 run=args.run_dir.resolve();out=run/'parameters';out.mkdir(exist_ok=False)
 labels=pd.read_parquet(run/'corrected/private/corrected_labels.parquet');features=pd.read_parquet(run/'derived/stage2b_explicit_stop_feature_matrix.parquet');checks=[]
 for variant in ['primary_traceable','freshness_180','gap_60']:
  evaluation=labels[(labels.variant==variant)&labels.included&(labels.partition=='locked_evaluation')].merge(features,on='patienthealthsystemstayid',validate='one_to_one').sort_values('patienthealthsystemstayid',kind='stable')
  path=run/f'corrected/private/{variant}_fitted_target_models.joblib';models=joblib.load(path)
  for name,model in models.items():
   if hasattr(model,'named_steps'):
    spec=export(model);original=model.predict_proba(evaluation[spec['ordered_input_features']]);rebuilt=predict_proba(evaluation,spec)
   else:
    source=name.split('_')[0];sourcefile=args.source_parameters/f'mimic_{source}__dynamic_multinomial_logistic.json';base=json.loads(sourcefile.read_text());p=predict_proba(evaluation,base)[:,base['estimator']['classes'].index(1)];lp=np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))
    spec={'kind':'recalibration','clip':1e-7,'base_parameter_file':sourcefile.name,'classes':[0,1]}
    if isinstance(model,float):spec.update(intercept=model,slope=1.0);original=expit(lp+model)
    else:spec.update(intercept=float(model.intercept_[0]),slope=float(model.coef_[0,0]),parameters=safe(model.get_params()));original=model.predict_proba(lp.reshape(-1,1))[:,1]
    rebuilt=recalibrate(p,spec)
   spec['provenance']={'version':'deterministic_20260910','model_key':name,'fitted_artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training_sort':['patienthealthsystemstayid']}
   key=f'{variant}__{name}';(out/f'{key}.json').write_text(json.dumps(safe(spec),indent=2,allow_nan=False))
   delta=float(np.max(np.abs(original-rebuilt)));checks.append(dict(model=key,n=len(evaluation),max_absolute_prediction_difference=delta,tolerance=1e-10,judgement='match' if delta<=1e-10 else 'DIFFERENCE'))
 pd.DataFrame(checks).to_csv(run/'corrected/results/fixed_model_json_replay.csv',index=False)
 assert all(x['judgement']=='match' for x in checks)
 print('Exported and replayed',len(checks),'new target/recalibration models.',flush=True)
if __name__=='__main__':main()
