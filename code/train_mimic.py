"""Fit frozen architectures to independently regenerated, deterministically sorted data."""
from pathlib import Path
import argparse,json,hashlib,sys,platform,importlib.metadata
import duckdb,joblib,numpy as np,pandas as pd
from model_from_json import build_unfitted
from portable_predict import predict_proba
from export_parameters import export,safe
import mimic_evaluation as evaluator

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args()
 run=args.run_dir.resolve();package=Path(__file__).resolve().parent.parent
 out=run/'models';out.mkdir(exist_ok=False);params=out/'parameters';params.mkdir();results=out/'results';results.mkdir()
 f=run/'derived/vasopressor_liberation_features_stage7.parquet';c=run/'scope_corrected/restricted/stage18_post_audit_scope_corrected_cohort.parquet'
 con=duckdb.connect(config={'threads':1});data=con.execute(f"SELECT f.* EXCLUDE(restart_24h, death_24h, failure_24h, next_episode_start), c.* EXCLUDE(subject_id,hadm_id,stay_id) FROM read_parquet('{f.as_posix()}') f INNER JOIN read_parquet('{c.as_posix()}') c USING(stay_id)").df();con.close()
 assert data.stay_id.is_unique
 data=data.sort_values(['subject_id','hadm_id','stay_id'],kind='stable').reset_index(drop=True)
 checks=[];counts=[]
 for label,flag in [('primary','primary_analysis_include'),('strict','strict_sensitivity_include')]:
  selected=data[data[flag]==1].reset_index(drop=True);validation=selected[selected.corrected_analysis_split=='temporal_validation'].reset_index(drop=True)
  counts.extend(selected.groupby(['corrected_analysis_split','corrected_outcome_code']).size().rename('n').reset_index().assign(analysis=label).to_dict('records'))
  # Historical 8590/689 identifies the previous version, not a target to force.
  # A deterministic resolution of tied Stopped/FinishedRunning records can
  # change eligibility under the unchanged explicit-Stopped predicate.
  assert len(validation)>0 and sum(selected.corrected_analysis_split=='development')>0
  for family in ['main','ablation']:
   paths=sorted((package/'historical_parameters').glob(f'mimic_{label}__*.json' if family=='main' else f'ablation__source__corrected_{label}__*.json'))
   assert len(paths)==(4 if family=='main' else 2)
   specs={p.stem.split('__')[-1]:json.loads(p.read_text()) for p in paths}
   templates={k:build_unfitted(v) for k,v in specs.items()};features={k:v['ordered_input_features'] for k,v in specs.items()}
   original_comparisons=evaluator.COMPARISONS
   if family=='ablation':evaluator.COMPARISONS=[]
   print('FIT AND BOOTSTRAP',label,family,flush=True)
   fitted,point,ci,paired,dca=evaluator.fit_and_evaluate('scope_corrected_'+label,selected,templates,features)
   evaluator.COMPARISONS=original_comparisons
   for suffix,frame in [('temporal_results',point),('patient_bootstrap_ci',ci),('paired_differences',paired),('decision_curve',dca)]:frame.to_csv(results/f'mimic_{label}_{family}_{suffix}.csv',index=False)
   joblib.dump({'models':fitted,'features':features},out/f'mimic_{label}_{family}_models.joblib')
   for name,model in fitted.items():
    spec=export(model);spec['provenance']={'version':'deterministic_20260910','historical_architecture_only':True,'training_sort':['subject_id','hadm_id','stay_id'],'input_sha256':{str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [f,c]}}
    key=f'mimic_{label}__{name}' if family=='main' else f'ablation__source__corrected_{label}__{name}'
    (params/f'{key}.json').write_text(json.dumps(safe(spec),indent=2,allow_nan=False))
    predicted=model.predict_proba(validation[features[name]]);reconstructed=predict_proba(validation,spec);delta=float(np.max(np.abs(predicted-reconstructed)))
    checks.append({'model':key,'n':len(validation),'max_absolute_prediction_difference':delta,'tolerance':1e-10,'judgement':'match' if delta<=1e-10 else 'DIFFERENCE'})
    print('EXPORTED',key,'JSON prediction difference',delta,flush=True)
 pd.DataFrame(checks).to_csv(results/'fixed_model_json_replay.csv',index=False);pd.DataFrame(counts).to_csv(results/'cohort_counts.csv',index=False)
 env={'python':sys.version,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy','duckdb','joblib','pyarrow']},'numerical_threads':1,'mimic_bootstrap_seed':evaluator.SEED}
 (out/'environment.json').write_text(json.dumps(env,indent=2));assert all(x['judgement']=='match' for x in checks)
 print('MIMIC TRAINING COMPLETE',flush=True)
if __name__=='__main__':main()
