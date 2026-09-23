"""Compare independent patient-level artifacts privately; emit aggregate-only checks."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,joblib,duckdb
from portable_predict import predict_proba

def compare_frames(a,b,keys,name,rows,tol=1e-10):
 assert not a.duplicated(keys).any() and not b.duplicated(keys).any(),name
 a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index()
 assert a.index.equals(b.index),(name,'record membership')
 assert set(a.columns)==set(b.columns),(name,'schema')
 for col in a:
  x=a[col];y=b[col];missing=int((x.isna()!=y.isna()).sum());numeric=pd.api.types.is_numeric_dtype(x.dtype) and not pd.api.types.is_bool_dtype(x.dtype)
  if numeric:
   valid=x.notna()&y.notna();delta=float(np.max(np.abs(x[valid].to_numpy(float)-y[valid].to_numpy(float)),initial=0));changed=int((np.abs(x[valid].to_numpy(float)-y[valid].to_numpy(float))>tol).sum())
  else:delta=None;changed=int((~(x.eq(y)|(x.isna()&y.isna()))).sum())
  rows.append(dict(artifact=name,field=col,n=len(a),missing_mask_differences=missing,changed_records=changed,max_absolute_difference=delta,tolerance=tol,judgement='match' if missing==changed==0 else 'DIFFERENCE'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--first',type=Path,required=True);ap.add_argument('--second',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();rows=[]
 for rel,keys in [('derived/vasopressor_liberation_features_stage7.parquet',['stay_id']),('scope_corrected/restricted/stage18_post_audit_scope_corrected_cohort.parquet',['stay_id'])]:compare_frames(pd.read_parquet(args.first/rel),pd.read_parquet(args.second/rel),keys,rel,rows)
 frames=[]
 for run in [args.first,args.second]:
  f=run/'derived/vasopressor_liberation_features_stage7.parquet';c=run/'scope_corrected/restricted/stage18_post_audit_scope_corrected_cohort.parquet'
  con=duckdb.connect(config={'threads':1});frame=con.execute(f"SELECT f.* EXCLUDE(restart_24h, death_24h, failure_24h, next_episode_start), c.* EXCLUDE(subject_id,hadm_id,stay_id) FROM read_parquet('{f.as_posix()}') f INNER JOIN read_parquet('{c.as_posix()}') c USING(stay_id)").df();con.close();frames.append(frame.sort_values(['subject_id','hadm_id','stay_id'],kind='stable'))
 for path in sorted((args.first/'models/parameters').glob('*.json')):
  flag='strict_sensitivity_include' if 'strict' in path.stem else 'primary_analysis_include';probs=[]
  for run,frame in zip([args.first,args.second],frames):
   val=frame[(frame[flag]==1)&(frame.corrected_analysis_split=='temporal_validation')];spec=json.loads((run/'models/parameters'/path.name).read_text());probs.append(predict_proba(val,spec))
  delta=float(np.abs(probs[0]-probs[1]).max());changed=int((np.abs(probs[0]-probs[1]).max(axis=1)>1e-10).sum())
  rows.append(dict(artifact=path.name,field='independently_retrained_predictions',n=len(probs[0]),missing_mask_differences=0,changed_records=changed,max_absolute_difference=delta,tolerance=1e-10,judgement='match' if changed==0 else 'DIFFERENCE'))
 for f in sorted((args.first/'models/results').glob('*.csv')):
  if f.name=='fixed_model_json_replay.csv':keys=['model']
  elif f.name=='cohort_counts.csv':keys=['analysis','corrected_analysis_split','corrected_outcome_code']
  elif 'temporal_results' in f.name:keys=['analysis','model','outcome']
  elif 'patient_bootstrap' in f.name:keys=['analysis','model','outcome','metric']
  elif 'paired_differences' in f.name:
   if 'ablation' in f.name:continue
   keys=['analysis','comparison','metric']
  elif 'decision_curve' in f.name:keys=['analysis','model','threshold']
  else:continue
  compare_frames(pd.read_csv(f),pd.read_csv(args.second/'models/results'/f.name),keys,f.name,rows,1e-8)
 args.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(args.output,index=False)
 print('Aggregate field comparisons',len(rows),'differences',sum(x['judgement']!='match' for x in rows),flush=True)
if __name__=='__main__':main()
