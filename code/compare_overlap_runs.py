from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from compare_independent_runs import compare_frames
from portable_predict import predict_proba,recalibrate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--first',type=Path,required=True);ap.add_argument('--second',type=Path,required=True);ap.add_argument('--eicu-run',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();rows=[]
 for file,keys in [('posthoc_overlap_exclusion_sensitivity.csv',['analysis','model']),('posthoc_overlap_exclusion_hospital_bootstrap_CI.csv',['analysis','model','metric'])]:compare_frames(pd.read_csv(args.first/file),pd.read_csv(args.second/file),keys,file,rows,1e-8)
 l=pd.read_parquet(args.eicu_run/'corrected/private/corrected_labels.parquet');f=pd.read_parquet(args.eicu_run/'derived/stage2b_explicit_stop_feature_matrix.parquet');d=l[l.variant=='primary_traceable'].merge(f,on='patienthealthsystemstayid',validate='one_to_one');part=d.groupby('uniquepid').partition.nunique();excluded=part[part>1].index;e=d[d.included&~d.uniquepid.isin(excluded)&(d.partition=='locked_evaluation')].sort_values('patienthealthsystemstayid',kind='stable')
 source=json.loads(args.source.read_text());raw=predict_proba(e,source)[:,source['estimator']['classes'].index(1)]
 for path in (args.first/'parameters').glob('*.json'):
  predictions=[]
  for run in [args.first,args.second]:
   spec=json.loads((run/'parameters'/path.name).read_text());predictions.append(recalibrate(raw,spec) if spec.get('kind')=='recalibration' else predict_proba(e,spec))
  delta=float(np.abs(predictions[0]-predictions[1]).max());rows.append(dict(artifact=path.name,field='independently_retrained_predictions',n=len(e),max_absolute_difference=delta,tolerance=1e-10,judgement='match' if delta<=1e-10 else 'DIFFERENCE'))
 pd.DataFrame(rows).to_csv(args.output,index=False);print('Overlap sensitivity comparisons',len(rows),'differences',sum(r['judgement']!='match' for r in rows))
if __name__=='__main__':main()
