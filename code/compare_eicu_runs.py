"""Numerical comparison of two independently reconstructed and refitted eICU runs."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from compare_independent_runs import compare_frames
from portable_predict import predict_proba,recalibrate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--first',type=Path,required=True);ap.add_argument('--second',type=Path,required=True);ap.add_argument('--source-parameters',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();rows=[];frames=[]
 for rel,keys in [('derived/stage1b_explicit_stop_cohort.parquet',['patienthealthsystemstayid']),('derived/stage2b_explicit_stop_feature_matrix.parquet',['patienthealthsystemstayid']),('corrected/private/corrected_labels.parquet',['variant','patienthealthsystemstayid'])]:compare_frames(pd.read_parquet(args.first/rel),pd.read_parquet(args.second/rel),keys,rel,rows)
 for run in [args.first,args.second]:
  labels=pd.read_parquet(run/'corrected/private/corrected_labels.parquet');features=pd.read_parquet(run/'derived/stage2b_explicit_stop_feature_matrix.parquet');frames.append(labels.merge(features,on='patienthealthsystemstayid',validate='many_to_one').sort_values(['variant','patienthealthsystemstayid'],kind='stable'))
 for variant in ['primary_traceable','freshness_180','gap_60']:
  for suffix,keys in [('performance',['variant','model']),('hospital_CI',['variant','model','metric']),('paired_differences',['variant','model','reference','metric'])]:
   name=f'{variant}_{suffix}.csv';compare_frames(pd.read_csv(args.first/'corrected/results'/name),pd.read_csv(args.second/'corrected/results'/name),keys,name,rows,1e-10 if suffix=='performance' else 1e-8)
  for path in sorted((args.first/'parameters').glob(variant+'__*.json')):
   predictions=[]
   for run,frame in zip([args.first,args.second],frames):
    evaluation=frame[(frame.variant==variant)&frame.included&(frame.partition=='locked_evaluation')];spec=json.loads((run/'parameters'/path.name).read_text())
    if spec.get('kind')=='recalibration':
     base=json.loads((args.source_parameters/spec['base_parameter_file']).read_text());prediction=recalibrate(predict_proba(evaluation,base)[:,base['estimator']['classes'].index(1)],spec)
    else:prediction=predict_proba(evaluation,spec)
    predictions.append(prediction)
   delta=float(np.abs(predictions[0]-predictions[1]).max());changed=int((np.abs(predictions[0]-predictions[1])>1e-10).sum());rows.append(dict(artifact=path.name,field='independently_retrained_predictions',n=len(predictions[0]),missing_mask_differences=0,changed_records=changed,max_absolute_difference=delta,tolerance=1e-10,judgement='match' if changed==0 else 'DIFFERENCE'))
 pd.DataFrame(rows).to_csv(args.output,index=False);print('eICU aggregate comparisons',len(rows),'differences',sum(r['judgement']!='match' for r in rows),flush=True)
if __name__=='__main__':main()
