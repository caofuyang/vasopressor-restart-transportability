"""Emit original report, independently reproduced value, difference and judgement."""
from pathlib import Path
import argparse
import numpy as np,pandas as pd

def compare(old,new,keys,artifact,rows,tolerance):
 a=pd.read_csv(old);b=pd.read_csv(new)
 merged=a.merge(b,on=keys,suffixes=('_reported','_reproduced'),validate='one_to_one',how='outer',indicator=True)
 for _,row in merged.iterrows():
  identity={k:row[k] for k in keys}
  if row['_merge']!='both':
   rows.append(dict(artifact=artifact,**identity,field='record_membership',reported_value=None,reproduced_value=None,difference=None,tolerance=0,judgement=str(row['_merge'])));continue
  for col in a.columns:
   if col in keys or col not in b or not pd.api.types.is_numeric_dtype(a[col]):continue
   x=row[col+'_reported'];y=row[col+'_reproduced'];delta=float(y-x) if pd.notna(x) and pd.notna(y) else np.nan
   match=(pd.isna(x) and pd.isna(y)) or (pd.notna(delta) and abs(delta)<=tolerance)
   rows.append(dict(artifact=artifact,**identity,field=col,reported_value=x,reproduced_value=y,difference=delta,tolerance=tolerance,judgement='match' if match else 'DIFFERENCE'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--historical',type=Path,required=True);ap.add_argument('--reproduced',type=Path,required=True);ap.add_argument('--database',choices=['mimic','eicu'],required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();rows=[]
 if args.database=='mimic':
  for label in ['primary','strict']:
   for suffix,keys in [('temporal_results',['analysis','model','outcome']),('patient_bootstrap_ci',['analysis','model','outcome','metric']),('paired_differences',['analysis','comparison','metric']),('decision_curve',['analysis','model','threshold'])]:
    old=args.historical/f'mimic_{label}_{suffix}.csv';new=args.reproduced/f'mimic_{label}_main_{suffix}.csv'
    compare(old,new,keys,old.name,rows,1e-8 if 'bootstrap' in suffix or 'paired' in suffix else 1e-10)
 else:
  for variant in ['primary_traceable','freshness_180','gap_60']:
   for suffix,keys in [('performance',['variant','model']),('hospital_CI',['variant','model','metric']),('paired_differences',['variant','model','reference','metric'])]:
    name=f'{variant}_{suffix}.csv';compare(args.historical/name,args.reproduced/name,keys,name,rows,1e-10 if suffix=='performance' else 1e-8)
 args.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(args.output,index=False)
 print('Compared',len(rows),'reported values;',sum(r['judgement']!='match' for r in rows),'differences. Differences remain visible; no result selection.',flush=True)
if __name__=='__main__':main()
