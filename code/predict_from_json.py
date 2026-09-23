"""Independent readable-parameter prediction; input and output contain restricted records."""
from pathlib import Path
import argparse,json
import pandas as pd
from portable_predict import predict_proba,recalibrate
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['input','model','output']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--recalibration',type=Path);a=p.parse_args()
 if a.output.exists():raise SystemExit('Refusing to overwrite predictions')
 frame=pd.read_parquet(a.input) if a.input.suffix=='.parquet' else pd.read_csv(a.input)
 spec=json.loads(a.model.read_text());prob=predict_proba(frame,spec);classes=spec['estimator']['classes']
 out=pd.DataFrame(prob,columns=[f'probability_class_{x}' for x in classes])
 if a.recalibration:out['recalibrated_restart_probability']=recalibrate(prob[:,classes.index(1)],json.loads(a.recalibration.read_text()))
 out.to_csv(a.output,index=False);print('Predictions saved privately in input row order; do not distribute patient-level output.')
if __name__=='__main__':main()
