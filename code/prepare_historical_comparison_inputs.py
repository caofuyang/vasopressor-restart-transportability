"""Apply genuine archived model parameters to deterministic raw-derived features.

Historical published performance rows remain labeled archival benchmarks; they
are not relabeled as current models or as raw-retrained results.
"""
from pathlib import Path
import os,json
import pandas as pd
from portable_predict import predict_proba

def main():
 package=Path(__file__).resolve().parent.parent;run=Path(os.environ['MIMIC_RUN']);e=Path(os.environ['EICU_RUN_ROOT']);out=Path(os.environ['HISTORY_REPLAY']);out.mkdir(exist_ok=False)
 cohort=pd.read_parquet(run/'derived/stage18_competing_outcomes_cross_icu_corrected.parquet');f=pd.read_parquet(run/'derived/vasopressor_liberation_features_stage7.parquet')
 d=cohort[['subject_id','hadm_id','stay_id','competing_outcome_code']].merge(f,on=['subject_id','hadm_id','stay_id'],validate='one_to_one').sort_values(['subject_id','hadm_id','stay_id'],kind='stable')
 spec=json.loads((package/'historical_pre_audit_parameters/dynamic_multinomial_logistic.json').read_text());p=predict_proba(d,spec)[:,spec['estimator']['classes'].index(1)]
 result=d[['subject_id','hadm_id','stay_id']].copy();result['observed_outcome']=d.competing_outcome_code;result['dynamic_multinomial_logistic__p_restart_first']=p;result.to_parquet(out/'legacy_fixed_predictions.parquet',index=False)
 old=pd.read_csv(package/'historical_reference_aggregates/eicu_historical_transport_before_after.csv');new=pd.read_csv(e/'corrected/results/all_performance.csv').set_index(['variant','model'])
 for i,row in old.iterrows():
  key=(row.variant,row.model)
  if key not in new.index:continue
  for col in new.columns:
   if col+'_new' in old.columns:old.loc[i,col+'_new']=new.loc[key,col]
 old.to_csv(out/'transport_before_after.csv',index=False)
 (out/'provenance.json').write_text(json.dumps({'fixed_prediction_label_comparison':'Archived pre-audit model parameters applied to deterministic features; both label versions use exactly the same predictions. Not an exact replay of historical feature tie choices.','archival_benchmark_rows':'Retained original reported aggregate values, explicitly labeled historical; current corrected rows updated from the independently repeated run.'},indent=2))
if __name__=='__main__':main()
