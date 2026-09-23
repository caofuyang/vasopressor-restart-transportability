"""Run the frozen eICU chain from authorized raw tables and readable source models.

All patient-level intermediates and refitted binaries stay under --run-dir.
Never distribute that run directory.
"""
from pathlib import Path
import argparse,os,subprocess,sys,json,time
import pandas as pd,numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--parameters',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args()
    code=Path(__file__).resolve().parent;package=code.parent;run=args.run_dir.resolve()
    if run.exists():raise SystemExit('Refusing to reuse an existing run directory: '+str(run))
    run.mkdir(parents=True);(run/'derived').mkdir();(run/'temp').mkdir()
    env=os.environ.copy();env.update(EICU_RAW=str(args.raw.resolve()),EICU_DERIVED=str(run/'derived'),REPRO_TEMP=str(run/'temp'),EICU_RUN=str(run/'corrected'),EICU_AUDIT_RAW=str(args.raw.resolve()),EICU_AUDIT_DERIVED=str(run/'derived'),REPRO_PARAMETERS=str(args.parameters.resolve()),PYTHONPATH=str(code),PYTHONIOENCODING='utf8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    executions=[]
    def execute(relative):
        started=time.time();print('START',relative,flush=True)
        with (run/(Path(relative).stem+'.log')).open('w',encoding='utf8') as log:
            result=subprocess.run([sys.executable,str(code/relative)],env=env,stdout=log,stderr=subprocess.STDOUT)
        executions.append({'step':relative,'exit_code':result.returncode,'elapsed_seconds':time.time()-started})
        (run/'execution.json').write_text(json.dumps(executions,indent=2))
        if result.returncode:raise SystemExit('Execution failed; inspect private run log: '+relative)
        print('COMPLETE',relative,flush=True)
    execute('eicu/eicu_stage1b_explicit_stop_cohort.py')
    execute('eicu/eicu_stage2_external_features.py')
    execute('eicu_correction/01_correct_cohort.py')
    labels=pd.read_parquet(run/'corrected/private/corrected_labels.parquet');features=pd.read_parquet(run/'derived/stage2b_explicit_stop_feature_matrix.parquet')
    assert len(features)==features.patienthealthsystemstayid.nunique()==8810
    assert len(labels)==26430
    selected=labels[(labels.variant=='primary_traceable')&labels.included]
    expected={'adaptation_development':3297,'recalibration':1378,'locked_evaluation':2104}
    assert selected.groupby('partition').size().to_dict()==expected
    evaluation=selected[selected.partition=='locked_evaluation'];assert evaluation.new_code.value_counts().to_dict()=={0:1314,1:734,2:56}
    (run/'corrected/results/pre_performance_QA.json').write_text(json.dumps({'all_passed':True,'scope':'Raw candidate/feature keys, fixed origins, frozen rule replay and primary partition/outcome counts; not a claim that model performance has already been verified.'}))
    execute('eicu_correction/03_evaluate.py')
    rows=[]
    for variant in ['primary_traceable','freshness_180','gap_60']:
        for suffix,keys in [('performance',['variant','model']),('hospital_CI',['variant','model','metric']),('paired_differences',['variant','model','reference','metric'])]:
            name=f'{variant}_{suffix}.csv';a=pd.read_csv(package/'reference_aggregates'/name);b=pd.read_csv(run/'corrected/results'/name)
            merged=a.merge(b,on=keys,suffixes=('_old','_new'),validate='one_to_one');assert len(merged)==len(a)==len(b)
            for _,r in merged.iterrows():
                for c in a.columns:
                    if c in keys or not pd.api.types.is_numeric_dtype(a[c]):continue
                    x=r[c+'_old'];y=r[c+'_new'];delta=float(y-x) if pd.notna(x) and pd.notna(y) else np.nan
                    matched=(pd.isna(x) and pd.isna(y)) or (pd.notna(delta) and abs(delta)<=1e-8)
                    rows.append(dict(file=name,**{k:r[k] for k in keys},field=c,reported_value=x,reproduced_value=y,difference=delta,tolerance=1e-8,judgement='match' if matched else 'DIFFERENCE'))
    df=pd.DataFrame(rows);df.to_csv(run/'PUBLIC_aggregate_comparison.csv',index=False)
    print('Independent raw-to-results execution complete;',len(df),'comparisons;',int((df.judgement!='match').sum()),'differences. Do not publish the private run directory.',flush=True)

if __name__=='__main__':main()
