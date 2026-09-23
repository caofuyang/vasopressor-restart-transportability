"""Run authorized raw tables through the frozen analysis in a new private directory."""
from pathlib import Path
import argparse,os,subprocess,sys,json,time

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['mimic-raw','eicu-raw','output']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();root=a.output.resolve()
 if root.exists():raise SystemExit('Choose a new output directory; historical runs are never overwritten.')
 root.mkdir(parents=True);code=Path(__file__).resolve().parent
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf8',PYTHONPATH=str(code),MIMIC_RUN=str(root/'mimic'),EICU_RUN_ROOT=str(root/'eicu'),REPRO_PARAMETERS=str(root/'mimic/models/parameters'),OVERLAP_OUTPUT=str(root/'overlap'),AGGREGATE_OUTPUT=str(root/'descriptive'),DISPLAY_OUTPUT=str(root/'display'),CLINICAL_OUTPUT=str(root/'clinical'),POSTHOC_OUTPUT=str(root/'posthoc'),HISTORY_REPLAY=str(root/'history'))
 jobs=[['run_mimic_raw_stages.py','--raw',a.mimic_raw.resolve(),'--run-dir',root/'mimic'],['train_mimic.py','--run-dir',root/'mimic'],['run_eicu.py','--raw',a.eicu_raw.resolve(),'--run-dir',root/'eicu','--parameters',root/'mimic/models/parameters'],['export_eicu.py','--run-dir',root/'eicu','--source-parameters',root/'mimic/models/parameters'],['replay_existing_overlap_sensitivity.py'],['build_descriptive_aggregates.py'],['build_result_displays.py'],['build_clinical_results.py'],['prepare_historical_comparison_inputs.py'],['replay_existing_posthoc.py'],['publication_figures.py','--clinical',root/'clinical','--mimic-results',root/'mimic/models/results','--eicu-results',root/'eicu/corrected/results','--posthoc',root/'posthoc','--display',root/'display','--out',root/'figures']]
 history=[]
 for i,job in enumerate(jobs):
  command=[sys.executable,str(code/job[0]),*[str(x) for x in job[1:]]];print('START',job[0],flush=True);t=time.time()
  with (root/f'{i:02d}_{job[0]}.log').open('w',encoding='utf8') as log:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
  history.append({'stage':job[0],'exit_code':r.returncode,'seconds':time.time()-t});(root/'execution.json').write_text(json.dumps(history,indent=2))
  if r.returncode:raise SystemExit('Failed; inspect private stage log. No successful reproduction is claimed.')
  print('COMPLETE',job[0],flush=True)
 print('Raw-to-results run complete. Compare a second independent run before claiming retraining reproducibility. Never publish this private output directory.')
if __name__=='__main__':main()
