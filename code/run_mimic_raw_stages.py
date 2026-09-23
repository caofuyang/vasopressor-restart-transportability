from pathlib import Path
import argparse,os,subprocess,sys,json,time
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args()
 run=args.run_dir.resolve()
 if run.exists():raise SystemExit('Refusing to reuse an existing run directory')
 run.mkdir(parents=True);(run/'derived').mkdir();(run/'temp').mkdir();code=Path(__file__).resolve().parent
 env=os.environ.copy();env.update(MIMIC_RAW=str(args.raw.resolve()),MIMIC_DERIVED=str(run/'derived'),REPRO_TEMP=str(run/'temp'),PYTHONIOENCODING='utf8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MIMIC_RUN=str(run))
 history=[]
 for stage in [1,2,3,4,5,6,7,11,12,13,17,18,'19_post_audit','20_scope_corrected']:
  name=f'stage{stage:02d}.py' if isinstance(stage,int) else f'stage{stage}.py';start=time.time();print('START',name,flush=True)
  with (run/(name+'.log')).open('w',encoding='utf8') as log:result=subprocess.run([sys.executable,str(code/'mimic'/name)],env=env,stdout=log,stderr=subprocess.STDOUT)
  history.append({'stage':stage,'exit_code':result.returncode,'seconds':time.time()-start});(run/'execution.json').write_text(json.dumps(history,indent=2))
  print('FINISHED',name,'exit',result.returncode,flush=True)
  if result.returncode:raise SystemExit('Stage failed; inspect private log')
if __name__=='__main__':main()
