"""Run prespecified independent experiment jobs; no hidden failure skipping."""
import concurrent.futures, subprocess, os, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'qa').mkdir(exist_ok=True)

def run(job):
    label,args=job
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (ROOT/'qa'/f'{label}.log').open('w') as log:
        p=subprocess.run(['python',str(ROOT/'src/revision_experiments.py'),*args],env=env,stdout=log,stderr=subprocess.STDOUT)
    return {'job':label,'exit_code':p.returncode}

if __name__=='__main__':
    jobs=[]
    for n in [12,30]:
        for setting in range(4):
            jobs.append((f'sim_{n}_{setting}',['sim','--n',str(n),'--setting',str(setting),'--reps','500']))
    for n in [10,30,49]:
        configs=['base','free','cost25','kappa_half','kappa_double','window126','window504','quarterly'] if n!=49 else ['base','window126']
        for config in configs:jobs.append((f'history_{n}_{config}',['history','--n',str(n),'--config',config]))
    if "--retry" in sys.argv:
        old=json.loads((ROOT/"qa/job_status.json").read_text())
        failed={x["job"] for x in old if x["exit_code"]}
        jobs=[j for j in jobs if j[0] in failed]
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,job) for job in jobs]):
            result=future.result();results.append(result);print(result,flush=True)
            (ROOT/'qa'/('retry_status.json' if '--retry' in sys.argv else 'job_status.json')).write_text(json.dumps(results,indent=2))
    if any(x['exit_code'] for x in results):raise SystemExit('Experiment failure: inspect all logs')
