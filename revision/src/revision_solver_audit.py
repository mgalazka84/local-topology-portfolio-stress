"""Re-solve all 15 archived historical fits with a non-success flag.

The two flagged original simulation fits are separately reconstructed using
their original sequential random stream. No flag is treated as failure of
the independent objective bound.
"""
from pathlib import Path
import json, importlib.util, sys
import numpy as np
import pandas as pd
from tree_risk import TreeRisk, allocate
from common import fitted_tree

ROOT=Path(__file__).resolve().parents[1]

def run():
    old_path=ROOT/'archive_source/tree_risk.py'
    if not old_path.exists():
        import urllib.request
        old_path.parent.mkdir(exist_ok=True)
        url='https://raw.githubusercontent.com/mgalazka84/local-topology-portfolio-stress/621731f62a1f155bea1614418674dbcaf10d01f0/src/tree_risk.py'
        with urllib.request.urlopen(url,timeout=60) as response:old_path.write_bytes(response.read())
    old_results=ROOT/'original_results'
    if not old_results.exists():old_results=ROOT.parent/'results' 
    spec=importlib.util.spec_from_file_location('original_tree_risk',old_path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    rows=[]
    for n in [10,30]:
        daily=pd.read_csv(ROOT/'data'/f'industry_{n}.csv',index_col=0,parse_dates=True)
        flags=pd.read_csv(old_results/f'empirical_{n}_solvers.csv')
        flags=flags[~flags.master_success]
        holdings=json.loads((old_results/f'empirical_{n}_holdings.json').read_text())
        for row in flags.itertuples():
            records=[h for h in holdings if h['method']==row.method and h['cost']==row.cost]
            k=next(k for k,h in enumerate(records) if h['date']==row.date)
            if k==0:prior=np.ones(n)/n
            else:
                previous=records[k-1];prior=np.array(previous['weights'])
                changes=(1+daily.loc[previous['date']:row.date].iloc[:-1]).prod().to_numpy()
                prior*=changes;prior/=prior.sum()
            date=pd.Timestamp(row.date);t=daily.index.get_loc(date)
            C,tree=fitted_tree(daily.iloc[t-252:t].to_numpy())
            A=C if row.method=='Shrinkage' else tree.covariance()
            tr=tree if row.method=='Robust' else None
            w,info=allocate(A,prior,row.cost,tree=tr,risk_multiplier=52.5)
            oldw=np.array(records[k]['weights'])
            def obj(v):return 52.5*(tree.query(v)[0] if tr else float(v@A@v))+row.cost*np.abs(v-prior).sum()
            rows.append({'source':'history','n':n,'case':row.date,'method':row.method,'cost':row.cost,
                'original_gap_scaled':row.gap_scaled,'objective_difference':obj(w)-obj(oldw),
                'weight_l1_difference':float(np.abs(w-oldw).sum()),**info})
    flags=pd.read_csv(old_results/'simulation_solvers.csv')
    flags=flags[~flags.master_success];rng=np.random.default_rng(730112)
    for n in [12,30]:
        for setting in ['Gaussian tree','Student t tree','Structural break','Factor misspecification']:
            for rep in range(100):
                edges=[(int(rng.integers(i)),i) for i in range(1,n)]
                rho=np.eye(n)
                for i,j in edges:rho[i,j]=rho[j,i]=rng.uniform(.35,.75)
                sigma=rng.uniform(.008,.02,n);S=TreeRisk(rho,edges,sigma).covariance()
                if setting=='Factor misspecification':
                    B=np.zeros((n,4));B[:,0]=rng.uniform(.25,.55,n)
                    for i in range(n):B[i,1+i%3]=rng.uniform(.35,.65)
                    Q=B@B.T+np.diag(rng.uniform(.25,.55,n));s=np.sqrt(np.diag(Q))
                    S=Q/s[:,None]/s[None,:]*sigma[:,None]*sigma[None,:]
                elif setting=='Structural break':
                    trueR=S/sigma[:,None]/sigma[None,:];base=TreeRisk(trueR,edges,sigma)
                    path=base.paths[int(rng.integers(len(base.paths)))];rng.integers(len(path.correlations));rng.uniform(.5,.85)
                X=rng.normal(size=(126,n))@np.linalg.cholesky(S).T
                if setting=='Student t tree':X*=np.sqrt(3/rng.chisquare(5,size=(126,1)))
                selected=flags[(flags.n==n)&(flags.setting==setting)&(flags.rep==rep)]
                if len(selected)==0:continue
                C,tree=fitted_tree(X);prior=np.ones(n)/n
                for row in selected.itertuples():
                    A=C if row.method=='Shrinkage' else tree.covariance();tr=tree if row.method=='Robust' else None
                    w,info=allocate(A,prior,.001,tree=tr,risk_multiplier=52.5)
                    oldw,oi=old.allocate(A,prior,.001,tree=tr,risk_multiplier=52.5)
                    def obj(v):return 52.5*(tree.query(v)[0] if tr else float(v@A@v))+.001*np.abs(v-prior).sum()
                    rows.append({'source':'simulation','n':n,'case':setting+':'+str(rep),'method':row.method,'cost':.001,
                        'original_gap_scaled':row.gap_scaled,'objective_difference':obj(w)-obj(oldw),
                        'weight_l1_difference':float(np.abs(w-oldw).sum()),**info})
    assert len(rows)==17
    result=pd.DataFrame(rows);result.to_csv(ROOT/'results/archived_flag_audit.csv',index=False)
    print(result[['source','n','case','method','objective_difference','weight_l1_difference']].to_string(index=False))

if __name__=='__main__':run()
