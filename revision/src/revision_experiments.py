"""Major-revision protocol. Parameters fixed before reading new outcomes.

Independent indexed random seeds permit replication and process-level jobs.
All fits use Clarabel with independently evaluated global objective bounds.
"""
import argparse, json, time, os, platform, sys, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.stats import rankdata
from tree_risk import TreeRisk, maximum_tree, allocate, hrp
from common import fitted_tree, account_cost, metrics

ROOT=Path(__file__).resolve().parents[1]
METHODS=['Equal','HRP','Sample','Shrinkage','Rank','Tree','Robust']
SETTINGS=['Gaussian tree','Student t tree','Structural break','Factor misspecification']

def rank_covariance(X):
    """Prespecified PSD Spearman covariance with robust marginal MAD scales.

    Direct rank correlation (no Gaussian sine conversion); 5% identity ridge.
    This is an outlier-resistant sensitivity comparator, not a calibrated
    Pearson estimator under an asserted elliptical population model.
    """
    rank=np.apply_along_axis(rankdata,0,X)
    rho=np.corrcoef(rank,rowvar=False)
    scale=1.482602218505602*np.median(np.abs(X-np.median(X,axis=0)),axis=0)
    scale=np.maximum(scale,1e-8)
    return (.95*rho+.05*np.eye(X.shape[1]))*scale[:,None]*scale[None,:]

def targets(X,C,tree,previous,cost,kappa=52.5,methods=METHODS):
    n=len(C);out={};info={}
    if 'Equal' in methods:out['Equal']=np.ones(n)/n
    if 'HRP' in methods:out['HRP']=hrp(C)
    matrices={'Shrinkage':C,'Tree':tree.covariance(),'Robust':tree.covariance()}
    if 'Sample' in methods:matrices['Sample']=np.cov(X,rowvar=False,ddof=1)
    if 'Rank' in methods:matrices['Rank']=rank_covariance(X)
    for name in methods:
        if name in matrices:
            out[name],info[name]=allocate(matrices[name],previous[name],cost,
                tree=tree if name=='Robust' else None,risk_multiplier=kappa)
    return out,info

def simulation(n,setting_id,reps):
    rows=[];solvers=[];setting=SETTINGS[setting_id]
    methods=[x for x in METHODS if x!='Rank']
    for rep in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([730112,n,setting_id,rep]))
        edges=[(int(rng.integers(i)),i) for i in range(1,n)]
        rho=np.eye(n)
        for i,j in edges:rho[i,j]=rho[j,i]=rng.uniform(.35,.75)
        sigma=rng.uniform(.008,.02,n)
        S=TreeRisk(rho,edges,sigma,allowed=edges).covariance();evaluation=S.copy()
        if setting=='Factor misspecification':
            B=np.zeros((n,4));B[:,0]=rng.uniform(.25,.55,n)
            for i in range(n):B[i,1+i%3]=rng.uniform(.35,.65)
            Q=B@B.T+np.diag(rng.uniform(.25,.55,n));s=np.sqrt(np.diag(Q))
            S=Q/s[:,None]/s[None,:]*sigma[:,None]*sigma[None,:];evaluation=S.copy()
        elif setting=='Structural break':
            trueR=S/sigma[:,None]/sigma[None,:];base=TreeRisk(trueR,edges,sigma)
            path=base.paths[int(rng.integers(len(base.paths)))];k=int(rng.integers(len(path.correlations)))
            swap=(tuple(sorted((path.nodes[k],path.nodes[k+1]))),path.endpoints)
            R=trueR.copy();u,v=path.endpoints;R[u,v]=R[v,u]=rng.uniform(.5,.85)
            evaluation=TreeRisk(R,base.exchanged_edges(swap),sigma,allowed=base.exchanged_edges(swap)).covariance()
        X=rng.normal(size=(126,n))@np.linalg.cholesky(S).T
        if setting=='Student t tree':X*=np.sqrt(3/rng.chisquare(5,size=(126,1)))
        C,tree=fitted_tree(X);prior={name:np.ones(n)/n for name in methods}
        oracle,oi=allocate(evaluation,np.ones(n)/n,0.)
        popopt=float(oracle@evaluation@oracle)
        solvers.append({'n':n,'setting':setting,'rep':rep,'cost':0.,'method':'Population oracle',**oi})
        for cost in [0.,.001]:
            weights,infos=targets(X,C,tree,prior,cost,methods=methods)
            for name,w in weights.items():
                risk=float(w@evaluation@w)
                rows.append({'n':n,'setting':setting,'rep':rep,'cost':cost,'method':name,
                    'population_variance':risk,'relative_population_variance':risk/popopt,
                    'two_way_turnover':float(np.abs(w-prior[name]).sum())})
            solvers.extend({'n':n,'setting':setting,'rep':rep,'cost':cost,'method':name,**info}
                           for name,info in infos.items())
        if (rep+1)%50==0:
            print('simulation',n,setting,rep+1,flush=True)
            pd.DataFrame(rows).to_csv(ROOT/'results'/f'sim_{n}_{setting_id}.csv',index=False)
            pd.DataFrame(solvers).to_csv(ROOT/'results'/f'sim_{n}_{setting_id}_solvers.csv',index=False)
    pd.DataFrame(rows).to_csv(ROOT/'results'/f'sim_{n}_{setting_id}.csv',index=False)
    pd.DataFrame(solvers).to_csv(ROOT/'results'/f'sim_{n}_{setting_id}_solvers.csv',index=False)

CONFIGS={
    'base':(252,52.5,1,.001), 'free':(252,52.5,1,0.),
    'cost25':(252,52.5,1,.0025),'kappa_half':(252,26.25,1,.001),
    'kappa_double':(252,105.,1,.001),'window126':(126,52.5,1,.001),
    'window504':(504,52.5,1,.001),'quarterly':(252,52.5,3,.001)}

def empirical(n,config):
    window,kappa,months,cost=CONFIGS[config]
    returns=pd.read_csv(ROOT/'data'/f'industry_{n}.csv',index_col=0,parse_dates=True)
    rf=pd.read_csv(ROOT/'data/risk_free.csv',index_col=0,parse_dates=True)['RF']
    X=returns.to_numpy();dates=returns.index
    start=int(np.flatnonzero(dates>='2000-01-01')[0]);assert start>=window
    methods=METHODS if config=='base' else ['Shrinkage','Tree','Robust']
    previous={name:np.ones(n)/n for name in methods}
    rows=[];solvers=[];holdings=[];count=0
    for t in range(start,len(X)):
        rebalance=(t==start or (dates[t].month!=dates[t-1].month and (dates[t].month-1)%months==0))
        if rebalance:
            train=X[t-window:t];fit_start=time.perf_counter();C,tree=fitted_tree(train)
            fit_seconds=time.perf_counter()-fit_start
            weights,infos=targets(train,C,tree,previous,cost,kappa,methods)
            trade={}
            for name in methods:
                k,turn=account_cost(weights[name],previous[name],cost)
                trade[name]=(k,turn)
                holdings.append({'date':str(dates[t].date()),'method':name,'weights':weights[name].tolist(),
                    'prior':previous[name].tolist(),'fee':k,'turnover':turn})
                previous[name]=weights[name]
                if name in infos:solvers.append({'date':str(dates[t].date()),'method':name,
                    'L':tree.L,'fit_seconds':fit_seconds,**infos[name]})
            count+=1
            if count%52==0:print('history',n,config,dates[t].date(),count,flush=True)
        for name in methods:
            w=previous[name];gross=float(w@X[t]);k,turn=trade[name] if rebalance else (0.,0.)
            net=(1-k)*(1+gross)-1
            rows.append({'date':str(dates[t].date()),'method':name,'net':net,'gross':gross,
                'fee':k,'turnover':turn,'excess':net-float(rf.loc[dates[t]])})
            previous[name]=w*(1+X[t])/(1+gross)
    prefix=f'history_{n}_{config}'
    daily=pd.DataFrame(rows);daily.to_csv(ROOT/'results'/f'{prefix}_daily.csv',index=False)
    pd.DataFrame(solvers).to_csv(ROOT/'results'/f'{prefix}_solvers.csv',index=False)
    (ROOT/'results'/f'{prefix}_holdings.json').write_text(json.dumps(holdings,separators=(',',':')))
    summary=[]
    for name,g in daily.groupby('method'):
        m=metrics(g.net,g.gross,g.turnover)
        m['sharpe']=float(np.sqrt(252)*g.excess.mean()/g.excess.std(ddof=1))
        m['annualized_fees']=float(g.fee.sum()*252/len(g))
        summary.append({'panel':n,'config':config,'cost':cost,'kappa':kappa,'window':window,
                        'rebalance_months':months,'decisions':count,'method':name,**m})
    pd.DataFrame(summary).to_csv(ROOT/'results'/f'{prefix}_summary.csv',index=False)
    print('COMPLETE history',n,config,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['sim','history']);parser.add_argument('--n',type=int,default=12)
    parser.add_argument('--setting',type=int,default=0);parser.add_argument('--reps',type=int,default=500)
    parser.add_argument('--config',default='base',choices=CONFIGS);args=parser.parse_args()
    if args.mode=='sim':simulation(args.n,args.setting,args.reps)
    else:empirical(args.n,args.config)
