"""Reproducible computational, simulation and historical-return experiments."""
import argparse,json,time,platform,sys,os,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from tree_risk import TreeRisk,maximum_tree,allocate,hrp

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'results').mkdir(exist_ok=True)
MULTIPLIER=5*21/2
METHODS=['Equal','HRP','Shrinkage','Tree','Robust']

def fitted_tree(X):
    C=LedoitWolf().fit(X).covariance_
    sigma=np.sqrt(np.diag(C));rho=C/sigma[:,None]/sigma[None,:]
    rho=np.clip(rho,-.999,.999);np.fill_diagonal(rho,1)
    tree=TreeRisk(rho,maximum_tree(rho),sigma)
    return C,tree

def targets(C,tree,previous,cost):
    out={'Equal':np.ones(len(C))/len(C),'HRP':hrp(C)};info={}
    for name,A,tr in [('Shrinkage',C,None),('Tree',tree.covariance(),None),('Robust',tree.covariance(),tree)]:
        out[name],info[name]=allocate(A,previous[name],cost,tree=tr,risk_multiplier=MULTIPLIER)
    return out,info

def account_cost(target,previous,cost):
    # Exact wealth accounting for proportional fees on pre-trade dollar trades.
    k=0.
    for _ in range(30):
        new=cost*np.abs((1-k)*target-previous).sum()
        if abs(new-k)<1e-15:k=new;break
        k=new
    return float(k),float(np.abs((1-k)*target-previous).sum())

def metrics(net,gross,turn):
    net=np.asarray(net);gross=np.asarray(gross);T=len(net)
    wealth=np.r_[1.,np.cumprod(1+net)]
    draw=wealth/np.maximum.accumulate(wealth)-1
    vol=float(net.std(ddof=1)*np.sqrt(252))
    return {'days':T,'annualized_net_mean':float(net.mean()*252),
        'annualized_gross_mean':float(gross.mean()*252),'annualized_volatility':vol,
        'zero_benchmark_ratio':float(net.mean()*252/vol),
        'net_cagr':float(wealth[-1]**(252/T)-1),'max_drawdown':float(draw.min()),
        'annualized_two_way_turnover':float(np.sum(turn)*252/T),'terminal_wealth':float(wealth[-1])}

def empirical(n):
    returns=pd.read_csv(ROOT/'data'/f'industry_{n}.csv',index_col=0,parse_dates=True)
    X=returns.to_numpy();dates=returns.index
    initial=int(np.flatnonzero(dates>='2000-01-01')[0]);assert initial>=252
    costs=[0.,.001,.0025]
    previous={c:{name:np.ones(n)/n for name in METHODS} for c in costs}
    rows=[];solvers=[];holdings=[];rebalance_index=0
    for t in range(initial,len(X)):
        is_rebalance=(t==initial or dates[t].month!=dates[t-1].month)
        if is_rebalance:
            fit_start=time.perf_counter();C,tree=fitted_tree(X[t-252:t]);fit_seconds=time.perf_counter()-fit_start
            for c in costs:
                weights,info=targets(C,tree,previous[c],c)
                for name in METHODS:
                    k,turn=account_cost(weights[name],previous[c][name],c)
                    holdings.append({'date':str(dates[t].date()),'cost':c,'method':name,'weights':weights[name].tolist(),
                        'cost_fraction':k,'two_way_turnover':turn,'stress_variance':tree.query(weights[name])[0],
                        'nominal_variance':tree.risk(weights[name])})
                    previous[c][name]=weights[name]
                    if name in info:
                        solvers.append({'date':str(dates[t].date()),'cost':c,'method':name,
                             'candidates':tree.candidate_count,'L':tree.L,'fit_seconds':fit_seconds,**info[name]})
                previous[c]['_trade']={h['method']:(h['cost_fraction'],h['two_way_turnover']) for h in holdings[-len(METHODS):]}
            rebalance_index+=1
            if rebalance_index%24==0:print('empirical',n,dates[t].date(),'rebalances',rebalance_index,flush=True)
        for c in costs:
            for name in METHODS:
                w=previous[c][name];gross=float(w@X[t]);k,turn=previous[c]['_trade'][name] if is_rebalance else (0.,0.)
                net=(1-k)*(1+gross)-1
                rows.append({'date':str(dates[t].date()),'cost':c,'method':name,'gross':gross,'net':net,'turnover':turn,'fee':k})
                previous[c][name]=w*(1+X[t])/(1+gross)
    daily=pd.DataFrame(rows);daily.to_csv(ROOT/'results'/f'empirical_{n}_daily.csv',index=False)
    pd.DataFrame(solvers).to_csv(ROOT/'results'/f'empirical_{n}_solvers.csv',index=False)
    (ROOT/'results'/f'empirical_{n}_holdings.json').write_text(json.dumps(holdings))
    summary=[]
    for (c,name),g in daily.groupby(['cost','method']):
        summary.append({'panel':n,'cost':c,'method':name,**metrics(g.net,g.gross,g.turnover)})
    pd.DataFrame(summary).to_csv(ROOT/'results'/f'empirical_{n}_summary.csv',index=False)
    print('COMPLETE empirical',n,flush=True)

def simulation(reps=100):
    rng=np.random.default_rng(730112)
    rows=[];allinfo=[]
    for n in [12,30]:
        for setting in ['Gaussian tree','Student t tree','Structural break','Factor misspecification']:
            for rep in range(reps):
                edges=[(int(rng.integers(i)),i) for i in range(1,n)]
                rho=np.eye(n)
                for i,j in edges:rho[i,j]=rho[j,i]=rng.uniform(.35,.75)
                sigma=rng.uniform(.008,.02,n)
                S=TreeRisk(rho,edges,sigma).covariance()
                target=S.copy()
                if setting=='Factor misspecification':
                    B=np.zeros((n,4));B[:,0]=rng.uniform(.25,.55,n)
                    for i in range(n):B[i,1+i%3]=rng.uniform(.35,.65)
                    Q=B@B.T+np.diag(rng.uniform(.25,.55,n))
                    s=np.sqrt(np.diag(Q));Q=Q/s[:,None]/s[None,:]
                    S=Q*sigma[:,None]*sigma[None,:];target=S.copy()
                elif setting=='Structural break':
                    # An external break may change an edge parameter and is not
                    # asserted to be contained in the estimated stress set.
                    trueR=S/sigma[:,None]/sigma[None,:]
                    base=TreeRisk(trueR,edges,sigma)
                    path=base.paths[int(rng.integers(len(base.paths)))];k=int(rng.integers(len(path.correlations)))
                    swap=(tuple(sorted((path.nodes[k],path.nodes[k+1]))),path.endpoints)
                    R=trueR.copy();u,v=path.endpoints;R[u,v]=R[v,u]=rng.uniform(.5,.85)
                    target=TreeRisk(R,base.exchanged_edges(swap),sigma).covariance()
                X=rng.normal(size=(126,n))@np.linalg.cholesky(S).T
                if setting=='Student t tree':X*=np.sqrt(3/rng.chisquare(5,size=(126,1)))
                C,tree=fitted_tree(X);prior={name:np.ones(n)/n for name in METHODS}
                weights,info=targets(C,tree,prior,.001)
                oracle,_=allocate(target,np.ones(n)/n,0.)
                popopt=float(oracle@target@oracle)
                for name,w in weights.items():
                    risk=float(w@target@w);stress=tree.query(w)[0]
                    turnover=float(np.abs(w-prior[name]).sum())
                    rows.append({'n':n,'setting':setting,'rep':rep,'method':name,
                        'population_variance':risk,'population_volatility':np.sqrt(252*risk),
                        'relative_population_variance':risk/popopt,'stress_variance':stress,
                        'two_way_turnover':turnover,'objective':MULTIPLIER*stress+.001*turnover})
                for name,x in info.items():allinfo.append({'n':n,'setting':setting,'rep':rep,'method':name,**x})
                if (rep+1)%25==0:print('simulation',n,setting,rep+1,flush=True)
            pd.DataFrame(rows).to_csv(ROOT/'results/simulation.csv',index=False)
            pd.DataFrame(allinfo).to_csv(ROOT/'results/simulation_solvers.csv',index=False)
    print('COMPLETE simulations',len(rows),flush=True)

def benchmark():
    rng=np.random.default_rng(82411);rows=[]
    for n in [16,32,64,96]:
        for shape in ['star','balanced','recursive','path']:
            if shape=='star':edges=[(0,i) for i in range(1,n)]
            elif shape=='balanced':edges=[((i-1)//2,i) for i in range(1,n)]
            elif shape=='path':edges=[(i-1,i) for i in range(1,n)]
            else:edges=[(int(rng.integers(i)),i) for i in range(1,n)]
            R=rng.uniform(-.7,.8,(n,n));R=np.triu(R,1);R+=R.T;np.fill_diagonal(R,1)
            t=time.perf_counter();tr=TreeRisk(R,edges);pre=time.perf_counter()-t
            path_times=[];naive_times=[];err=0
            for rep in range(3):
                w=rng.dirichlet(np.ones(n))
                t=time.perf_counter();v,_=tr.query(w);path_times.append(time.perf_counter()-t)
                t=time.perf_counter();u=tr.naive_messages_query(w);naive_times.append(time.perf_counter()-t)
                err=max(err,abs(u-v));assert err<1e-10
            row={'n':n,'shape':shape,'L':tr.L,'candidates':tr.candidate_count,
                'preprocessing_seconds':pre,'oracle_seconds':float(np.median(path_times)),
                'per_candidate_messages_seconds':float(np.median(naive_times)),
                'speedup':float(np.median(naive_times)/np.median(path_times)),'max_error':err}
            rows.append(row);print('benchmark',row,flush=True)
            pd.DataFrame(rows).to_csv(ROOT/'results/benchmark.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['simulation','empirical','benchmark']);parser.add_argument('--n',type=int,default=10);parser.add_argument('--reps',type=int,default=100)
    args=parser.parse_args()
    env={'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),
         'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','scikit-learn','matplotlib']},
         'threads':{name:os.environ.get(name,'not set') for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']}}
    (ROOT/'results/environment.json').write_text(json.dumps(env,indent=2))
    if args.mode=='simulation':simulation(args.reps)
    elif args.mode=='empirical':empirical(args.n)
    else:benchmark()
