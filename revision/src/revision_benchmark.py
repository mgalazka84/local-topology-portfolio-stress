"""Strong comparator, sparse permitted graphs, and independent cross-checks."""
from pathlib import Path
import time, json
import numpy as np
import pandas as pd
from tree_risk import TreeRisk, canonical, maximum_tree
from cut_baseline import CutReroot

ROOT=Path(__file__).resolve().parents[1]

def knn_edges(rho, edges, k=4):
    pairs=set(canonical(edges)); n=len(rho)
    for i in range(n):
        candidates=sorted(((-abs(rho[i,j]),j) for j in range(n) if j!=i))
        for _, j in candidates[:k]: pairs.add(tuple(sorted((i,j))))
    return sorted(pairs)

def verify():
    rng=np.random.default_rng(20261104); errors=[]; count=0
    for n in [3,5,8,12]:
        for rep in range(12):
            edges=[(int(rng.integers(i)),i) for i in range(1,n)]
            R=rng.uniform(-.95,.95,(n,n));R=np.triu(R,1);R+=R.T;np.fill_diagonal(R,1)
            for i,j in edges[::4]:R[i,j]=R[j,i]=0
            allowed=None if rep%2 else knn_edges(R,edges,2)
            tree=TreeRisk(R,edges,allowed=allowed); w=rng.normal(size=n)
            methods=[tree.query,tree.query_prefix,CutReroot(tree).query]
            records=[dict(fn(w,True)[2]) for fn in methods]
            assert all(set(x)==set(records[0]) for x in records)
            for sw in records[0]:
                exact=float(w@tree.covariance(sw)@w)
                errors.extend(abs(x[sw]-exact) for x in records);count+=1
    # Extreme correlations exercise cancellation; all remain strictly below one.
    stability=[]
    for r in [0.,-.999999,.999999]:
        n=30;R=np.full((n,n),r);np.fill_diagonal(R,1)
        tr=TreeRisk(R,[(i-1,i) for i in range(1,n)],allowed=[(i,j) for i in range(n) for j in range(i+1,min(n,i+4))])
        w=rng.normal(size=n); d=tr.query(w,True)[2];p=dict(tr.query_prefix(w,True)[2])
        err=max(abs(q-float(w@tr.covariance(sw)@w)) for sw,q in d)
        stability.append({'correlation':r,'direct_dense_max_error':err,
            'direct_prefix_max_error':max(abs(q-p[sw]) for sw,q in d)})
    result={'status':'PASS','reference_trees':48,'scenarios':count,
        'three_method_dense_checks':len(errors),'max_absolute_error':max(errors),'stability':stability}
    assert max(errors)<2e-10 and max(x['direct_dense_max_error'] for x in stability)<2e-8
    (ROOT/'results/revision_verification.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)

def benchmark():
    rng=np.random.default_rng(82412);rows=[]
    for n in [16,32,64,128,256,512]:
        for shape in ['star','balanced','recursive','path']:
            edges=[(0 if shape=='star' else (i-1)//2 if shape=='balanced' else i-1
                    if shape=='path' else int(rng.integers(i)),i) for i in range(1,n)]
            base=np.eye(n)
            for i,j in edges:base[i,j]=base[j,i]=rng.uniform(.35,.8)
            # Sparse selection by nearest correlations of a tree population;
            # noisy allowed values need not satisfy the path-product identities.
            R0=TreeRisk(base,edges,allowed=edges).covariance()
            noise=rng.uniform(-.04,.04,(n,n)); noise=(noise+noise.T)/2
            R=np.clip(R0+noise,-.95,.95);np.fill_diagonal(R,1)
            for mode in ['complete','knn4']:
                if mode=='complete' and n>256:continue
                t=time.perf_counter();allowed=None if mode=='complete' else knn_edges(R,edges,4)
                select=time.perf_counter()-t
                t=time.perf_counter();tr=TreeRisk(R,edges,allowed=allowed);path_pre=time.perf_counter()-t
                t=time.perf_counter();cut=CutReroot(tr);cut_pre=time.perf_counter()-t
                times={x:[] for x in ['direct','prefix','cut']}; error=0.
                funcs={'direct':tr.query,'prefix':tr.query_prefix,'cut':cut.query}
                for rep in range(4):
                    w=rng.dirichlet(np.ones(n));values=[]
                    for name in rng.permutation(list(funcs)):
                        t=time.perf_counter();q,_=funcs[name](w);dt=time.perf_counter()-t
                        if rep:times[name].append(dt)
                        values.append(q)
                    error=max(error,max(values)-min(values))
                assert error<1e-10
                row={'n':n,'shape':shape,'graph':mode,'L':tr.L,'active_cuts':len(cut.cuts),
                    'permitted_edges':n-1+len(tr.paths),'graph_selection_seconds':select,
                    'path_preparation_seconds':path_pre,'cut_additional_preparation_seconds':cut_pre,
                    **{k+'_seconds':float(np.median(v)) for k,v in times.items()},'max_error':error}
                row['cut_over_direct']=row['cut_seconds']/row['direct_seconds']
                rows.append(row);pd.DataFrame(rows).to_csv(ROOT/'results/revision_benchmark.csv',index=False)
                print('benchmark',n,shape,mode,tr.L,round(row['cut_over_direct'],2),flush=True)

if __name__=='__main__':
    verify();benchmark()
