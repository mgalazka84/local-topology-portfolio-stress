"""Independent exhaustive and randomized numerical checks of the paper."""
import json,itertools,heapq,time
from pathlib import Path
import numpy as np
from tree_risk import TreeRisk,maximum_tree,allocate,solve_master,tangent_bound

ROOT=Path(__file__).resolve().parents[1]

def prufer_tree(code):
    n=len(code)+2;degrees=np.ones(n,dtype=int)
    for x in code: degrees[x]+=1
    leaves=[i for i in range(n) if degrees[i]==1];heapq.heapify(leaves);edges=[]
    for x in code:
        leaf=heapq.heappop(leaves);edges.append((leaf,x))
        degrees[x]-=1
        if degrees[x]==1:heapq.heappush(leaves,x)
    edges.append(tuple(leaves))
    return edges

def run():
    rng=np.random.default_rng(20261103)
    checked=0;tree_count=0;maxerr=0.;maxrank=0;min_eigen=1.
    for n in range(2,6):
        for code in itertools.product(range(n),repeat=n-2):
            edges=prufer_tree(code)
            R=rng.uniform(-.85,.85,(n,n));R=(R+R.T)/2;np.fill_diagonal(R,1)
            if tree_count%3==0 and n>2:R[0,1]=R[1,0]=0
            tr=TreeRisk(R,edges,sigma=rng.uniform(.5,2,n))
            w=rng.normal(size=n)
            _,_,records=tr.query(w,True);nominal=tr.covariance()
            for swap,value in records:
                S=tr.covariance(swap)
                err=abs(value-float(w@S@w));maxerr=max(maxerr,err)
                assert err<2e-10
                min_eigen=min(min_eigen,float(np.linalg.eigvalsh(S)[0]))
                rank=int(np.linalg.matrix_rank(S-nominal,tol=1e-10));maxrank=max(maxrank,rank)
                assert rank<=4
                checked+=1
            tree_count+=1
    for _ in range(100):
        n=int(rng.integers(6,16));edges=prufer_tree(rng.integers(n,size=n-2))
        R=rng.uniform(-.99,.99,(n,n));R=np.triu(R,1);R=R+R.T+np.eye(n)
        tr=TreeRisk(R,edges,sigma=rng.uniform(.2,3,n));w=rng.normal(size=n)
        best,_,records=tr.query(w,True)
        for swap,value in records:
            S=tr.covariance(swap);err=abs(value-w@S@w);maxerr=max(maxerr,float(err))
            assert err<2e-9;checked+=1
        assert abs(best-tr.naive_messages_query(w))<2e-9
        tree_count+=1
    # Independent full-scenario optimization, same analytic convex formulation.
    allocation_checks=[]
    for n in [4,6,8]:
        for cost in [0.,.005,.05]:
            X=rng.normal(size=(80,n));X+=rng.normal(size=(80,1))
            S=np.cov(X,rowvar=False);sig=np.sqrt(np.diag(S));R=S/sig[:,None]/sig[None,:]
            tr=TreeRisk(R,maximum_tree(R),sig);previous=rng.dirichlet(np.ones(n))
            w,info=allocate(tr.covariance(),previous,cost,tree=tr)
            all_scenarios=[tr.covariance(swap) for swap,_ in tr.query(w,True)[2]]
            full,status=solve_master(all_scenarios,previous,cost,ftol=1e-12)
            full_obj=max(full@A@full for A in all_scenarios)+cost*np.abs(full-previous).sum()
            local_obj=tr.query(w)[0]+cost*np.abs(w-previous).sum()
            lower,_=tangent_bound(all_scenarios,full,previous,cost)
            assert abs(local_obj-full_obj)<1e-6
            assert lower<=local_obj+1e-8
            assert info['lower']<=full_obj+1e-8
            allocation_checks.append({'n':n,'cost':cost,'objective_difference':float(abs(local_obj-full_obj)),
                                      'full_gap':float(full_obj-lower),**info})
    # Rectangular nonnegative correlation uncertainty reaches the upper corner.
    interval_checks=0
    for _ in range(20):
        n=4;upper=rng.uniform(.15,.8,(n,n));upper=np.triu(upper,1);upper+=upper.T;np.fill_diagonal(upper,1)
        lower=upper*.5;np.fill_diagonal(lower,1)
        edges=prufer_tree(rng.integers(n,size=n-2));w=rng.dirichlet(np.ones(n))
        for bits in itertools.product([0,1],repeat=3):
            R=upper.copy()
            for (i,j),b in zip(edges,bits):R[i,j]=R[j,i]=upper[i,j] if b else lower[i,j]
            lo=TreeRisk(R,edges).risk(w);hi=TreeRisk(upper,edges).risk(w)
            assert lo<=hi+1e-12;interval_checks+=1
    # Global redundancy for a path-consistent positive correlation matrix.
    redundancy_checks=0
    for _ in range(10):
        n=5;edges=prufer_tree(rng.integers(n,size=n-2));R=np.eye(n)
        for i,j in edges:R[i,j]=R[j,i]=rng.uniform(.1,.9)
        R0=TreeRisk(R,edges).covariance();w=rng.dirichlet(np.ones(n));base=float(w@R0@w)
        for code in itertools.product(range(n),repeat=n-2):
            value=TreeRisk(R0,prufer_tree(code)).risk(w)
            assert value<=base+1e-12;redundancy_checks+=1
    result={'trees':tree_count,'dense_scenario_comparisons':checked,'max_absolute_risk_error':maxerr,
            'largest_observed_exchange_rank':maxrank,'minimum_eigenvalue_small_cases':min_eigen,
            'full_scenario_allocation_checks':allocation_checks,'interval_corner_checks':interval_checks,
            'global_redundancy_checks':redundancy_checks,'seed':20261103,'status':'PASS'}
    (ROOT/'results/verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='full_scenario_allocation_checks'},indent=2))

if __name__=='__main__':run()
