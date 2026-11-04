"""Exact one-exchange tree risk queries and numerically checked convex allocation.

Copyright (c) 2026 Marek Galazka and Hanna Wdowicka.
Theory uses exact arithmetic; computations use ordinary IEEE binary64.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize, linprog


def canonical(edges):
    return tuple(sorted(tuple(sorted((int(i), int(j)))) for i, j in edges))


def maximum_tree(rho):
    """Gaussian Chow--Liu tree, with deterministic tie breaking."""
    n = len(rho)
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    candidates = sorted((-float(rho[i,j]**2),i,j)
                        for i in range(n) for j in range(i+1,n))
    edges = []
    for _, i, j in candidates:
        a,b = root(i),root(j)
        if a != b:
            parent[a] = b
            edges.append((i,j))
            if len(edges) == n-1: break
    return canonical(edges)


@dataclass
class PathPlan:
    endpoints: tuple
    nodes: np.ndarray
    forward: np.ndarray
    backward: np.ndarray
    correlations: np.ndarray
    prefix: np.ndarray
    suffix: np.ndarray
    valid: np.ndarray


class TreeRisk:
    def __init__(self, rho, edges, sigma=None, allowed=None, score_tolerance=np.inf):
        self.rho = np.asarray(rho,dtype=float)
        self.n = len(rho)
        self.sigma = np.ones(self.n) if sigma is None else np.asarray(sigma,dtype=float)
        self.edges = canonical(edges)
        assert len(self.edges) == self.n-1
        self.adj = [[] for _ in range(self.n)]
        self.directed = {}
        for k,(i,j) in enumerate(self.edges):
            r = self.rho[i,j]
            assert abs(r) < 1
            self.adj[i].append((j,r,2*k,2*k+1))
            self.adj[j].append((i,r,2*k+1,2*k))
            self.directed[i,j] = 2*k
            self.directed[j,i] = 2*k+1
        self.parent = np.full(self.n,-1,dtype=int)
        self.depth = np.zeros(self.n,dtype=int)
        order = [0]
        self.parent[0] = 0
        for i in order:
            for j,_,_,_ in self.adj[i]:
                if self.parent[j] == -1:
                    self.parent[j] = i
                    self.depth[j] = self.depth[i]+1
                    order.append(j)
        assert len(order) == self.n, 'Edges must form a connected tree'
        self.order = order
        base = set(self.edges)
        if allowed is None:
            allowed = [(i,j) for i in range(self.n) for j in range(i+1,self.n)]
        allowed = canonical(allowed)
        self.paths = []
        self.L = 0
        self.candidate_count = 0
        for u,v in allowed:
            if (u,v) in base: continue
            assert abs(self.rho[u,v]) < 1
            left,right = [u],[v]
            i,j = u,v
            while i != j:
                if self.depth[i] >= self.depth[j]:
                    i = self.parent[i]
                    left.append(i)
                else:
                    j = self.parent[j]
                    right.append(j)
            nodes = np.array(left + right[-2::-1],dtype=int)
            fwd = np.array([self.directed[i,j] for i,j in zip(nodes[:-1],nodes[1:])])
            bwd = np.array([self.directed[j,i] for i,j in zip(nodes[:-1],nodes[1:])])
            r = self.rho[nodes[:-1],nodes[1:]]
            pre = np.r_[1., np.cumprod(r)]
            suf = np.r_[np.cumprod(r[::-1])[::-1],1.]
            score_loss = -.5*np.log1p(-r*r) + .5*np.log1p(-self.rho[u,v]**2)
            valid = score_loss <= score_tolerance + 1e-14
            self.paths.append(PathPlan((u,v),nodes,fwd,bwd,r,pre,suf,valid))
            self.L += len(r)
            self.candidate_count += int(valid.sum())

    def messages(self, exposures):
        a = np.asarray(exposures,dtype=float)
        msg = np.zeros(2*(self.n-1))
        for i in reversed(self.order[1:]):
            p = self.parent[i]
            value = a[i]
            for j,r,outgoing,incoming in self.adj[i]:
                if j != p: value += r*msg[incoming]
            msg[self.directed[i,p]] = value
        total = np.zeros(self.n)
        for i in self.order:
            value = a[i] + sum(r*msg[inc] for _,r,_,inc in self.adj[i])
            total[i] = value
            for j,r,outgoing,incoming in self.adj[i]:
                if self.parent[j] == i and j != 0:
                    msg[outgoing] = value - r*msg[incoming]
        return msg,total

    def risk(self, weights):
        a = self.sigma*np.asarray(weights)
        _,s = self.messages(a)
        return float(a@s)

    def query_prefix(self, weights, return_all=False):
        a = self.sigma*np.asarray(weights)
        msg,s = self.messages(a)
        nominal = float(a@s)
        best,best_swap = nominal,None
        records = [(None,nominal)] if return_all else None
        for path in self.paths:
            b = s[path.nodes].copy()
            b[:-1] -= path.correlations*msg[path.backward]
            b[1:] -= path.correlations*msg[path.forward]
            left = np.cumsum(b*path.prefix)[:-1]
            right = np.cumsum((b*path.suffix)[::-1])[::-1][1:]
            old = path.correlations*msg[path.forward]*msg[path.backward]
            new = self.rho[path.endpoints]*left*right
            risks = nominal+2*(new-old)
            indices = np.flatnonzero(path.valid)
            if not len(indices): continue
            k = indices[np.argmax(risks[indices])]
            if risks[k] > best:
                best = float(risks[k])
                best_swap = (tuple(sorted((path.nodes[k],path.nodes[k+1]))),path.endpoints)
            if return_all:
                for k in indices:
                    swap = (tuple(sorted((path.nodes[k],path.nodes[k+1]))),path.endpoints)
                    records.append((swap,float(risks[k])))
        return (best,best_swap,records) if return_all else (best,best_swap)

    def query(self, weights, return_all=False):
        a = self.sigma*np.asarray(weights)
        msg,s = self.messages(a)
        nominal = float(a@s)
        best,best_swap = nominal,None
        records = [(None,nominal)] if return_all else None
        for path in self.paths:
            # Subtract the opposite component's contribution to R_T a.
            # No division, including when an edge correlation is zero.
            left = s[path.endpoints[0]] - path.prefix[1:] * msg[path.backward]
            right = s[path.endpoints[1]] - path.suffix[:-1] * msg[path.forward]
            old = path.correlations*msg[path.forward]*msg[path.backward]
            new = self.rho[path.endpoints]*left*right
            risks = nominal+2*(new-old)
            indices = np.flatnonzero(path.valid)
            if not len(indices): continue
            k = indices[np.argmax(risks[indices])]
            if risks[k] > best:
                best = float(risks[k])
                best_swap = (tuple(sorted((path.nodes[k],path.nodes[k+1]))),path.endpoints)
            if return_all:
                for k in indices:
                    swap = (tuple(sorted((path.nodes[k],path.nodes[k+1]))),path.endpoints)
                    records.append((swap,float(risks[k])))
        return (best,best_swap,records) if return_all else (best,best_swap)

    def exchanged_edges(self,swap=None):
        if swap is None: return self.edges
        old,new = swap
        return canonical([e for e in self.edges if e != old]+[new])

    def covariance(self,swap=None):
        """Independent dense construction by one traversal per source vertex."""
        edges = self.exchanged_edges(swap)
        adj = [[] for _ in range(self.n)]
        for i,j in edges:
            adj[i].append(j); adj[j].append(i)
        R = np.eye(self.n)
        for source in range(self.n):
            stack = [(source,-1,1.)]
            while stack:
                i,p,prod = stack.pop()
                R[source,i] = prod
                for j in adj[i]:
                    if j != p: stack.append((j,i,prod*self.rho[i,j]))
        return R*self.sigma[:,None]*self.sigma[None,:]

    def naive_messages_query(self,weights):
        """O(n) per scenario comparator: no dense matrices or path caching."""
        a = self.sigma*np.asarray(weights)
        def single(edges):
            adj = [[] for _ in range(self.n)]
            for i,j in edges:
                adj[i].append(j);adj[j].append(i)
            parent = np.full(self.n,-1,dtype=int);parent[0]=0;order=[0]
            for i in order:
                for j in adj[i]:
                    if parent[j] < 0: parent[j]=i;order.append(j)
            mass=a.copy();risk=float(a@a)
            for j in reversed(order[1:]):
                i=parent[j];r=self.rho[i,j]
                risk += 2*r*mass[i]*mass[j]
                mass[i] += r*mass[j]
            return risk
        best=single(self.edges)
        for p in self.paths:
            for k in np.flatnonzero(p.valid):
                swap=(tuple(sorted((p.nodes[k],p.nodes[k+1]))),p.endpoints)
                candidate=[e for e in self.edges if e != swap[0]]+[swap[1]]
                best=max(best,single(candidate))
        return best


def solve_master(matrices, previous, cost, start=None, ftol=1e-11):
    A=np.asarray(matrices); n=len(previous)
    c=np.broadcast_to(np.asarray(cost,dtype=float),n)
    w=previous.copy() if start is None else start.copy()
    w=np.maximum(w,0);w/=w.sum()
    if len(A)==1 and np.all(c==0):
        H=A[0]
        result=minimize(lambda x:float(x@H@x),w,jac=lambda x:2*H@x,method='SLSQP',
            bounds=[(0,1)]*n,constraints=[{'type':'eq','fun':lambda x:x.sum()-1,
            'jac':lambda x:np.ones(n)}],options={'ftol':ftol,'maxiter':500})
        w=np.maximum(result.x,0);w/=w.sum()
        support=np.flatnonzero(w>1e-8)
        try:
            z=np.linalg.solve(H[np.ix_(support,support)],np.ones(len(support)))
            z/=z.sum()
            candidate=np.zeros(n);candidate[support]=z
            gradient=2*H@candidate
            if z.min()>=-1e-12 and gradient.min()>=float(candidate@gradient)-1e-10:
                w=np.maximum(candidate,0);w/=w.sum()
        except np.linalg.LinAlgError:
            pass
        return w,{'success':bool(result.success),'message':str(result.message),
                  'iterations':int(result.nit),'constraint_residual':0.}
    x0=np.r_[w,np.abs(w-previous),max(np.einsum('i,kij,j->k',w,A,w))]
    def objective(x): return x[-1]+c@x[n:2*n]
    grad=np.r_[np.zeros(n),c,1.]
    def inequalities(x):
        w=x[:n]; z=x[n:2*n]
        return np.r_[x[-1]-np.einsum('i,kij,j->k',w,A,w),z-w+previous,z+w-previous]
    def jacobian(x):
        result=np.zeros((len(A)+2*n,2*n+1))
        result[:len(A),:n]=-2*np.einsum('kij,j->ki',A,x[:n])
        result[:len(A),-1]=1
        result[len(A):len(A)+n,:n]=-np.eye(n)
        result[len(A):len(A)+n,n:2*n]=np.eye(n)
        result[len(A)+n:,:n]=np.eye(n)
        result[len(A)+n:,n:2*n]=np.eye(n)
        return result
    result=minimize(objective,x0,jac=lambda x:grad,method='SLSQP',
        bounds=[(0,1)]*n+[(0,None)]*n+[(0,None)],
        constraints=[{'type':'eq','fun':lambda x: x[:n].sum()-1,
                      'jac':lambda x:np.r_[np.ones(n),np.zeros(n+1)]},
                     {'type':'ineq','fun':inequalities,'jac':jacobian}],
        options={'ftol':ftol,'maxiter':500,'disp':False})
    w=np.maximum(result.x[:n],0);w/=w.sum()
    return w, {'success':bool(result.success),'message':str(result.message),
               'iterations':int(result.nit),'constraint_residual':float(np.min(inequalities(result.x)))}


def tangent_bound(matrices,w,previous,cost,return_status=False):
    """Optimize a globally valid affine-minorant lower bound on the simplex.

    A certificate in exact arithmetic; HiGHS results are floating-point values.
    """
    A=np.asarray(matrices);k=len(A);n=len(w)
    q=np.einsum('i,kij,j->k',w,A,w)
    g=2*np.einsum('kij,j->ki',A,w)
    c=np.broadcast_to(np.asarray(cost,dtype=float),n)
    obj=np.r_[q,previous,-1.]
    Aub=np.column_stack((-g.T,-np.eye(n),np.ones(n)))
    eq=np.r_[np.ones(k),np.zeros(n+1)][None,:]
    failures=[]
    for method,options in [('highs',{}),('highs-ipm',{'presolve':False})]:
        res=linprog(obj,A_ub=Aub,b_ub=np.zeros(n),A_eq=eq,b_eq=[1.],
            bounds=[(0,None)]*k+[(-float(x),float(x)) for x in c]+[(None,None)],
            method=method,options=options)
        if res.success:break
        failures.append(str(res.message))
    # Restore simple dual feasibility explicitly; no claim of directed rounding.
    if res.success:
        alpha=np.maximum(res.x[:k],0);alpha/=alpha.sum()
        h=np.minimum(np.maximum(res.x[k:k+n],-c),c)
    else:
        # A weak, explicitly feasible lower bound still exists. The separate
        # quadratic dual bound must close the gap, or allocation raises.
        alpha=np.ones(k)/k;h=np.zeros(n)
    eta=float(np.min(g.T@alpha+h))
    lower=eta-alpha@q-h@previous
    if return_status:return float(lower),alpha,{'failures':failures,'solved':bool(res.success)}
    return float(lower),alpha


def allocate(covariance, previous, cost=0., tree=None, risk_multiplier=1.,
             tol=2e-7, backend="clarabel"):
    """Convex allocation; accept only an independently checked global gap.

    All restricted solver calls, including refinements, contribute to diagnostics.
    The certificate is mathematical; the implementation is ordinary binary64.
    """
    from collections import Counter
    from conic_master import solve_master_conic
    solver = solve_master_conic if backend == "clarabel" else solve_master
    start_time = time.perf_counter()
    H0 = np.asarray(covariance)*risk_multiplier
    scale = float(np.mean(np.diag(H0)))
    matrices = [H0/scale]; swaps = [None]
    cost_scaled = np.asarray(cost)/scale
    w = previous.copy(); masters = []; lp_statuses=[]
    for step in range(100):
        added = False
        for accuracy in [1e-9, 1e-11, 1e-13]:
            w, status = solver(matrices, previous, cost_scaled, start=w, ftol=accuracy)
            masters.append(status)
            if tree is None:
                value = float(w@matrices[0]@w); swap = None
            else:
                value, swap = tree.query(w); value *= risk_multiplier/scale
            upper = value+float(np.broadcast_to(cost_scaled,len(w))@np.abs(w-previous))
            lower, _, lpstatus = tangent_bound(matrices, w, previous, cost_scaled,return_status=True)
            lp_statuses.append(lpstatus)
            lower = max(lower, status.get('quadratic_lower', -np.inf))
            raw_gap = upper-lower
            if raw_gap < -1e-9:
                raise RuntimeError(f'Lower bound exceeds upper bound: {raw_gap}')
            gap = max(0.,raw_gap)
            if gap <= tol:
                return w, {'gap_scaled':gap, 'gap':gap*scale,
                    'upper':upper*scale, 'lower':lower*scale,
                    'scenarios':len(matrices), 'steps':step+1,
                    'seconds':time.perf_counter()-start_time, 'backend':backend,
                    'master_success':all(x['success'] for x in masters),
                    'master_calls':len(masters),
                    'non_success_calls':sum(not x['success'] for x in masters),
                    'master_statuses':dict(Counter(x['message'] for x in masters)),
                    'lp_non_success_calls':sum(len(x['failures']) for x in lp_statuses),
                    'lp_unresolved_calls':sum(not x['solved'] for x in lp_statuses),
                    'master_max_primal_residual':max(abs(x['constraint_residual']) for x in masters)}
            if swap not in swaps:
                matrices.append(tree.covariance(swap)*risk_multiplier/scale)
                swaps.append(swap); added=True; break
        if not added:
            raise RuntimeError(f'Unresolved {backend} master: gap {gap:g}; {status}')
    raise RuntimeError('Constraint generation did not converge in 100 iterations')


def hrp(covariance):
    from scipy.cluster.hierarchy import linkage,leaves_list
    from scipy.spatial.distance import squareform
    cov=np.asarray(covariance);sigma=np.sqrt(np.diag(cov))
    rho=np.clip(cov/sigma[:,None]/sigma[None,:],-1,1)
    distance=np.sqrt(np.maximum((1-rho)/2,0));np.fill_diagonal(distance,0)
    order=leaves_list(linkage(squareform(distance,checks=False),method='single')).tolist()
    weights=np.ones(len(cov));clusters=[order]
    def cluster_variance(ids):
        sub=cov[np.ix_(ids,ids)]
        w=1/np.diag(sub);w/=w.sum()
        return float(w@sub@w)
    while clusters:
        following=[]
        for ids in clusters:
            if len(ids)<2: continue
            split=len(ids)//2;left,right=ids[:split],ids[split:]
            va,vb=cluster_variance(left),cluster_variance(right)
            alpha=vb/(va+vb)
            weights[left]*=alpha;weights[right]*=1-alpha
            following.extend((left,right))
        clusters=following
    return weights/weights.sum()
