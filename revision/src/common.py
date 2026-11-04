"""Shared covariance fitting and independent wealth-accounting utilities."""
import numpy as np
from sklearn.covariance import LedoitWolf
from tree_risk import TreeRisk,maximum_tree

def fitted_tree(X):
    C=LedoitWolf().fit(X).covariance_
    sigma=np.sqrt(np.diag(C));rho=C/sigma[:,None]/sigma[None,:]
    rho=np.clip(rho,-.999,.999);np.fill_diagonal(rho,1)
    tree=TreeRisk(rho,maximum_tree(rho),sigma)
    return C,tree

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

