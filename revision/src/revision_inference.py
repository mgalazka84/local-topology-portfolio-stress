"""Studentized paired circular-block inference, with explicit fixed tuning.

Adaptation of the smooth-statistic framework of Ledoit and Wolf (2008, 2011):
log variance ratio and excess-return Sharpe difference. Original-sample SEs
use a Bartlett HAC (10 lags here); bootstrap SEs use complete resampled block
sums, as in their natural block-variance formula. We do not use prewhitening
or their block-size calibration. Block-size
sensitivity is 21/63/126 days; 63 is primary. No strategy refitting is claimed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import rfft, irfft, next_fast_len

ROOT=Path(__file__).resolve().parents[1]

def statistic_influence(x,y,kind):
    # Batches in rows, observations in columns.
    mx=x.mean(axis=-1,keepdims=True);my=y.mean(axis=-1,keepdims=True)
    dx=x-mx;dy=y-my;vx=(dx*dx).mean(axis=-1,keepdims=True);vy=(dy*dy).mean(axis=-1,keepdims=True)
    if kind=='log_variance_ratio':
        stat=np.log(vx/vy)[...,0];influence=dx*dx/vx-dy*dy/vy
    else:
        annual_scale=np.sqrt(252*(x.shape[-1]-1)/x.shape[-1])
        stat=(annual_scale*(mx/np.sqrt(vx)-my/np.sqrt(vy)))[...,0]
        influence=annual_scale*(dx/np.sqrt(vx)-mx*(dx*dx-vx)/(2*vx**1.5)
                              -dy/np.sqrt(vy)+my*(dy*dy-vy)/(2*vy**1.5))
    return stat,influence

def hac_se(influence,lags=None):
    z=influence-influence.mean(axis=-1,keepdims=True);n=z.shape[-1]
    if lags is None:lags=int(np.floor(4*(n/100)**(2/9)))
    f=rfft(z,n=next_fast_len(2*n),axis=-1)
    ac=irfft(f*np.conj(f),n=next_fast_len(2*n),axis=-1)[...,:lags+1]/n
    v=ac[...,0]+2*np.sum(ac[...,1:]*(1-np.arange(1,lags+1)/(lags+1)),axis=-1)
    if np.any(v<=0):raise RuntimeError('Nonpositive HAC variance')
    return np.sqrt(v/n)

def bootstrap(x,y,kind,block=63,reps=1999,seed=821104):
    rng=np.random.default_rng(seed);n=len(x)
    est,influence=statistic_influence(x[None,:],y[None,:],kind)
    est=float(est[0]);se=float(hac_se(influence)[0]);pivots=[]
    for start in range(0,reps,50):
        b=min(50,reps-start)
        starts=rng.integers(n,size=(b,int(np.ceil(n/block))))
        ids=((starts[:,:,None]+np.arange(block))%n).reshape(b,-1)[:,:n]
        bs,inf=statistic_influence(x[ids],y[ids],kind)
        complete=n//block
        block_sums=inf[:,:complete*block].reshape(b,complete,block).sum(axis=-1)
        bootstrap_se=np.sqrt(np.mean(block_sums**2,axis=-1)/(block*n))
        if np.any(bootstrap_se<=0):raise RuntimeError('Nonpositive bootstrap SE')
        pivots.extend((bs-est)/bootstrap_se)
    pivots=np.asarray(pivots)
    critical=float(np.quantile(np.abs(pivots),.95))
    return {'estimate':est,'se':se,'ci_low':est-critical*se,'ci_high':est+critical*se,
            'p_value':(1+int(np.sum(np.abs(pivots)>=abs(est/se))))/(reps+1),
            'block':block,'replicates':reps,'hac_lags':int(np.floor(4*(n/100)**(2/9)))}

def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.zeros(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return out

def verify():
    rng=np.random.default_rng(513);x=rng.normal(size=(2,100));y=.3*x+rng.normal(size=(2,100))
    for kind in ['log_variance_ratio','sharpe_difference']:
        a,ia=statistic_influence(x,y,kind);b,ib=statistic_influence(y,x,kind)
        assert np.max(abs(a+b))<1e-12 and np.max(abs(ia+ib))<1e-10
        c,_=statistic_influence(3*x,3*y,kind);assert np.max(abs(a-c))<1e-12
    z=rng.normal(size=(2,100));ac=np.array([np.sum(z[:,k:]*z[:,:100-k],axis=1)/100 for k in range(11)]).T
    z-=z.mean(axis=1,keepdims=True)
    brute=(np.sum(z*z,axis=1)/100+sum(2*(1-k/11)*np.sum(z[:,k:]*z[:,:100-k],axis=1)/100 for k in range(1,11)))/100
    assert np.max(abs(hac_se(z,lags=10)**2-brute))<1e-12
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])

def run():
    verify();rows=[]
    for n in [10,30,49]:
        df=pd.read_csv(ROOT/'results'/f'history_{n}_base_daily.csv')
        for other in ['Tree','Shrinkage']:
            for kind,col in [('log_variance_ratio','net'),('sharpe_difference','excess')]:
                p=df.pivot(index='date',columns='method',values=col)
                for block in [21,63,126]:
                    row={'panel':n,'comparison':'Robust - '+other,'statistic':kind,
                         **bootstrap(p.Robust.to_numpy(),p[other].to_numpy(),kind,block=block,seed=821104+n+block)}
                    rows.append(row);print(n,other,kind,block,round(row['p_value'],4),flush=True)
    out=pd.DataFrame(rows)
    # The 12 tests (3 panels x 2 contrasts x 2 statistics) form one family.
    out['p_holm']=out.groupby('block').p_value.transform(lambda x:holm(x.to_numpy()))
    out.to_csv(ROOT/'results/revision_inference.csv',index=False)

if __name__=='__main__':run()
