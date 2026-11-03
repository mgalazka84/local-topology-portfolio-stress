"""Check archived output integrity and reproduce the historical accounting ledger."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def run():
    s=pd.read_csv(ROOT/'results/simulation.csv')
    assert len(s)==4000 and not s.isna().any().any()
    assert s.groupby(['n','setting','rep']).size().eq(5).all()
    fits=pd.read_csv(ROOT/'results/simulation_solvers.csv')
    assert len(fits)==2400 and fits.gap_scaled.max()<=2e-7
    report={'simulation_samples':800,'simulation_method_rows':len(s),
            'simulation_fits':len(fits),'max_simulation_gap_scaled':float(fits.gap_scaled.max()),'panels':[]}
    for n in [10,30]:
        d=pd.read_csv(ROOT/'results'/f'empirical_{n}_daily.csv')
        fit=pd.read_csv(ROOT/'results'/f'empirical_{n}_solvers.csv')
        h=pd.DataFrame(json.loads((ROOT/'results'/f'empirical_{n}_holdings.json').read_text()))
        x=pd.read_csv(ROOT/'data'/f'industry_{n}.csv',index_col=0)
        assert len(d)==6539*15 and d.date.nunique()==6539
        assert not d.isna().any().any() and not fit.isna().any().any()
        assert len(fit)==312*9 and fit.gap_scaled.max()<=2e-7
        assert len(h)==312*15 and h.date.nunique()==312
        weights=np.array(h.weights.tolist())
        assert weights.min()>=-1e-14 and np.max(np.abs(weights.sum(axis=1)-1))<1e-12
        assert (d.fee>=-1e-15).all() and (d.turnover>=-1e-15).all()
        assert np.max(np.abs(d.fee-d.cost*d.turnover))<1e-14
        assert np.max(np.abs(d.net-((1-d.fee)*(1+d.gross)-1)))<1e-14
        assert np.min(h.stress_variance-h.nominal_variance)>-1e-14
        gross_error=0.;net_error=0.;fee_error=0.
        for (cost,method),g in d.groupby(['cost','method'],sort=False):
            targets=h[(h.cost==cost)&(h.method==method)].set_index('date')
            w=np.ones(n)/n
            for row in g.itertuples():
                k=0.
                if row.date in targets.index:
                    target=np.array(targets.loc[row.date,'weights'])
                    # Independent monotone-root bisection, not the production fixed-point rule.
                    lo=0.;hi=1.
                    for _ in range(50):
                        mid=(lo+hi)/2
                        if mid-cost*np.abs((1-mid)*target-w).sum()>0:hi=mid
                        else:lo=mid
                    k=(lo+hi)/2
                    w=target
                r=x.loc[row.date].to_numpy()
                gross=float(w@r)
                net=(1-k)*(1+gross)-1
                gross_error=max(gross_error,abs(gross-row.gross))
                net_error=max(net_error,abs(net-row.net))
                fee_error=max(fee_error,abs(k-row.fee))
                w=w*(1+r)/(1+gross)
        assert max(gross_error,net_error,fee_error)<1e-12
        report['panels'].append({'assets':n,'daily_observations':6539,'rebalances':312,
            'method_cost_series':15,'optimization_fits':len(fit),
            'max_gap_scaled':float(fit.gap_scaled.max()),
            'ledger_max_gross_error':gross_error,'ledger_max_net_error':net_error,
            'ledger_max_fee_error':fee_error})
    report['status']='PASS'
    (ROOT/'results/release_integrity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()
