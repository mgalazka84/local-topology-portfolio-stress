"""Independent replay of all historical dollar trades and return ledgers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from revision_experiments import CONFIGS

ROOT=Path(__file__).resolve().parents[1]

def run():
    records=[];max_error=0.;count=0
    for summary_path in sorted((ROOT/'results').glob('history_*_summary.csv')):
        summary=pd.read_csv(summary_path);n=int(summary.panel.iloc[0]);config=summary.config.iloc[0]
        prefix=f'history_{n}_{config}';cost=CONFIGS[config][3]
        X=pd.read_csv(ROOT/'data'/f'industry_{n}.csv',index_col=0,parse_dates=True)
        daily=pd.read_csv(ROOT/'results'/f'{prefix}_daily.csv',parse_dates=['date'])
        holdings=json.loads((ROOT/'results'/f'{prefix}_holdings.json').read_text())
        assert not daily.duplicated(['date','method']).any()
        for name,g in daily.groupby('method'):
            assert len(g)==6539 and g.date.min()==pd.Timestamp('2000-01-03') and g.date.max()==pd.Timestamp('2025-12-31')
            target={h['date']:h for h in holdings if h['method']==name};w=np.ones(n)/n;errors=[]
            for row in g.itertuples():
                key=str(row.date.date());fee=0.;turn=0.
                if key in target:
                    h=target[key];new=np.array(h['weights']);prior=np.array(h['prior'])
                    assert new.min()>=0 and abs(new.sum()-1)<1e-12
                    errors.append(float(np.max(abs(prior-w))))
                    fee=h['fee'];turn=float(np.abs((1-fee)*new-w).sum())
                    assert fee<=2*cost/(1+cost)+1e-12
                    errors.extend([abs(fee-cost*turn),abs(turn-h['turnover'])])
                    w=new
                returns=X.loc[row.date].to_numpy();gross=float(w@returns);net=(1-fee)*(1+gross)-1
                errors.extend([abs(gross-row.gross),abs(net-row.net),abs(fee-row.fee),abs(turn-row.turnover)])
                w=w*(1+returns)/(1+gross);count+=1
            err=max(errors);assert err<2e-11;max_error=max(max_error,err)
            records.append({'panel':n,'config':config,'method':name,'rows':len(g),'decisions':len(target),'max_error':err})
        sol=pd.read_csv(ROOT/'results'/f'{prefix}_solvers.csv')
        assert sol.gap_scaled.le(2e-7).all() and np.isfinite(sol.gap_scaled).all()
    result={'status':'PASS','ledger_rows':count,'max_absolute_error':max_error,'strategy_runs':len(records),'runs':records}
    (ROOT/'results/revision_integrity.json').write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k!='runs'},flush=True)

if __name__=='__main__':run()
