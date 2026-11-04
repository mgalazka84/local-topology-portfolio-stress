"""Re-score the unchanged 4,000 simulation samples for review round two.

No portfolios are re-fitted. The decision penalty is c*||w-prior||_1,
not the exact self-financing historical fee. Both optimizing and heuristic
methods are evaluated by the same population objective.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t

KAPPA = 52.5
METHODS = ['Equal', 'HRP', 'Sample', 'Shrinkage', 'Tree', 'Robust']
SETTINGS = ['Gaussian tree', 'Student t tree', 'Structural break', 'Factor misspecification']


def build_outputs(sim=None, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    res, man = root/'results', root/'manuscript'
    res.mkdir(exist_ok=True); man.mkdir(exist_ok=True)
    sources = [res/f'sim_{n}_{k}.csv' for n in [12, 30] for k in range(4)]
    if sim is None:
        sim = pd.concat(pd.read_csv(p) for p in sources)
    sim = sim.copy()
    keys = ['n', 'setting', 'rep', 'cost', 'method']
    assert not sim.duplicated(keys).any()
    assert len(sim) == 48000
    assert sim.groupby(['n', 'setting', 'cost', 'method']).size().eq(500).all()
    assert set(sim.method) == set(METHODS) and set(sim.cost) == {0., .001}
    assert np.isfinite(sim[['population_variance', 'relative_population_variance', 'two_way_turnover']]).all().all()
    assert (sim.population_variance > 0).all()
    assert (sim.two_way_turnover >= -1e-12).all()
    assert (sim.two_way_turnover <= 2*(1-1/sim.n)+1e-7).all()
    optimum = sim.population_variance / sim.relative_population_variance
    pop_range = optimum.groupby([sim.n, sim.setting, sim.rep]).agg(lambda x: x.max()-x.min())
    assert pop_range.max() < 1e-15
    sim['risk_component_bps'] = 1e4*KAPPA*sim.population_variance
    sim['cost_penalty_bps'] = 1e4*sim.cost*sim.two_way_turnover
    sim['population_objective_bps'] = sim.risk_component_bps + sim.cost_penalty_bps
    assert sim.loc[sim.cost == 0, 'cost_penalty_bps'].eq(0).all()
    assert sim.loc[sim.method == 'Equal', 'two_way_turnover'].abs().max() < 1e-12
    summaries=[]; paired=[]
    metrics=['relative_population_variance', 'risk_component_bps', 'cost_penalty_bps', 'population_objective_bps']
    for (n, setting, cost, method), g in sim.groupby(['n','setting','cost','method'], sort=False):
        record={'n':int(n),'setting':setting,'cost':float(cost),'method':method,'replications':len(g)}
        for metric in metrics:
            record[metric+'_mean']=float(g[metric].mean())
            record[metric+'_se']=float(g[metric].std(ddof=1)/np.sqrt(len(g)))
        summaries.append(record)
    worst_identity_error=0.
    for (n, setting, cost), g in sim.groupby(['n','setting','cost'], sort=False):
        pivots={m:g.pivot(index='rep',columns='method',values=m) for m in metrics}
        for left,right in [('Robust','Tree'),('Robust','Shrinkage'),('Robust','Sample'),('Sample','Shrinkage'),('Sample','HRP')]:
            rd=pivots['risk_component_bps'][left]-pivots['risk_component_bps'][right]
            cd=pivots['cost_penalty_bps'][left]-pivots['cost_penalty_bps'][right]
            jd=pivots['population_objective_bps'][left]-pivots['population_objective_bps'][right]
            worst_identity_error=max(worst_identity_error,float((jd-rd-cd).abs().max()))
            for metric in metrics:
                p=pivots[metric]
                if metric == 'relative_population_variance':
                    delta=100*(p[left]/p[right]-1);unit='percent'
                else:
                    delta=p[left]-p[right];unit='bps'
                mean=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
                half=float(t.ppf(.975,len(delta)-1)*se)
                paired.append({'n':int(n),'setting':setting,'cost':float(cost),'comparison':left+' - '+right,
                    'metric':metric,'unit':unit,'replications':len(delta),'mean':mean,'paired_se':se,
                    'ci_low':mean-half,'ci_high':mean+half})
    summary=pd.DataFrame(summaries);pairs=pd.DataFrame(paired)
    summary.to_csv(res/'round2_simulation_summary.csv',index=False)
    pairs.to_csv(res/'round2_simulation_paired.csv',index=False)
    rows=[]
    panels=[(0.,'relative_population_variance','A. Relative population variance: no cost penalty'),
            (.001,'relative_population_variance','B. Relative population variance: 10 bps cost'),
            (.001,'cost_penalty_bps','C. Decision cost penalty: 10 bps cost (basis points)'),
            (.001,'population_objective_bps','D. Population risk-plus-cost objective: 10 bps cost (basis points)')]
    for cost,metric,label in panels:
        rows.append([r'\multicolumn{9}{l}{\textit{'+label+'}}'])
        for n in [12,30]:
            for setting,short in zip(SETTINGS,['Gaussian','Student $t$','Break','Factor']):
                g=summary[(summary.n==n)&(summary.setting==setting)&(summary.cost==cost)].set_index('method')
                d=pairs[(pairs.n==n)&(pairs.setting==setting)&(pairs.cost==cost)&
                        (pairs.comparison=='Robust - Tree')&(pairs.metric==metric)].iloc[0]
                rows.append([n,short,*[f'{g.loc[m,metric+"_mean"]:.3f}' for m in METHODS],
                             f'{d["mean"]:+.3f} ({d.paired_se:.3f})'])
    text=r'\begin{tabular}{llrrrrrrr}'+'\n'+r'\toprule'+'\n'
    text+=r'$n$ & Model & Equal & HRP & Sample & LW & Tree & Robust & $\Delta$ (SE) \\'+'\n'+r'\midrule'+'\n'
    text+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    text+='\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'
    (man/'revision_simulation.tex').write_text(text)
    ten=summary[summary.cost==.001]
    rankings=[]
    for (n,setting),g in ten.groupby(['n','setting'],sort=False):
        best_var=g.loc[g.relative_population_variance_mean.idxmin(),'method']
        best_j=g.loc[g.population_objective_bps_mean.idxmin(),'method']
        rankings.append({'n':int(n),'setting':setting,'minimum_mean_relative_variance':best_var,
                         'minimum_mean_population_objective':best_j})
    assert all(x['minimum_mean_relative_variance']=='Sample' for x in rankings)
    assert worst_identity_error < 1e-11
    verification={'allocation_records':len(sim),'independent_training_samples':4000,'new_fits':0,
        'kappa':KAPPA,'cost_penalty':'c*two_way_turnover','objective_bps':'10000*(52.5*population_variance+c*two_way_turnover)',
        'max_paired_decomposition_error_bps':worst_identity_error,
        'max_population_minimum_consistency_error':float(pop_range.max()),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'descriptive_rankings_at_10bps':rankings,'intervals':'paired t499, unadjusted descriptive Monte Carlo intervals'}
    (res/'round2_scoring_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return summary,pairs,verification


if __name__ == '__main__':
    _,_,checks=build_outputs()
    print(json.dumps(checks,indent=2))
