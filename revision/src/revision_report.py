"""Generate all revision tables and exact, vector scientific figures."""
from pathlib import Path
import json, ast, sys, os, platform, importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from revision_experiments import METHODS, SETTINGS

ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';MAN=ROOT/'manuscript';FIG=ROOT/'figures'
MAN.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
COLORS={'Equal':'#777777','HRP':'#E69F00','Sample':'#56B4E9','Shrinkage':'#0072B2',
        'Rank':'#009E73','Tree':'#CC79A7','Robust':'#D55E00'}
LINES={'Equal':':','HRP':'--','Sample':(0,(1,2,4,2)),'Shrinkage':'-',
       'Rank':(0,(5,1,1,1)),'Tree':'--','Robust':'-.'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
    'savefig.bbox':'tight','pdf.fonttype':42,'font.family':'DejaVu Sans'})

def table(name,columns,rows,align=None):
    text='\\begin{tabular}{'+(align or 'l'*len(columns))+'}\n\\toprule\n'+' & '.join(columns)+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n'
    (MAN/name).write_text(text)

def run():
    sim=pd.concat(pd.read_csv(RES/f'sim_{n}_{k}.csv') for n in [12,30] for k in range(4))
    assert len(sim)==48000 and sim.groupby(['n','setting','cost','method']).size().eq(500).all()
    sim.to_csv(RES/'simulation.csv',index=False)
    paired=[]
    for (n,setting,cost),g in sim.groupby(['n','setting','cost']):
        p=g.pivot(index='rep',columns='method',values='population_variance')
        for other in ['Tree','Shrinkage']:
            d=100*(p.Robust/p[other]-1);se=d.std(ddof=1)/np.sqrt(len(d));crit=t.ppf(.975,len(d)-1)
            paired.append({'n':n,'setting':setting,'cost':cost,'comparison':'Robust - '+other,
                'mean_percent':d.mean(),'paired_se':se,'ci_low':d.mean()-crit*se,'ci_high':d.mean()+crit*se})
    paired=pd.DataFrame(paired);paired.to_csv(RES/'simulation_paired.csv',index=False)
    rows=[]
    for cost in [0.,.001]:
        rows.append([r'\multicolumn{9}{l}{\textit{'+('No trading penalty' if cost==0 else 'Trading penalty: 10 bps')+'}}'])
        for n in [12,30]:
            for setting,short in zip(SETTINGS,['Gaussian','Student $t$','Break','Factor']):
                g=sim[(sim.n==n)&(sim.setting==setting)&(sim.cost==cost)]
                means=g.groupby('method').relative_population_variance.mean()
                d=paired[(paired.n==n)&(paired.setting==setting)&(paired.cost==cost)&(paired.comparison=='Robust - Tree')].iloc[0]
                rows.append([n,short,*[f'{means[m]:.3f}' for m in ['Equal','HRP','Sample','Shrinkage','Tree','Robust']],f'{d.mean_percent:+.3f} ({d.paired_se:.3f})'])
    table('revision_simulation.tex',['$n$','Model','Equal','HRP','Sample','LW','Tree','Robust',r'$\Delta$ (SE), \%'],rows,'llrrrrrrr')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.8),sharey=True)
    for ax,cost in zip(axes,[0.,.001]):
        labels=[]
        for i,(n,setting) in enumerate((n,s) for n in [12,30] for s in SETTINGS):
            labels.append(f'{n}: '+setting.replace(' misspecification','').replace('Structural break','Break'))
            for other,offset,color,mark in [('Tree',-.12,COLORS['Tree'],'o'),('Shrinkage',.12,COLORS['Shrinkage'],'s')]:
                d=paired[(paired.n==n)&(paired.setting==setting)&(paired.cost==cost)&(paired.comparison=='Robust - '+other)].iloc[0]
                ax.errorbar(d.mean_percent,i+offset,xerr=[[d.mean_percent-d.ci_low],[d.ci_high-d.mean_percent]],fmt=mark,color=color,capsize=2,markersize=4,label=other if i==0 else None)
        ax.axvline(0,color='black',lw=.7);ax.set_yticks(range(8),labels);ax.invert_yaxis();ax.set_title('No cost penalty' if cost==0 else '10 bps cost penalty');ax.set_xlabel('Robust: paired variance change (%)');ax.grid(axis='x',alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,-.07),title='Comparison method')
    fig.tight_layout();fig.savefig(FIG/'simulation_comparison.pdf');plt.close(fig)
    bench=pd.read_csv(RES/'revision_benchmark.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5));shapes=['star','balanced','recursive','path'];cols=['#0072B2','#D55E00','#009E73','#CC79A7']
    for ax,graph,title in zip(axes,['complete','knn4'],['Complete graph','Sparse graph (4-NN)']):
        for shape,col,mark in zip(shapes,cols,['o','s','^','D']):
            g=bench[(bench.graph==graph)&(bench['shape']==shape)].sort_values('L')
            ax.loglog(g.L,g.direct_seconds*1000,color=col,marker=mark,ms=3,lw=1)
            ax.loglog(g.L,g.cut_seconds*1000,color=col,marker=mark,ms=3,lw=1,ls='--')
        ax.set_title(title);ax.set_xlabel('L: permitted exchanges');ax.grid(True,which='both',alpha=.18)
    axes[0].set_ylabel('Median query time (ms)')
    handles=[Line2D([],[],color=c,marker=m,lw=0,label=s.title()) for s,c,m in zip(shapes,cols,['o','s','^','D'])]
    handles += [Line2D([],[],color='black',label='Path oracle'),Line2D([],[],color='black',ls='--',label='Cut reroot')]
    fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.12),fontsize=9)
    fig.tight_layout();fig.savefig(FIG/'benchmark.pdf');plt.close(fig)
    rows=[]
    for n,mode in [(256,'complete'),(256,'knn4'),(512,'knn4')]:
        for shape in shapes:
            r=bench[(bench.n==n)&(bench.graph==mode)&(bench['shape']==shape)].iloc[0]
            rows.append([n,shape,'Full' if mode=='complete' else '4-NN',f'{int(r.L):,}',f'{1000*r.direct_seconds:.2f}',f'{1000*r.cut_seconds:.2f}',f'{r.cut_over_direct:.2f}',f'{1000*r.path_preparation_seconds:.1f}',f'{1000*r.cut_additional_preparation_seconds:.1f}'])
    table('revision_benchmark.tex',['$n$','Shape','$E$','$L$','Path','Cut','Cut/path','Prep.','Group'],rows,'rllrrrrrr')
    hist=pd.concat(pd.read_csv(p) for p in sorted(RES.glob('history_*_summary.csv')));hist.to_csv(RES/'empirical_summary.csv',index=False)
    assert len(hist.groupby(['panel','config']))==18
    rows=[]
    for n in [10,30,49]:
        for method in METHODS:
            d=hist[(hist.panel==n)&(hist.config=='base')&(hist.method==method)].iloc[0]
            rows.append([n,method.replace('Shrinkage','LW'),f'{d.annualized_volatility*100:.2f}',f'{d.annualized_net_mean*100:.2f}',f'{d.net_cagr*100:.2f}',f'{d.sharpe:.3f}',f'{d.max_drawdown*100:.2f}',f'{d.annualized_two_way_turnover:.3f}'])
    table('revision_empirical.tex',['$n$','Method',r'Vol. (\%)',r'Mean (\%)',r'CAGR (\%)','Sharpe',r'MDD (\%)','Turnover'],rows,'rlrrrrrr')
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.4),sharey=True)
    for n,ax in zip([10,30,49],axes):
        daily=pd.read_csv(RES/f'history_{n}_base_daily.csv',parse_dates=['date'])
        for method in METHODS:
            g=daily[daily.method==method];ax.plot(g.date,np.cumprod(1+g.net),color=COLORS[method],ls=LINES[method],lw=1.15,label=method.replace('Shrinkage','LW'))
        from matplotlib.dates import YearLocator,DateFormatter
        ax.set_yscale('log');ax.set_title(f'{n} industries');ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(YearLocator(10));ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    axes[0].set_ylabel('Net wealth (initial = 1)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=7,bbox_to_anchor=(.5,-.06),fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'empirical_wealth.pdf');plt.close(fig)
    rows=[]
    labels={'base':'Base','free':'0 bps','cost25':'25 bps','kappa_half':r'$\kappa=26.25$','kappa_double':r'$\kappa=105$',
            'window126':'126-day window','window504':'504-day window','quarterly':'Quarterly'}
    for n in [10,30,49]:
        for config in labels:
            g=hist[(hist.panel==n)&(hist.config==config)].set_index('method')
            if not len(g):continue
            vols=g.annualized_volatility*100;r=g.loc['Robust']
            rows.append([n,labels[config],*[f'{vols[m]:.2f}' for m in ['Shrinkage','Tree','Robust']],f'{vols.Robust-vols.Tree:+.3f}',f'{vols.Robust-vols.Shrinkage:+.3f}',f'{r.sharpe:.3f}',f'{r.annualized_two_way_turnover:.3f}'])
    table('revision_sensitivity.tex',['$n$','Configuration','LW vol.','Tree vol.','Robust vol.',r'$R-T$',r'$R-LW$','R Sharpe','R turn.'],rows,'rlrrrrrrr')
    inf=pd.read_csv(RES/'revision_inference.csv');rows=[]
    for n in [10,30,49]:
        for other in ['Tree','Shrinkage']:
            g=inf[(inf.panel==n)&(inf.comparison=='Robust - '+other)&(inf.block==63)].set_index('statistic')
            v=g.loc['log_variance_ratio'];s=g.loc['sharpe_difference']
            rows.append([n,'R--'+other.replace('Shrinkage','LW'),f'{np.exp(v.estimate):.4f}',f'[{np.exp(v.ci_low):.4f}, {np.exp(v.ci_high):.4f}]',f'{v.p_holm:.3f}',f'{s.estimate:+.4f}',f'[{s.ci_low:+.4f}, {s.ci_high:+.4f}]',f'{s.p_holm:.3f}'])
    table('revision_inference.tex',['$n$','Pair','Var. ratio',r'95\% interval','$p_H$',r'$\Delta$ Sharpe',r'95\% interval','$p_H$'],rows,'rlrrrrrr')
    solvers=[]
    for p in sorted(RES.glob('*_solvers.csv')):
        if not p.name.startswith(('sim_','history_')):continue
        d=pd.read_csv(p);d['source']='simulation' if p.name.startswith('sim_') else 'history';solvers.append(d)
    sols=pd.concat(solvers);summary=[];statuses={}
    for source,g in sols.groupby('source'):
        counts={}
        for value in g.master_statuses:
            for k,v in ast.literal_eval(value).items():counts[k]=counts.get(k,0)+v
        statuses[source]=counts
        summary.append({'source':source,'allocations':len(g),'master_calls':int(g.master_calls.sum()),
            'non_success_allocations':int((~g.master_success).sum()),'non_success_calls':int(g.non_success_calls.sum()),
            'lp_non_success_calls':int(g.lp_non_success_calls.fillna(0).sum()),'lp_unresolved_calls':int(g.lp_unresolved_calls.fillna(0).sum()),
            'max_gap_scaled':float(g.gap_scaled.max()),'max_scenarios':int(g.scenarios.max())})
    pd.DataFrame(summary).to_csv(RES/'solver_summary.csv',index=False)
    (RES/'solver_status_counts.json').write_text(json.dumps(statuses,indent=2))
    env={'python':sys.version,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','clarabel','matplotlib']},
         'experiment_threads':1,'simulations':4000,'simulation_costs':[0,.001],
         'conic_tolerances':[1e-9,1e-11,1e-13],'global_scaled_gap_tolerance':2e-7}
    (RES/'environment.json').write_text(json.dumps(env,indent=2))
    facts={'solvers':summary,'statuses':statuses,'benchmark_dense_ratio_range':bench[bench.graph=='complete'].cut_over_direct.agg(['min','max']).to_dict(),
        'benchmark_sparse_512_ratio_range':bench[(bench.graph=='knn4')&(bench.n==512)].cut_over_direct.agg(['min','max']).to_dict()}
    (RES/'revision_facts.json').write_text(json.dumps(facts,indent=2));print(json.dumps(facts,indent=2))

if __name__=='__main__':run()
