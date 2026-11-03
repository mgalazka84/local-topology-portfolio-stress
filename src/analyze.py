"""Generate tables, paired uncertainty summaries, and publication figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
for directory in ['manuscript','figures','results']:
    (ROOT/directory).mkdir(exist_ok=True)
COLORS={'Equal':'#7e8794','HRP':'#bd8c2c','Shrinkage':'#3267a8','Tree':'#c75d56','Robust':'#168574'}
ORDER=['Equal','HRP','Shrinkage','Tree','Robust']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':140,
                     'savefig.bbox':'tight','pdf.fonttype':42})

def table(path,headers,rows,alignment=None):
    if alignment is None:alignment='l'+'r'*(len(headers)-1)
    text='\\begin{tabular}{'+alignment+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(str(x) for x in row)+' \\\\' for row in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n'
    (ROOT/'manuscript'/path).write_text(text)

def empirical():
    all_summary=[];inference=[];solver_summary=[]
    rng=np.random.default_rng(112030)
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.7))
    for n,ax in zip([10,30],axs):
        s=pd.read_csv(ROOT/'results'/f'empirical_{n}_summary.csv');all_summary.append(s)
        s=s[s.cost==.001].set_index('method').loc[ORDER]
        rows=[]
        for name,r in s.iterrows():
            rows.append([name,f'{100*r.annualized_volatility:.2f}',f'{100*r.annualized_net_mean:.2f}',
                         f'{100*r.net_cagr:.2f}',f'{100*r.max_drawdown:.2f}',f'{r.annualized_two_way_turnover:.3f}'])
        table(f'empirical_{n}.tex',['Method','Vol. (\\%)','Mean (\\%)','CAGR (\\%)','MDD (\\%)','Turnover'],rows)
        d=pd.read_csv(ROOT/'results'/f'empirical_{n}_daily.csv',parse_dates=['date'])
        d=d[d.cost==.001]
        panel=d.pivot(index='date',columns='method',values='net')
        for name in ORDER:
            wealth=(1+panel[name]).cumprod()
            ax.plot(wealth.index,wealth,color=COLORS[name],label=name,lw=1.5 if name=='Robust' else 1.)
        ax.set(title=f'{n} industry portfolios',yscale='log',ylabel='Hypothetical wealth (start = 1)')
        ax.grid(alpha=.17)
        T=len(panel);block=63;nb=(T+block-1)//block
        stats={name:[] for name in ['Tree','Shrinkage']}
        for _ in range(999):
            indices=((rng.integers(T,size=(nb,1))+np.arange(block))%T).ravel()[:T]
            a=panel.iloc[indices]
            for name in stats:
                stats[name].append([float((a.Robust.mean()-a[name].mean())*252),
                                   float((a.Robust.std(ddof=1)-a[name].std(ddof=1))*np.sqrt(252))])
        for name,z in stats.items():
            z=np.array(z);lo,hi=np.quantile(z,[.025,.975],axis=0)
            estimates=[float((panel.Robust.mean()-panel[name].mean())*252),
                       float((panel.Robust.std(ddof=1)-panel[name].std(ddof=1))*np.sqrt(252))]
            for i,metric in enumerate(['annualized_net_mean','annualized_volatility']):
                inference.append({'panel':n,'comparison':'Robust - '+name,'metric':metric,
                    'estimate':estimates[i],'lower':float(lo[i]),'upper':float(hi[i]),'block_length':block,'replicates':999})
        sol=pd.read_csv(ROOT/'results'/f'empirical_{n}_solvers.csv')
        for name,g in sol.groupby('method'):
            solver_summary.append({'panel':n,'method':name,'solves':len(g),'median_seconds':g.seconds.median(),
                'max_seconds':g.seconds.max(),'max_gap_scaled':g.gap_scaled.max(),
                'median_scenarios':g.scenarios.median(),'max_scenarios':g.scenarios.max(),
                'max_gap':g.gap.max(),'success_fraction':g.master_success.mean(),
                'median_candidates':g.candidates.median()})
    axs[0].legend(ncol=3,loc='upper left',fontsize=8,frameon=False)
    fig.tight_layout();fig.savefig(ROOT/'figures/empirical_wealth.pdf');plt.close(fig)
    full=pd.concat(all_summary);full.to_csv(ROOT/'results/empirical_summary.csv',index=False)
    pd.DataFrame(inference).to_csv(ROOT/'results/empirical_bootstrap.csv',index=False)
    pd.DataFrame(solver_summary).to_csv(ROOT/'results/solver_summary.csv',index=False)
    rows=[]
    for n in [10,30]:
        for c in [0,.001,.0025]:
            p=full[(full.panel==n)&(full.cost==c)].set_index('method')
            rows.append([n,int(c*10000),*[f'{p.loc[name,"annualized_volatility"]*100:.2f}' for name in ['Shrinkage','Tree','Robust']],
                         *[f'{p.loc[name,"annualized_two_way_turnover"]:.3f}' for name in ['Shrinkage','Tree','Robust']]])
    table('cost_sensitivity.tex',['$n$','bps','LW vol.','Tree vol.','Rob. vol.','LW turn.','Tree turn.','Rob. turn.'],rows)
    rows=[]
    for x in inference:
        rows.append([x['panel'],x['comparison'].replace(' - ',' $-$ '),
            'Mean' if x['metric']=='annualized_net_mean' else 'Volatility',
            f'{100*x["estimate"]:.3f}',f'[{100*x["lower"]:.3f}, {100*x["upper"]:.3f}]'])
    table('bootstrap.tex',['$n$','Comparison','Metric','Difference','95\\% interval'],rows,'rllrr')

def simulation():
    d=pd.read_csv(ROOT/'results/simulation.csv');rows=[];paired=[]
    settings=['Gaussian tree','Student t tree','Structural break','Factor misspecification']
    fig,axs=plt.subplots(1,2,figsize=(10.0,3.6),sharey=True)
    for n,ax in zip([12,30],axs):
        for j,setting in enumerate(settings):
            p=d[(d.n==n)&(d.setting==setting)].pivot(index='rep',columns='method',values='relative_population_variance')
            delta=100*(p.Robust/p.Tree-1)
            half=float(student_t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)))
            x=float(delta.mean());paired.append({'n':n,'setting':setting,'mean_percent_difference':x,'lower':x-half,'upper':x+half,'replications':len(delta)})
            ax.errorbar(x,j,xerr=half,fmt='o',color=COLORS['Robust'],capsize=4)
            rows.append([n,setting,*[f'{p[name].mean():.3f}' for name in ORDER],f'{x:.2f}'])
        ax.axvline(0,color='#555555',lw=.8,ls='--');ax.set_yticks(range(4),settings)
        ax.set(title=f'{n} assets',xlabel='Population variance: Robust / Tree - 1 (%)')
        ax.grid(axis='x',alpha=.2)
    axs[0].invert_yaxis();fig.tight_layout();fig.savefig(ROOT/'figures/simulation_comparison.pdf');plt.close(fig)
    pd.DataFrame(paired).to_csv(ROOT/'results/simulation_paired.csv',index=False)
    table('simulation.tex',['$n$','DGP','Equal','HRP','LW','Tree','Robust','$\\Delta$ (\\%)'],rows,'rlrrrrrr')

def benchmark():
    d=pd.read_csv(ROOT/'results/benchmark.csv');rows=[]
    for _,r in d[d.n.isin([32,96])].iterrows():
        rows.append([int(r.n),r['shape'],int(r.L),f'{1000*r.preprocessing_seconds:.1f}',
                     f'{1000*r.oracle_seconds:.2f}',f'{1000*r.per_candidate_messages_seconds:.1f}',f'{r.speedup:.1f}'])
    table('benchmark.tex',['$n$','Tree','$L$','Prep. ms','Query ms','Repeat ms','Ratio'],rows,'rlrrrrr')
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.5))
    for shape,style in [('star','o-'),('balanced','s-'),('recursive','^-'),('path','d-')]:
        p=d[d['shape']==shape]
        axs[0].loglog(p.n,p.oracle_seconds*1000,style,label=shape)
        axs[1].loglog(p.n,p.speedup,style,label=shape)
    axs[0].set(xlabel='Assets',ylabel='Oracle query (ms)')
    axs[1].set(xlabel='Assets',ylabel='Repeated-message time / oracle time')
    for ax in axs:ax.grid(alpha=.2,which='both')
    axs[0].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(ROOT/'figures/benchmark.pdf');plt.close(fig)

def diagram():
    # Exact explanatory example, independently evaluated as dense quadratic forms.
    w=np.array([.45,.05,.50])
    cov0=np.array([[1,.8,.56],[.8,1,.7],[.56,.7,1]])
    cov1=np.array([[1,.8,.6],[.8,1,.48],[.6,.48,1]])
    q0=float(w@cov0@w);q1=float(w@cov1@w)
    assert np.allclose([q0,q1],[.778,.785],rtol=0,atol=1e-14)
    pos={1:(-.9,0),2:(0,1.15),3:(.9,0)}
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.5))
    panels=[('Reference tree',q0,{(1,2):(.8,True),(2,3):(.7,True),(1,3):(.56,False)}),
            ('Replace link 2–3 by 1–3',q1,{(1,2):(.8,True),(2,3):(.48,False),(1,3):(.6,True)})]
    labels={(1,2):(-.63,.63),(2,3):(.63,.63),(1,3):(0,-.18)}
    for ax,(title,q,edges) in zip(axs,panels):
        for (i,j),(corr,selected) in edges.items():
            color='#168574' if selected else '#9c5d33'
            ax.plot([pos[i][0],pos[j][0]],[pos[i][1],pos[j][1]],color=color,
                    lw=2.2 if selected else 1.6,ls='-' if selected else '--',zorder=1)
            x,y=labels[(i,j)]
            label=f'{corr:.2f}' if selected else f'implied\n{corr:.2f}'
            ax.text(x,y,label,ha='center',va='center',fontsize=10,color=color,
                    bbox={'facecolor':'white','edgecolor':'none','pad':2},zorder=5)
        for i,(x,y) in pos.items():
            ax.scatter(x,y,s=500,color='#e7eef4',edgecolor='#31566b',zorder=3)
            ax.text(x,y,str(i),ha='center',va='center',zorder=4,fontsize=11)
        ax.set_title(title,fontsize=11,pad=7)
        ax.text(0,-.53,f'Portfolio variance = {q:.3f}',ha='center',fontsize=11,fontweight='bold',color='#173b57')
        ax.set(xlim=(-1.36,1.36),ylim=(-.72,1.47));ax.axis('off')
    fig.text(.5,.025,f'Same portfolio weights:  variance increases by {q1-q0:.3f} ({100*(q1/q0-1):.2f}%)',
             ha='center',fontsize=10,color='#173b57')
    fig.tight_layout(rect=(0,.09,1,1));fig.savefig(ROOT/'figures/exchange.pdf');plt.close(fig)

if __name__=='__main__':
    empirical();simulation();benchmark();diagram()
    print('Analysis, tables, and figures written.')
