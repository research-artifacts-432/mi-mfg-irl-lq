"""Rebuild frozen unseen summaries and scientific figures from retained results only."""
from pathlib import Path
import csv,json,os,tempfile
import numpy as np
from .unseen import aggregate,preserve_csv,stopping_decision
from .runs import configurations,family_name
from .io import write_csv

BAND_CAPTION='Error bars represent ±2.58 sample SD across independent dataset-level estimates; they are descriptive dispersion bands, not confidence intervals for the mean.'
V1_BETAS=[0,.001,.01,.03,.1,.3,1,10,100]


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def v1_beta_recovery_data(output):
    """Use only the canonical fixed-40 observations and saved summary statistics."""
    output=Path(output)
    observations=read_csv(output/'estimates.csv')
    summaries=read_csv(output/'summary.csv')
    groups=[]
    for beta in V1_BETAS:
        summary=next(r for r in summaries if r['version']=='V1' and float(r['beta'])==beta)
        rows=sorted((r for r in observations if r['family']==summary['family']),key=lambda r:int(r['dataset_id']))
        if len(rows)!=40 or len({r['dataset_id'] for r in rows})!=40:
            raise ValueError('Figure 3 requires 40 distinct fixed-sample estimates per V1 family.')
        groups.append(dict(beta=beta,eta=float(summary['eta']),
            values=np.array([float(r['lambda_hat']) for r in rows]),
            mean=float(summary['mean_lambda']),sd=float(summary['sample_sd'])))
    return groups


def reconstruct_tables(output):
    output=Path(output);spec=json.loads((output/'config.json').read_text());account=json.loads((output/'run_accounting.json').read_text())
    if not account['completed']:raise ValueError('Scientific execution is incomplete.')
    rows=read_csv(output/'estimates.csv');m=account['M_stop']
    if len(rows)!=10*m or {int(r['dataset_id']) for r in rows}!=set(spec['dataset_ids'][:m]):raise ValueError('Incomplete stopped dataset prefix.')
    cumulative,trace,decision=aggregate(rows,spec)
    if not decision['stop'] or decision['reason']!=account['stopping_reason']:raise ValueError('Stopping reconstruction differs.')
    eligible=sorted({r['M'] for r in trace if r['M']>=spec['stopping']['M_min']})
    earliest=next((n for n in eligible if stopping_decision(trace,n,spec['stopping'])['stop']),None)
    if earliest!=m:raise ValueError('Stopped prefix is not the earliest prescribed stopping point.')
    fixed=[r for r in cumulative if r['M']==spec['stopping']['M_fixed']];stopped=[r for r in cumulative if r['M']==m]
    preserve_csv(output/'summary.csv',fixed)
    write_csv(output/'cumulative.csv',cumulative);write_csv(output/'stopping.csv',trace)
    return spec,account,rows,fixed,cumulative,trace,stopped


def scientific_figures(output,inputs=None):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'mi_mfg_irl_lq_matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42})
    output=Path(output);spec,account,rows,fixed,cumulative,trace,stopped=inputs or reconstruct_tables(output)
    folder=output/'figures';folder.mkdir(exist_ok=True)
    groups=configurations(spec);mstop=account['M_stop'];mfixed=spec['stopping']['M_fixed'];paths=[]
    def label(group):return ('V2' if group['version']=='V2' else f"V1 β={group['beta']:g}")+f"; η={group['eta']:g}"
    def save(fig,name,caption):
        fig.text(.5,.012,caption,ha='center',va='bottom',fontsize=9,wrap=True)
        for extension in ['png','pdf']:
            path=folder/f'{name}.{extension}';fig.savefig(path,dpi=200,bbox_inches='tight');paths.append(path.relative_to(output).as_posix())
        plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(13,16),sharex=True,sharey=True)
    for axis,group in zip(axes.flat,groups):
        family=family_name(group['version'],group['beta']);rr=[r for r in cumulative if r['family']==family]
        x=[r['M'] for r in rr];y=[r['mean_altruism'] for r in rr];width=[r['half_width'] for r in rr]
        axis.errorbar(x,y,yerr=width,fmt='o-',ms=3,lw=1,capsize=2,color='#1764a0')
        axis.axhline(1-spec['reporting']['target_lambda'],color='black',ls='--',lw=1,label='True altruism = 0.1')
        axis.axvline(mfixed,color='#777777',ls=':',lw=1)
        axis.axvline(mstop,color='#b32424',ls='-.',lw=1)
        axis.set_title(label(group));axis.grid(alpha=.2);axis.set_xlabel('Independent expert datasets M');axis.set_ylabel('1 − cumulative mean λ')
    handles=[plt.Line2D([],[],color='black',ls='--',label='True altruism = 0.1'),
             plt.Line2D([],[],color='#777777',ls=':',label=f'Fixed sample M={mfixed}'),
             plt.Line2D([],[],color='#b32424',ls='-.',label=f'Stop M={mstop}')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.972),ncol=3)
    fig.suptitle('Cumulative estimation of altruism level',fontsize=16,y=.997)
    fig.tight_layout(rect=(0,.055,1,.96))
    save(fig,'altruism','Error bars represent ±2.58 sample SD across independent dataset-level estimates;\nthey are descriptive dispersion bands, not confidence intervals for the mean.')
    fig,axes=plt.subplots(5,2,figsize=(13,16),sharex=True,sharey=True)
    prefix=set(spec['dataset_ids'][:mfixed]);target=spec['reporting']['target_lambda'];threshold=spec['reporting']['recovery_threshold']
    for axis,group in zip(axes.flat,groups):
        rr=[r for r in rows if r['family']==family_name(group['version'],group['beta']) and int(r['dataset_id']) in prefix]
        axis.axhspan(target-threshold,target+threshold,color='#91cf9a',alpha=.4,label='Recovery band [0.89, 0.91]')
        axis.axhline(target,color='black',ls='--',lw=1,label='True λ = 0.9')
        axis.scatter([int(r['dataset_id']) for r in rr],[float(r['lambda_hat']) for r in rr],s=20,color='#1764a0',alpha=.85,zorder=3)
        axis.set_title(label(group));axis.set_xlabel('Unseen dataset ID');axis.set_ylabel('Recovered λ at K1000');axis.grid(alpha=.2)
        axis.ticklabel_format(axis='x',style='plain',useOffset=False)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.971),ncol=2)
    fig.suptitle(f'Fixed-{mfixed} dataset-level recovery: every last iterate',fontsize=16,y=.997)
    fig.tight_layout(rect=(0,.04,1,.96));save(fig,'recovery','One point is one independent expert dataset; configurations on the same dataset are paired.\nAll recovered values are displayed on shared parameter axes; no cumulative averages replace them.')
    beta_groups=v1_beta_recovery_data(output)
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for axis,selected,title in zip(axes,[beta_groups,beta_groups[2:]],
                                   ['All nine V1 families — full observed range','Magnified: β ≥ 0.01 — same observations']):
        for index,group in enumerate(selected):
            # Deterministic horizontal offsets reveal overlapping observations without altering λ.
            axis.scatter(index+np.linspace(-.18,.18,40),group['values'],s=22,alpha=.35,
                color='#1764a0',edgecolors='none',label='Individual dataset estimates' if index==0 else None)
        axis.errorbar(np.arange(len(selected)),[g['mean'] for g in selected],
            yerr=[g['sd'] for g in selected],fmt='D',ms=5,capsize=4,lw=1.6,
            color='#b34f00',label='Mean ± sample SD (ddof=1)',zorder=4)
        axis.axhline(.9,color='black',ls='--',lw=1.2,label='True λ = 0.9')
        axis.set_xticks(np.arange(len(selected)),[f"{g['beta']:g}\nη={g['eta']:g}" for g in selected])
        axis.set_xlabel('Consistency weight β (categorical positions; frozen η shown)')
        axis.set_ylabel('Recovered λ at K1000');axis.set_title(title);axis.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.suptitle('V1 parameter recovery across consistency weights',fontsize=16,y=.99)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,fontsize=10)
    fig.tight_layout(rect=(0,.12,1,.85))
    save(fig,'beta_recovery','Fixed-40 sampled-BR estimates only; error bars are ±1 sample SD across datasets, not SE or confidence intervals.\nThe right panel magnifies the same seven families; horizontal offsets only separate markers. No V2 or Plugin comparator is included.')
    fig,axes=plt.subplots(2,1,figsize=(13,10),sharex=True)
    colors=plt.get_cmap('tab10').colors
    for group,color in zip(groups,colors):
        rr=[r for r in cumulative if r['family']==family_name(group['version'],group['beta'])];x=[r['M'] for r in rr]
        axes[0].plot(x,[r['MAE'] for r in rr],'o-',ms=3,color=color,label=label(group))
        axes[1].plot(x,[r['success_rate'] for r in rr],'o-',ms=3,color=color,label=label(group))
    axes[0].set_ylabel('Cumulative MAE');axes[1].set_ylabel('Cumulative recovery success rate');axes[1].set_ylim(-.025,1.025)
    for axis in axes:
        axis.axvline(mfixed,color='#777777',ls=':');axis.axvline(mstop,color='#b32424',ls='-.');axis.grid(alpha=.2)
    axes[1].set_xlabel('Independent expert datasets M');fig.suptitle('Cumulative recovery quality — descriptive evaluation',fontsize=15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.035),ncol=5,fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.96));save(fig,'cumulative_recovery','Success means |λ̂ − 0.9| ≤ 0.01. These recovery metrics are not stopping inputs.')
    fig,axis=plt.subplots(figsize=(11,6));worst={r['M']:r['Delta'] for r in trace if r['Delta'] is not None}
    axis.plot(list(worst),list(worst.values()),'o-',color='#1764a0',label='Worst-family Δ(M)')
    axis.axhline(spec['stopping']['relative_threshold'],color='black',ls='--',label='Stability threshold 0.05')
    axis.axvline(mfixed,color='#777777',ls=':',label=f'Earliest stop M={mfixed}')
    axis.axvline(mstop,color='#b32424',ls='-.',label=f'Actual stop M={mstop}')
    axis.set_xlabel('Independent expert datasets M');axis.set_ylabel('Maximum relative half-width change Δ(M)');axis.grid(alpha=.2)
    axis.set_title(f"Stopping diagnostics: {account['stopping_reason']}");axis.legend(loc='best')
    fig.tight_layout(rect=(0,.065,1,1));save(fig,'stopping','Stability uses two consecutive passing checks across all ten families.\nIt measures sample-dispersion stabilization, not recovery accuracy.')
    return paths
