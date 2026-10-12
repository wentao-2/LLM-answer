"""Reproducible full-width figures from audited completed experimental results.

Points use the frozen common complete parent set if it covers all six families.
Otherwise they use finite valid pairs and explicitly display that condition.
Intervals are ranges of family means, not confidence intervals.
"""
from pathlib import Path
from collections import defaultdict
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from common import PHI, QWEN, load, sha
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

# Reuse the paper's color semantics without depending on an external checkout.
COLORS = {'positive': '#258F96', 'independent': '#2265C3',
          'normative': '#68717D', 'ink': '#101B36', 'grid': '#E6EBF0'}


def setup():
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
        'axes.labelsize': 10, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'axes.linewidth': .65, 'axes.edgecolor': COLORS['normative'],
        'text.color': COLORS['ink'], 'axes.labelcolor': COLORS['ink'],
        'xtick.color': COLORS['ink'], 'ytick.color': COLORS['ink'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.unicode_minus': True})

MODELS = ((PHI,'Phi-4'),(QWEN,'Qwen3-8B'))
ARMS = (('native_self','Native history'),('replay_self','Self replay'))
KINDS = (('agree','Agreeing record',COLORS['positive'],'o'),
         ('conflict','Conflicting record',COLORS['independent'],'^'))
OUTPUT = ROOT/'figures'


def export(fig, name, provenance):
    OUTPUT.mkdir(exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_text = [t for t in fig.findobj(match=Text) if t.get_visible() and t.get_text()]
    for t in visible_text:
        assert t.get_fontsize() >= 9.5, t.get_text()
        box = t.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, t.get_text()
    provenance['rendering'] = dict(width_inches=fig.get_figwidth(), height_inches=fig.get_figheight(),
        minimum_font_points=min(t.get_fontsize() for t in visible_text), font='Times New Roman',
        text_within_canvas=True, source_sha256=sha(Path(__file__)))
    fig.savefig(OUTPUT/(name+'.pdf'),metadata={'Title':name,'Creator':'Matplotlib, audited I16 data'})
    fig.savefig(OUTPUT/(name+'.svg'))
    fig.savefig(OUTPUT/(name+'.png'),dpi=180)
    (OUTPUT/(name+'.data.json')).write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plt.close(fig)


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color=COLORS['grid'],lw=.6,zorder=0)
    ax.tick_params(length=3,pad=3)


def choose(row):
    common = row['common_all']
    if common['effective_weight']['n_families']==6:
        return common, 'common complete parents'
    return row, 'finite valid pairs'


def partial(data):
    fig=plt.figure(figsize=(6.3,5.6))
    axes=np.array([[fig.add_axes([.125,.60,.365,.275]),fig.add_axes([.60,.60,.365,.275])],
                   [fig.add_axes([.125,.17,.365,.275]),fig.add_axes([.60,.17,.365,.275])]])
    provenance=[]
    extent=[0,1]
    for k,(model,name) in enumerate(MODELS):
        for j,(arm,label) in enumerate(ARMS):
            ax=axes[k,j]
            style(ax)
            panel_counts=[]
            panel_modes=set()
            for branch,legend,color,marker in KINDS:
                rows=sorted([r for r in data['pair_summaries'] if r['module']=='partial_dependence'
                    and r['model']==model and r['history_arm']==arm and r['branch']==branch],key=lambda r:r['rho'])
                x,y,lo,hi,norm=[],[],[],[],[]
                counts=[]
                modes=set()
                for row in rows:
                    selected,mode=choose(row)
                    modes.add(mode)
                    weight=selected['effective_weight']
                    x.append(row['rho'])
                    y.append(weight['equal_family_mean'] if weight['equal_family_mean'] is not None else np.nan)
                    band=weight['family_range']
                    lo.append(band[0] if band else np.nan)
                    hi.append(band[1] if band else np.nan)
                    pair_rows=[p for p in data['pair_rows'] if p['module']=='partial_dependence' and p['model']==model
                        and p['history_arm']==arm and p['branch']==branch and p['rho']==row['rho']
                        and p['effective_weight'] is not None
                        and (mode!='common complete parents' or p['common_parent_all_cells'])]
                    family=defaultdict(list)
                    for p in pair_rows:family[p['family_id']].append(p['normative_weight'])
                    norm.append(sum(sum(v)/len(v) for v in family.values())/6 if len(family)==6 else np.nan)
                    counts.append(f'{weight["n"]}/{row["n_planned"]}')
                    provenance.append(dict(model=model,history_arm=arm,branch=branch,rho=row['rho'],
                        selection=mode,n=weight['n'],planned=row['n_planned'],estimate=weight['equal_family_mean'],
                        family_range=band,normative=norm[-1] if math.isfinite(norm[-1]) else None))
                ax.fill_between(x,lo,hi,color=color,alpha=.11,lw=0)
                ax.plot(x,norm,'--',color=color,lw=1.2,alpha=.7,zorder=2)
                ax.plot(x,y,marker=marker,color=color,lw=1.5,ms=4.8,zorder=3)
                extent.extend(v for v in lo+hi if math.isfinite(v))
                panel_modes.update(modes)
                panel_counts.extend(int(v.split('/')[0]) for v in counts)
            ax.set_xticks([0,.25,.5,.75,1],['0','0.25','0.5','0.75','1'])
            ax.set_xlim(-.025,1.025)
            desc='Common parents' if panel_modes=={'common complete parents'} else 'Finite pairs'
            count=str(min(panel_counts)) if min(panel_counts)==max(panel_counts) else f'{min(panel_counts)} to {max(panel_counts)}'
            box=ax.get_position()
            caption_y=.527 if k==0 else .060
            fig.text(box.x0+box.width/2,caption_y,f'({"abcd"[2*k+j]}) {name}, {label.lower()}',ha='center',va='top',fontsize=10)
            fig.text(box.x0+box.width/2,caption_y-.029,f'{desc}: {count} of 24',ha='center',va='top',fontsize=9.5)
            if j==0:ax.set_ylabel('Effective independent\nevidence weight',fontsize=10)
            else:ax.tick_params(labelleft=False)
            if k==1:ax.set_xlabel('Copying branch probability, ρ',fontsize=10)
    margin=.06*(max(extent)-min(extent) or 1)
    for ax in axes.flat:ax.set_ylim(min(extent)-margin,max(extent)+margin)
    handles=[Line2D([],[],color=c,marker=m,lw=1.5,label=t,ms=4.8) for _,t,c,m in KINDS]
    handles += [Line2D([],[],color=COLORS['normative'],ls='--',lw=1.2,label='Bayes reference')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.987),ncol=3,frameon=False,fontsize=9.5,columnspacing=1.4)
    export(fig,'partial_dependence',dict(analysis_sha256=sha(ROOT/'analysis/RESULTS.json'),
        interval='range of six family means, not a confidence interval',points=provenance))


def intervention(data):
    fig,axes=plt.subplots(1,2,figsize=(6.3,3.5),sharex=True,sharey=True)
    fig.subplots_adjust(left=.215,right=.98,top=.76,bottom=.29,wspace=.18)
    conditions=(('original','Original'),('general_bayes','General Bayes'),('source_structure','Source structure'))
    branches=(('copy','Copy',COLORS['positive'],'o',.18),
              ('independent_agree','Independent agree',COLORS['independent'],'s',0),
              ('independent_conflict','Independent conflict',COLORS['independent'],'^',-.18))
    provenance=[]
    extent=[0,1]
    for j,(arm,title) in enumerate(ARMS):
        ax=axes[j]
        style(ax)
        ax.grid(False)
        ax.grid(axis='x',color=COLORS['grid'],lw=.6,zorder=0)
        for z in (0,1):ax.axvline(z,color=COLORS['normative'],ls=':',lw=.8,zorder=1)
        for k,(condition,name) in enumerate(conditions):
            for branch,legend,color,marker,offset in branches:
                row=next(r for r in data['pair_summaries'] if r['module']=='source_intervention'
                    and r['history_arm']==arm and r['condition']==condition and r['branch']==branch)
                selected,mode=choose(row)
                weight=selected['effective_weight']
                norm=0 if branch=='copy' else 1
                pos=2-k+offset
                ax.plot(norm,pos,marker='D',mfc='white',mec=COLORS['normative'],ms=4.5,mew=.8,zorder=2)
                if weight['equal_family_mean'] is not None:
                    low,high=weight['family_range']
                    ax.plot([low,high],[pos,pos],color=color,lw=1.8,zorder=3)
                    ax.plot(weight['equal_family_mean'],pos,marker=marker,color=color,ms=5,zorder=4)
                    extent += [low,high]
                provenance.append(dict(history_arm=arm,condition=condition,branch=branch,selection=mode,
                    n=weight['n'],planned=row['n_planned'],estimate=weight['equal_family_mean'],
                    family_range=weight['family_range'],normative=norm))
        ax.set_yticks([2,1,0],[v for _,v in conditions])
        ax.set_ylim(-.43,2.43)
        ax.set_xlabel('Effective independent\nevidence weight',fontsize=10)
        box=ax.get_position()
        fig.text(box.x0+box.width/2,.095,f'({"ab"[j]}) {title}',ha='center',va='top',fontsize=10)
    margin=.07*(max(extent)-min(extent) or 1)
    for ax in axes:
        low,high=min(extent)-margin,max(extent)+margin
        ax.set_xlim(low,high)
        ax.set_xticks([value for value in (0,.5,1) if low<=value<=high])
    handles=[Line2D([],[],color=c,marker=m,lw=0,label=t,ms=5) for _,t,c,m,_ in branches]
    handles += [Line2D([],[],marker='D',mfc='white',mec=COLORS['normative'],lw=0,label='Bayes reference',ms=4.5)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.56,.987),ncol=2,frameon=False,fontsize=9.5,columnspacing=1.5)
    export(fig,'source_intervention',dict(analysis_sha256=sha(ROOT/'analysis/RESULTS.json'),
        interval='range of six family means, not a confidence interval',points=provenance))


def main():
    setup()
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    audit=load(ROOT/'audit/AUDIT.json')
    assert audit['state']=='passed' and audit['analysis_sha256']==sha(ROOT/'analysis/RESULTS.json')
    assert load(ROOT/'audit/AGGREGATES.json')['state']=='passed'
    data=load(ROOT/'analysis/RESULTS.json')
    partial(data)
    intervention(data)
    print(json.dumps(dict(figures=['partial_dependence','source_intervention'],source_sha256=sha(Path(__file__)),
                          analysis_sha256=sha(ROOT/'analysis/RESULTS.json'))))


if __name__=='__main__':main()
