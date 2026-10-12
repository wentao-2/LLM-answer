"""Reproducible diagnosis and history plots from the audited supplement exports."""
import csv,json
from pathlib import Path
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_extension import setup,export,COLORS,source

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/review_supplement'
ARM_NAMES={'native_self':'Native history','replay_self':'Self replay','replay_other':'Other replay','replay_neutral':'Neutral replay'}
ARM_COLORS={'native_self':COLORS['ink'],'replay_self':COLORS['self'],'replay_other':COLORS['other'],'replay_neutral':COLORS['normative']}
ARMS=tuple(ARM_NAMES)
WORDINGS={'original':('Original',COLORS['normative']),'authorship':('Author label','#7A62A3'),'dependence':('Evidence rule',COLORS['positive']),'author_independence':('Label independence','#AE5267')}

def clean(ax,zero=True):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',color=COLORS['grid'],lw=.55,zorder=0)
    ax.tick_params(axis='y',length=0)
    if zero:ax.axvline(0,color=COLORS['normative'],lw=.65,ls=':')

def save(fig,stem,selection):
    export(fig,dict(kind='review_supplement',sources={'summaries':source(DATA/'summaries.json'),'static_csv':source(DATA/'phi_static.csv'),'legacy_csv':source(ROOT/'data/full_results/attribution_decomposition.csv')},selection=selection),SimpleNamespace(output_dir=ROOT/'figures',stem=stem,preview_dir=ROOT/'build/qa'))
    plt.close(fig)

def diagnosis(docs):
    fig=plt.figure(figsize=(6.3,2.94))
    a=fig.add_axes([.15,.35,.31,.40]);b=fig.add_axes([.66,.35,.31,.40])
    rows=list(csv.DictReader((ROOT/'data/full_results/attribution_decomposition.csv').open(encoding='utf-8-sig')))
    phi=list(csv.DictReader((DATA/'phi_static.csv').open(encoding='utf-8')))
    keys=('self_directional_error','other_directional_error','attribution_gap')
    colors=[COLORS[k] for k in ('self','other','positive')]
    counts=[]
    for j,(model,rep) in enumerate([('Qwen3','likelihood'),('Qwen3','frequency'),('Phi-4','likelihood'),('Phi-4','frequency')]):
        y=3-j
        if model=='Qwen3':
            r=next(x for x in rows if all(x[k]==v for k,v in dict(model='qwen3-8b-thinking',protocol='independent',visibility='signal_visible',relation='agree',reminder='legacy',output_instruction='restrictive',template_id=rep+'_v3').items()))
            vals=[(float(r['metrics.'+k+'.estimate']),float(r['missing_probability_bounds.'+k+'.lower']),float(r['missing_probability_bounds.'+k+'.upper'])) for k in keys]
            counts.append(r['complete_pairs']+'/24')
        else:
            r=next(x for x in phi if x['reminder']=='legacy' and x['wording']=='restrictive' and x['representation']==rep)
            vals=[(float(r[k]),float(r[k]),float(r[k])) for k in ('self_error','other_error','gap')]
            counts.append(r['complete']+'/24')
        for off,(p,lo,hi),c in zip((.18,0,-.18),vals,colors):
            a.plot([lo,hi],[y+off,y+off],color=c,lw=1.3);a.plot(p,y+off,'o',color=c,ms=3.8)
        a.text(.99,y,counts[-1],transform=a.get_yaxis_transform(),fontsize=9.5,va='center',ha='right')
    a.set(ylim=(-.5,3.5),yticks=[3,2,1,0],yticklabels=['Qwen3\nLikelihood','Qwen3\nFrequency','Phi-4\nLikelihood','Phi-4\nFrequency'],xlim=(-.21,.08),xticks=[-.2,-.1,0],xlabel='Directional error or gap')
    clean(a)
    selected=[x for x in docs['four_wordings']['strata'] if 'template_id' not in x['stratum']]
    for j,row in enumerate(selected):
        y=3-j
        for off,(v,(_,color)) in zip((.24,.08,-.08,-.24),WORDINGS.items()):
            x=row['versions'][v]['gap'];lo=x['bounds']['lower'];hi=x['bounds']['upper']
            b.plot([lo,hi],[y+off,y+off],color=color,lw=1.4,alpha=.8)
            b.plot(x['point']['estimate'],y+off,'o',color=color,ms=3.5)
        b.text(.985,y+.30,f"{row['complete_twelve_sets']}/48",transform=b.get_yaxis_transform(),ha='right',va='bottom',fontsize=9.5)
    b.set(ylim=(-.5,3.5),yticks=[3,2,1,0],yticklabels=['Legacy\ncompatible','Legacy\nrestrictive','Scoped\ncompatible','Scoped\nrestrictive'],xlim=(-.058,.13),xticks=[-.05,0,.05,.10],xlabel='Attribution gap')
    clean(b)
    a.legend([Line2D([],[],marker='o',color=c,lw=0,ms=4) for c in colors],['Self error','Other error','Gap'],loc='lower left',bbox_to_anchor=(-.35,1.06),ncol=2,frameon=False,fontsize=9.5,handletextpad=.2,columnspacing=.7)
    b.legend([Line2D([],[],marker='o',color=c,lw=0,ms=4) for _,c in WORDINGS.values()],[s for s,_ in WORDINGS.values()],loc='lower left',bbox_to_anchor=(-.4,1.03),ncol=2,frameon=False,fontsize=9.5,handletextpad=.2,columnspacing=.7)
    fig.text(.5,.14,'Dots: complete units. Lines: full design bounds. Counts: complete / planned.',ha='center',fontsize=9.5)
    fig.text(.3,.05,'(a) Legacy / restrictive static diagnosis',ha='center',fontweight='bold',fontsize=9.5)
    fig.text(.80,.05,'(b) Four wordings on common sets',ha='center',fontweight='bold',fontsize=9.5)
    save(fig,'review_diagnosis','Panel a: independent agreeing readings in both representations, legacy restrictive condition. Qwen retained SDPA and standard Phi-4. Panel b: all four uniform-vLLM wording strata, common twelve reports; counts differ from full-design bound targets.')

def cohort(docs,model,name): return next(c for c in docs[model]['cohorts'] if c['cohort']==name)

def history(docs):
    fig=plt.figure(figsize=(6.3,4.2))
    axes=[fig.add_axes([.18,.67,.29,.20]),fig.add_axes([.66,.67,.30,.20]),fig.add_axes([.18,.19,.29,.25]),fig.add_axes([.66,.19,.30,.25])]
    for ax,kind in zip(axes[:2],('copy','independent')):
        for i,arm in enumerate(ARMS):
            y=3-i
            for model,off,marker in [('qwen_history',.10,'o'),('phi_history',-.10,'D')]:
                x=next(z for z in cohort(docs,model,'all')['individual_update_errors'] if z['history_arm']==arm and z['evidence_kind']==kind)
                ax.plot(x['absolute']['estimate'],y+off,marker,color=ARM_COLORS[arm],ms=4)
        ax.set(xscale='log',xlim=(3e-5,.2),xticks=[.0001,.001,.01,.1],ylim=(-.5,3.5),yticks=[3,2,1,0],yticklabels=list(ARM_NAMES.values()),xlabel='Mean absolute update residual')
        clean(ax,False)
    axes[0].legend([Line2D([],[],marker=m,color=COLORS['ink'],lw=0,ms=4) for m in ('o','D')],['Qwen3','Phi-4'],loc='lower left',bbox_to_anchor=(-.4,1.11),ncol=2,frameon=False,fontsize=9.5)
    a=axes[2];names=('all','large_error','wrong_choice','overconfident','visible_prefix')
    labels=('All valid','Large error','Incorrect choice','Overconfident','Explanation')
    vals=[]
    for y,(n,label) in zip(range(4,-1,-1),zip(names,labels)):
        c=cohort(docs,'qwen_history',n)
        r=next(x for x in c['analysis']['presentation'] if x['stratum']=={'evidence_kind':'none'})
        s=r['contrasts']['history_initial_choice_direction'];bd=s['bounds'];p=s['point']['estimate']
        a.plot([bd['lower'],bd['upper']],[y,y],color=COLORS['ink'],lw=1.4)
        if p is not None:a.plot(p,y,'o',color=COLORS['ink'],ms=4)
        vals.extend([bd['lower'],bd['upper']]);a.text(.99,y,f"{s['complete_valid_sets']}/{s['candidate_sets']}",transform=a.get_yaxis_transform(),ha='right',va='bottom',fontsize=9.5)
    a.set(ylim=(-.5,4.5),yticks=[4,3,2,1,0],yticklabels=labels,xlim=(min(vals)-.03,max(vals)+.05),xlabel='Native minus self replay')
    clean(a)
    a=axes[3]
    for y,arm in zip(range(3,-1,-1),ARMS):
        r=next(x for x in cohort(docs,'qwen_history','large_error')['individual_update_errors'] if x['history_arm']==arm and x['evidence_kind']=='independent')
        a.plot(r['signed']['estimate'],y+.08,'o',mfc='white',mec=ARM_COLORS[arm],ms=4)
        a.plot(r['absolute']['estimate'],y-.08,'D',color=ARM_COLORS[arm],ms=4)
        a.text(.99,y,f"{r['absolute']['n_observations']}/{r['planned_pairs']}",transform=a.get_yaxis_transform(),ha='right',va='bottom',fontsize=9.5)
    a.set(ylim=(-.5,3.5),yticks=[3,2,1,0],yticklabels=list(ARM_NAMES.values()),xlim=(-.007,.018),xticks=[-.005,0,.005,.01,.015],xlabel='Update residual')
    clean(a)
    a.legend([Line2D([],[],marker='o',mfc='white',color=COLORS['ink'],lw=0,ms=4),Line2D([],[],marker='D',color=COLORS['ink'],lw=0,ms=4)],['Signed mean','Mean absolute'],loc='lower left',bbox_to_anchor=(-.25,1.08),ncol=2,frameon=False,fontsize=9.5,handletextpad=.2,columnspacing=.4)
    for x,y,title in [(.33,.535,'(a) Copying: zero warranted increment'),(.81,.535,'(b) Independent evidence'),(.33,.055,'(c) Qwen3: no new evidence'),(.81,.055,'(d) Qwen3: large parent errors')]:
        fig.text(x,y,title,ha='center',fontsize=9.5,fontweight='bold')
    save(fig,'review_history','All four arms and both qualified checkpoints for itemwise absolute errors in panels a/b. Panel c: no-new-evidence total presentation contrasts and full constructed-cohort bounds. Panel d: large-parent-error cohort, independent update signed and absolute means on identical valid pairs. Counts label complete/planned pairs; panel a/b pair denominators are exported in the CSV and main table.')

if __name__=='__main__':
    setup();docs=json.loads((DATA/'summaries.json').read_text());diagnosis(docs);history(docs)
