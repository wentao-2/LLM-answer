"""Reproduce the two main followup figures from audited, unrounded exports."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'data/followup/evidence.json'
PHI = 'phi4-standard-r6'
COLORS = {'copy': '#258F96', 'independent': '#2265C3', 'ordinary': '#D99924',
          'explicit': '#2265C3', 'normative': '#68717D', 'ink': '#101B36',
          'grid': '#E6EBF0'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
        'axes.labelsize': 10, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'axes.linewidth': .65, 'axes.edgecolor': COLORS['normative'],
        'text.color': COLORS['ink'], 'axes.labelcolor': COLORS['ink'],
        'xtick.color': COLORS['ink'], 'ytick.color': COLORS['ink'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.unicode_minus': True})


def style(ax, horizontal=False):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x' if horizontal else 'y', color=COLORS['grid'], lw=.6, zorder=0)
    ax.tick_params(length=3, pad=3)


def export(fig, name, points):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text = [t for t in fig.findobj(match=Text) if t.get_visible() and t.get_text()]
    for item in text:
        assert item.get_fontsize() >= 9.5, item.get_text()
        box = item.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, item.get_text()
    output = ROOT/'figures'
    fig.savefig(output/(name+'.pdf'), metadata={'Title': name, 'Creator': 'Matplotlib'})
    fig.savefig(output/(name+'.svg'))
    fig.savefig(output/(name+'.png'), dpi=180)
    provenance = dict(input_sha256=sha(INPUT), script_sha256=sha(__file__),
        width_inches=fig.get_figwidth(), height_inches=fig.get_figheight(),
        minimum_font_points=min(t.get_fontsize() for t in text), font='Times New Roman',
        text_within_canvas=True, interval='range of family means, not confidence interval',
        points=points)
    (output/(name+'.data.json')).write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plt.close(fig)


def dependence(data):
    fig = plt.figure(figsize=(6.3, 3.05))
    axes = [fig.add_axes([x,.25,.245,.56]) for x in (.105,.425,.745)]
    points = []
    for j, arm in enumerate(('native_self','replay_self')):
        ax = axes[j]
        style(ax)
        for branch, color, marker in (('agree',COLORS['copy'],'o'),
                                       ('conflict',COLORS['independent'],'^')):
            rows = sorted([x for x in data['dependence_points'] if x['model']==PHI
                and x['history_arm']==arm and x['branch']==branch], key=lambda x:x['rho'])
            assert all(x['n']==x['planned']==24 and x['selection']=='common complete parents' for x in rows)
            x = [r['rho'] for r in rows]
            y = [r['estimate'] for r in rows]
            low, high = ([r['family_range'][i] for r in rows] for i in (0,1))
            ax.fill_between(x,low,high,color=color,alpha=.14,lw=0)
            ax.plot(x,[r['normative'] for r in rows],ls='--',color=color,lw=1.2)
            ax.plot(x,y,color=color,marker=marker,ms=4.8,lw=1.5)
            points.extend(rows)
        ax.set_xlim(-.04,1.04)
        ax.set_ylim(-.08,1.9)
        ax.set_yticks([0,.5,1,1.5])
        ax.set_xticks([0,.5,1])
        ax.set_xlabel('Copying probability, ρ')
        if j==0: ax.set_ylabel('Effective evidence weight')
        else: ax.tick_params(labelleft=False)
        fig.text(.2275 if j==0 else .5475,.04,
                 '(a) Native history' if j==0 else '(b) Self replay',ha='center',va='bottom')
    ax=axes[2]
    style(ax)
    for arm,marker,fill in (('native_self','o',COLORS['copy']),('replay_self','s','white')):
        rows=[next(x for x in data['reminder_points'] if x['history_arm']==arm
            and x['condition']==c and x['branch']=='copy') for c in ('original','general_bayes','source_structure')]
        assert all(r['n']==r['planned']==48 for r in rows)
        x=[0,1,2]
        ax.fill_between(x,[r['family_range'][0] for r in rows],
                           [r['family_range'][1] for r in rows],color=COLORS['copy'],alpha=.07,lw=0)
        ax.plot(x,[r['estimate'] for r in rows],marker=marker,mfc=fill,
                color=COLORS['copy'],lw=1.5,ms=4.8,label='Native history' if arm=='native_self' else 'Self replay')
        points.extend(rows)
    ax.axhline(0,color=COLORS['normative'],ls='--',lw=1.2)
    ax.set_ylim(-.08,1.9)
    ax.set_yticks([0,.5,1,1.5])
    ax.tick_params(labelleft=False)
    ax.set_xlim(-.25,2.25)
    ax.set_xticks([0,1,2],['Ordinary','Bayes','Source'])
    ax.set_xlabel('Reminder')
    ax.legend(frameon=False,loc='upper left',fontsize=9.5,handlelength=1.1,borderaxespad=.2)
    fig.text(.8675,.04,'(c) Copy updates',ha='center',va='bottom')
    handles=[Line2D([],[],color=COLORS['copy'],marker='o',label='Agreeing reading'),
             Line2D([],[],color=COLORS['independent'],marker='^',label='Conflicting reading'),
             Line2D([],[],color=COLORS['normative'],ls='--',label='Normative weight')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1),ncol=3,
               frameon=False,fontsize=9.5,columnspacing=1.1,handlelength=1.7)
    export(fig,'dependence_diagnosis',points)


def observations(data):
    fig=plt.figure(figsize=(6.3,3.15))
    ax=fig.add_axes([.10,.245,.22,.57])
    weight_ax=fig.add_axes([.515,.245,.20,.57])
    error_ax=fig.add_axes([.78,.245,.20,.57])
    points=[]
    style(ax)
    for rule,color,marker in (('direct',COLORS['ordinary'],'o'),('selector',COLORS['explicit'],'s')):
        rows=[next(x for x in data['expression_errors'] if x['rule']==rule and x['condition']==condition)
              for condition in ('general','explicit')]
        assert all(x['statuses']=={'ok':192} for x in rows)
        values=[r['metrics']['posterior_error'] for r in rows]
        ax.fill_between([0,1],[x['family_range'][0] for x in values],
                        [x['family_range'][1] for x in values],color=color,alpha=.10,lw=0)
        ax.plot([0,1],[x['estimate'] for x in values],color=color,marker=marker,
                lw=1.5,ms=4.8,label='Direct' if rule=='direct' else 'Selector')
        for row in rows:points.append(dict(panel='expression',rule=rule,condition=row['condition'],metric=row['metrics']['posterior_error']))
    ax.set_xlim(-.15,1.15)
    ax.set_ylim(0,.08)
    ax.set_yticks([0,.02,.04,.06,.08])
    ax.set_xticks([0,1],['General','Explicit'])
    ax.set_xlabel('Conditioning question')
    ax.set_ylabel('Posterior MAE')
    ax.legend(frameon=False,loc='upper right',bbox_to_anchor=(1.08,1.34),fontsize=9.5,handlelength=1.4)
    fig.text(.21,.04,'(a) Equivalent expressions',ha='center',va='bottom')
    settings=((0,'agree'),(0,'conflict'),(.5,'agree'),(.5,'conflict'),(1,'agree'))
    labels=['ρ=0, agree','ρ=0, conflict','ρ=0.5, agree','ρ=0.5, conflict','ρ=1, copy']
    for k,plot_ax in enumerate((weight_ax,error_ax)):
        style(plot_ax,horizontal=True)
        plot_ax.set_ylim(-.45,4.45)
        plot_ax.set_yticks(range(5),labels if k==0 else [])
        plot_ax.invert_yaxis()
        metric='effective_weight' if k==0 else 'absolute_update_residual'
        for condition,color,marker,offset in (('ordinary',COLORS['ordinary'],'o',-.10),
                                               ('explicit',COLORS['explicit'],'s',.10)):
            for j,(rho,branch) in enumerate(settings):
                row=next(x for x in data['branch_updates'] if x['model']==PHI
                    and x['condition']==condition and x['rho']==rho and x['branch']==branch)
                m=row['metrics'][metric]
                assert row['planned']==32 and m['families']==4
                assert m['n']==32 if k==1 or condition=='explicit' else m['n'] in (31,32)
                low,high=m['family_range']
                plot_ax.plot([low,high],[j+offset,j+offset],color=color,lw=1.6)
                plot_ax.plot(m['estimate'],j+offset,marker=marker,color=color,ms=4.8,zorder=3)
                if k==0 and condition=='ordinary':
                    target=row['metrics']['normative_weight']['estimate']
                    plot_ax.plot(target,j,marker='D',mfc='white',mec=COLORS['normative'],ms=4.3,mew=.8,zorder=2)
                points.append(dict(panel='weight' if k==0 else 'update_error',condition=condition,
                    rho=rho,branch=branch,planned=row['planned'],metric=m))
        if k==0:
            plot_ax.set_xlim(-.08,1.45)
            plot_ax.set_xticks([0,.5,1])
            plot_ax.set_xlabel('Effective evidence weight', fontsize=9.5)
        else:
            plot_ax.set_xlim(0,.165)
            plot_ax.set_xticks([0,.08,.16])
            plot_ax.set_xlabel('Mean absolute residual')
    fig.text(.615,.04,'(b) Selective updates',ha='center',va='bottom')
    fig.text(.88,.04,'(c) Update errors',ha='center',va='bottom')
    handles=[Line2D([],[],color=COLORS['ordinary'],marker='o',lw=0,label='Ordinary'),
             Line2D([],[],color=COLORS['explicit'],marker='s',lw=0,label='Explicit observations'),
             Line2D([],[],color=COLORS['normative'],marker='D',mfc='white',lw=0,label='Normative weight')]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(1,1.005),ncol=2,
               frameon=False,fontsize=9.5,columnspacing=.6,handletextpad=.4,handlelength=1)
    export(fig,'observation_control',points)


if __name__=='__main__':
    setup()
    data=json.loads(INPUT.read_text(encoding='utf-8'))
    assert data['new_calls']['total']==4992
    dependence(data)
    observations(data)
    print(json.dumps(dict(state='passed',figures=['dependence_diagnosis','observation_control'],input_sha256=sha(INPUT))))
