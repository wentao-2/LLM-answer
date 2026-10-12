"""Exact aggregate tables and a human-readable report from completed audited data."""
from pathlib import Path
import csv,json,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from common import PHI,QWEN,load,sha

NAMES={PHI:'Phi-4',QWEN:'Qwen3-8B','native_self':'Native history','replay_self':'Self replay',
       'original':'Original','general_bayes':'General Bayes','source_structure':'Source structure'}


def number(x):
    if x is None:return '无法估计'
    return f'{x:.3e}' if 0<abs(x)<1e-4 else f'{x:.4f}'


def choice(row):
    if row['common_all']['effective_weight']['n_families']==6:
        return row['common_all'],'common complete parents'
    return row,'finite valid pairs'


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert load(ROOT/'audit/AUDIT.json')['state']=='passed'
    assert load(ROOT/'audit/AGGREGATES.json')['state']=='passed'
    data=load(ROOT/'analysis/RESULTS.json')
    errors={(r['model'],r['module'],r['history_arm'],r['rho'],r['condition'],r['branch']):r for r in data['summaries']}
    pairs={(r['model'],r['module'],r['history_arm'],r['rho'],r['condition'],r['branch']):r for r in data['pair_summaries']}
    full=[]
    for key,error in errors.items():
        pair=pairs.get(key)
        row={k:error[k] for k in ('model','module','history_arm','rho','condition','branch')}
        row.update(n_planned=error['n_planned'],n_valid=error['n_valid'],n_endpoints=error['n_endpoints'],
            posterior_mae=error['posterior_absolute_error']['equal_family_mean'],
            posterior_mae_lower=error['error_lower']['equal_family_mean'],
            posterior_mae_upper=error['error_upper']['equal_family_mean'],
            choice_accuracy=error['choice_correct']['equal_family_mean'],
            choice_accuracy_lower=error['accuracy_lower']['equal_family_mean'],
            choice_accuracy_upper=error['accuracy_upper']['equal_family_mean'],
            n_complete_probability_pairs=pair['n_complete_probability_pairs'] if pair else None,
            n_finite_logit_pairs=pair['effective_weight']['n'] if pair else None,
            effective_weight=pair['effective_weight']['equal_family_mean'] if pair else None,
            weight_absolute_residual=pair['absolute_weight_residual']['equal_family_mean'] if pair else None,
            update_absolute_residual=pair['absolute_probability_update_residual']['equal_family_mean'] if pair else None,
            update_absolute_residual_lower=pair['absolute_update_residual_lower']['equal_family_mean'] if pair else None,
            update_absolute_residual_upper=pair['absolute_update_residual_upper']['equal_family_mean'] if pair else None,
            common_complete_parent_count=error['common_all']['n_planned'],
            common_posterior_mae=error['common_all']['posterior_absolute_error']['equal_family_mean'],
            common_effective_weight=pair['common_all']['effective_weight']['equal_family_mean'] if pair else None)
        full.append(row)
    write_csv(ROOT/'analysis/aggregate_summary.csv',full)
    lines=['# 增强实验结果','',f'分析文件 SHA256：`{sha(ROOT/"analysis/RESULTS.json")}`。',
        '', '新增 2,112 次科学调用与 384 个原对照保留为不同来源。所有估计来自已终止阶段。六数值家庭等权，完整概率配对、有限 log odds 配对和原设计分母分别记录。',
        '', '| 模型 | 新增调用 | 有效 | 无效 | 截断 | 工程 error |', '|---|---:|---:|---:|---:|---:|']
    for model in (PHI,QWEN):
        counts=data['new_status_counts'][model]
        lines.append(f'| {NAMES[model]} | {sum(counts.values())} | {counts.get("ok",0)} | {counts.get("invalid",0)} | {counts.get("truncated",0)} | {counts.get("error",0)} |')
    native_agree=[r for r in data['pair_rows'] if r['model']==PHI and r['module']=='partial_dependence'
                  and r['history_arm']=='native_self' and r['branch']=='agree']
    unchanged=sum(r['probability'] is not None and r['probability']==r['none_probability'] for r in native_agree)
    native_conflict=[r['effective_weight']['equal_family_mean'] for r in data['pair_summaries']
                     if r['model']==PHI and r['module']=='partial_dependence'
                     and r['history_arm']=='native_self' and r['branch']=='conflict']
    baseline=[r['posterior_absolute_error']['equal_family_mean'] for r in data['summaries']
              if r['model']==PHI and r['module']=='partial_dependence'
              and r['history_arm']=='native_self' and r['branch']=='none']
    iw=lambda arm,condition:pairs[(PHI,'source_intervention',arm,None,condition,'copy')]['effective_weight']['equal_family_mean']
    lines += ['', '## 主要发现与论文定位', '',
        f'**新增结果没有呈现规范的部分依赖响应。** Phi-4 原生历史的 {unchanged}/{len(native_agree)} 个一致记录输出与各自 none 基线完全相同。冲突权重在各 rho 下为 {min(native_conflict):.4f} 至 {max(native_conflict):.4f}，规范目标为 1。相应 none 基线 MAE 为 {min(baseline):.8f} 至 {max(baseline):.8f}，因此需要把后续更新错误与基线后验位置分开解释。', '',
        f'**两个提醒均未纠正原生历史中的复制多计数。** 原提示、一般 Bayes 提醒和来源结构提醒的复制权重依次为 {iw("native_self","original"):.4f}、{iw("native_self","general_bayes"):.4f}、{iw("native_self","source_structure"):.4f}，复制规范目标为 0。独立读数的权重仍接近 1，完整逐项配对差见后表。', '',
        f'**提醒在 self replay 中增加了复制误差。** 复制权重由原提示的 {iw("replay_self","original"):.4f} 增至一般提醒的 {iw("replay_self","general_bayes"):.4f} 和来源提醒的 {iw("replay_self","source_structure"):.4f}。新增干预应作为实际负结果报告。', '',
        '这些结果把依赖诊断扩展到五个复制概率设置和两种提醒条件。可支撑的贡献是具有规范预测的诊断与条件分析，新增提醒没有形成有效纠正方法。论文整合应连接已保存的复制端点、当前混合规则下的不对称更新和提醒的负结果，不按实验批次增加独立故事。']
    lines += ['', '## 实验问题与规范预测', '',
        '部分依赖实验改变第二条记录来自复制分支的概率 rho，保持保存的父答案、可靠性和呈现条件。来源干预实验比较原提示、一般 Bayes 提醒和具体来源结构提醒，检查复制错误的减少是否伴随独立证据权重的保留。两个新增提醒不提供真实状态或正确后验。', '',
        '令 $D\\sim\\mathrm{Bernoulli}(\\rho)$，$D$ 独立于状态 $H$ 和读数 $S,T$。给定 $H$，$S$ 与 $T$ 条件独立，可靠性均为 $r$。第二条记录为 $B=S$（当 $D=1$）或 $B=T$（当 $D=0$）。因此', '',
        '$$P(B=b\\mid S=s,H=h)=\\rho\\mathbf{1}[b=s]+(1-\\rho)P(T=b\\mid H=h).$$', '',
        '有效独立证据权重用同一条件的无新增读数续答作为基线：', '',
        '$$\\widehat w=\\frac{(2B-1)\\{\\operatorname{logit}(p_B)-\\operatorname{logit}(p_{\\varnothing})\\}}{\\log[r/(1-r)]}.$$', '',
        '一致记录的规范权重为', '',
        '$$w^*_{\\mathrm{agree}}(\\rho)=\\frac{\\log\\{[\\rho+(1-\\rho)r]/[\\rho+(1-\\rho)(1-r)]\\}}{\\log[r/(1-r)]}.$$', '',
        '当 $r>1/2$ 时它从 1 单调下降至 0。冲突记录排除了复制分支，所以当 $\\rho<1$ 时规范权重始终为 1。这个区别使实验能够区分依赖处理、忽略依赖和统一压低更新三种输出行为。该推导是 Bayes 法则在本设计中的应用。']
    lines += ['', '## 部分依赖响应', '',
        '一致读数的规范有效独立证据权重随复制分支概率下降。冲突读数在 rho 小于 1 时的规范权重为 1，rho 等于 1 时该事件不可能发生。下表优先显示覆盖六家庭的共同完整父集合，否则使用有限有效配对。原分母各为 24，全部逐项值和缺失界见 CSV 与 JSON。', '',
        '| 模型 | 呈现 | rho | 一致权重 | 冲突权重 | 一致有限配对 | 冲突有限配对 | 估计集合 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for model in (PHI,QWEN):
        for arm in ('native_self','replay_self'):
            for rho in (0,.25,.5,.75,1):
                agree=pairs[(model,'partial_dependence',arm,rho,None,'agree')]
                aa,aset=choice(agree)
                conflict=pairs.get((model,'partial_dependence',arm,rho,None,'conflict'))
                cc,cset=choice(conflict) if conflict else (None,None)
                value=number(cc['effective_weight']['equal_family_mean']) if cc else '不可发生'
                count=f'{cc["effective_weight"]["n"]}/24' if cc else '不生成'
                selected=aset if not cset or aset==cset else aset+' / '+cset
                lines.append(f'| {NAMES[model]} | {NAMES[arm]} | {rho:g} | {number(aa["effective_weight"]["equal_family_mean"])} | {value} | {aa["effective_weight"]["n"]}/24 | {count} | {selected} |')
    qcommon=next(r for r in data['common_complete_sets'] if r['model']==QWEN and r['module']=='partial_dependence')
    qfamilies={r['family_id'] for r in data['error_rows'] if r['model']==QWEN and r['module']=='partial_dependence'
               and r['common_parent_all_cells']}
    lines += ['',
        f'Qwen3-8B 跨全部 28 个单元的共同完整父集合仅有 {qcommon["n_complete"]}/24 项，覆盖 {len(qfamilies)}/6 个数值家庭。因此图与上表使用各点覆盖六家庭的有限有效配对，父集合随 rho 和分支变化。曲线表示条件估计，不能视为同一完整队列的变化或据此排序两个模型。全分母概率误差与更新残差的缺失界保留在精确 CSV 和原分析中。', '',
        '一个可核查的错误实例为 Phi-4 原生历史，prior=0.82、r=0.92、S=1、B=0、rho=0.25。正确的联合后验为 0.82，保存输出为 0.2838。输出中的可见计算把目标写成 P(H=1|B=0)，省略了已观察的 S=1。该数值接近只使用 B 的后验 0.283737。记录 ID 为 `9765e39ec7a5aca747c55b60`，完整提示与输出在 `queues/phi/merged.sqlite`。该实例解释一种输出错误，完整个体分布仍以所有计划记录为准。']
    lines += ['', '## 来源干预', '',
        'Original 来自原实验终态输出，General Bayes 与 Source structure 是两个新增条件。各条件使用自己的 none 基线。复制规范权重为 0，两个独立分支为 1。复制误差减少须与独立证据的保留同时解释。下表权重使用与上表相同的集合选择规则。', '',
        '| 呈现 | 条件 | 复制权重 | 独立同向权重 | 独立反向权重 | 复制后验 MAE | 复制有限配对 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm in ('native_self','replay_self'):
        for condition in ('original','general_bayes','source_structure'):
            selected=[]
            for branch in ('copy','independent_agree','independent_conflict'):
                row=pairs[(PHI,'source_intervention',arm,None,condition,branch)]
                selected.append(choice(row)[0])
            er=errors[(PHI,'source_intervention',arm,None,condition,'copy')]
            mae=(er['common_all'] if selected[0] is pairs[(PHI,'source_intervention',arm,None,condition,'copy')]['common_all'] else er)['posterior_absolute_error']['equal_family_mean']
            lines.append(f'| {NAMES[arm]} | {NAMES[condition]} | {number(selected[0]["effective_weight"]["equal_family_mean"])} | {number(selected[1]["effective_weight"]["equal_family_mean"])} | {number(selected[2]["effective_weight"]["equal_family_mean"])} | {number(mae)} | {selected[0]["effective_weight"]["n"]}/48 |')
    lines += ['', '## 相同父答案上的干预差值', '',
        '下表直接使用冻结分析中的逐项配对差，变化方向为新增提醒减原提示。负的后验 MAE 或绝对权重残差变化表示误差减小。优先使用覆盖六家庭的共同完整父集合，其他情况保留有效配对的条件估计。不能把不同完整集合的均值相减当成配对效果。', '',
        '| 呈现 | 提醒 | 分支 | 后验 MAE 变化 | 绝对权重残差变化 | 完整概率比较 | 有限权重比较 | 估计集合 |',
        '|---|---|---|---:|---:|---:|---:|---|']
    branch_names={'copy':'复制','independent_agree':'独立同向','independent_conflict':'独立反向'}
    for row in data['contrast_summaries']:
        common=row['common_all']
        if common['absolute_weight_residual_change']['n_families']==6:
            selected,mode=common,'common complete parents'
        else:
            selected,mode=row,'valid paired contrasts'
        lines.append(f'| {NAMES[row["history_arm"]]} | {NAMES[row["condition"]]} | {branch_names[row["branch"]]} | {number(selected["posterior_absolute_error_change"]["equal_family_mean"])} | {number(selected["absolute_weight_residual_change"]["equal_family_mean"])} | {selected["n_complete"]}/48 | {selected["absolute_weight_residual_change"]["n"]}/48 | {mode} |')
    lines += ['', '## 验证与解释范围', '',
        '原始 ID、科学 metadata、父答案、模型版本、全量权重 SHA、原生模板、公式、读数规则和分母已分别核验。逐项概率误差、log odds 权重、缺失输出界、共同父集合和干预配对差均独立复算。端点概率不裁剪，有限 log odds 不可估计时明确保留该状态。', '',
        '区间表示六个家庭均值的范围，不是总体置信区间。这是固定数值网格的受控诊断。有效独立证据权重描述输出行为，不能据此识别模型内部算法。两个模型的原始运行接口及预算不同，结果用于各模型内部比较。', '',
        '新增原始输出见 `queues/`，复用的原对照与保存父答案见 `inputs/`。完整科学分析、逐项配对差与图件见 `analysis/` 和 `figures/`。`aggregate_summary.csv` 提供全部分支的精确汇总，`intervention_contrasts.csv` 提供逐项干预比较。', '',
        '图件为论文实际跨栏宽度 6.3 英寸，使用 Times New Roman，正文文字至少 9.5 pt，子图标题在图下。可编辑 SVG 与矢量 PDF 来自 `assets/plot_results.py`。正式稿件尚未在本轮修改。']
    (ROOT/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(ROOT/'RESULTS_REPORT.md'),aggregate_rows=len(full),analysis_sha256=sha(ROOT/'analysis/RESULTS.json'))))


if __name__=='__main__':main()
