# I16 extension protocol

状态：2026-10-09 前瞻性设计，尚无新增科学输出。用户已授权跨家族、真实自生成历史和推理拆分三类实验。本文档、生成器、分析器、模型配置及所有固定矩阵须在 gate 输出前共同冻结。旧 R1–R3 证据和 `arr-overleaf` 保持原样。

## 1. 问题和最小证据包

| 模块 | 回答的问题 | 可识别比较 | 不能由该比较直接推出的结论 |
|---|---|---|---|
| 跨家族 | 归属与规范误差诊断在另一家族的合格 checkpoint 上呈现什么结果 | 相同外生概率世界、固定文本的 self/other/neutral 对比，以及完整归属误差分解 | 有限 checkpoint 不能代表整个模型家族，更不能自动推出稳定的普遍自偏好 |
| 真实历史 | 实际生成的初始回答进入后续对话时，规范诊断能否解释其保留与修正 | 同一真实初始回答分叉为 native continuation 和完整文本 replay，后者再交换归属 | native 角色与 user 引用之间的差异包含历史呈现方式，不能归因于不可观测的心理承诺 |
| 推理拆分 | 原生模式、输出措辞及最大生成预算分别如何影响准确度、归属和覆盖率 | 同一 Qwen3 权重下的 2 × 2 × 2 完整因子设计 | 原生模式是可干预的运行配置，实际 token 数和思考是否为空仍是生成后变量 |

设计依据是冻结 R3 `EXPERIMENT_PROTOCOL.md`、`design_snapshot/build_validation.py`、`source/run.py`、`source/analyze_mechanism.py`，以及 `ccfa-workfiles/writing/arr-draft/REVIEW_EXTRACTION.md` 的 E01–E04。本轮不复跑全部旧矩阵，不为了筛选有利效果追加候选或种子。

## 2. 固定模型和环境

主机已确认有 8 张空闲 RTX 4090，每张 49140 MiB。环境为 Python 运行时 `/workspace/codex-auto-research/env/bin/python`、torch 2.6.0+cu124、transformers 5.3.0。以执行端实际写入的环境和模型文件哈希为最终记录。

| ID | checkpoint / revision | 原生配置 | 用途 |
|---|---|---|---|
| `qwen3-8b-thinking` | 现有 Qwen/Qwen3-8B，沿用实际本地 provenance | 同一权重支持 `enable_thinking=True/False` | 已有参照、历史、模式因子 |
| `phi4-reasoning` | microsoft/Phi-4-reasoning，`1de18ec97600877ce63dbf60c73b998da99f0195` | Phi3ForCausalLM，官方 chat template，原生 think 必须闭合 | 固定非 Qwen 候选 1 |
| `mistral-nemo-instruct` | mistralai/Mistral-Nemo-Instruct-2407，`04d8a90549d23fc6bd7f642064003592df51e9b3` | MistralForCausalLM，官方 chat template，无 thinking toggle | 固定非 Qwen 候选 2 |

两候选均执行资格测试，通过者均进入跨家族矩阵。历史模块选 Qwen3，以及按 Phi-4、Mistral-Nemo 固定顺序首个合格候选。选择只依据无归属资格，不看 self gap。若二者均不合格，保留失败并完成 Qwen 模块，不用旧 Mistral/OLMo 替补。

每个模型用完整官方权重、tokenizer 和原生模板，不人为统一其内部控制 token 或 system 包装。本实验统一 greedy、BF16、num_beams=1、SDPA、禁用输入截断。Phi 的官方采样建议不同，本研究运行的是明确声明的 greedy 配置，不将其称为官方最佳能力复现。默认最大新 token 为 12288，Qwen 起始 batch 8、两个新候选起始 batch 2，OOM 仅减 batch，不改变科学条件。模型来源和实际模板哈希必须封存。

## 3. 无归属能力资格

新候选每模型 96 调用。复用 R3 能力网格 `prior=.25/.75`、`reliability=.65/.90`、两个模板。positive、zero、negative 的 S hidden 及仅 S 的 baseline 各 16，共 64。新增 independent 双信号组保留四个 `(S,A)` 组合，共 32。五组分别要求有效率 ≥ .95、有效报告 family 等权 MAE ≤ .05，独立组同时保留 agree/conflict 分项，不以其效果方向筛选。

四个 16 题组须 16/16 有效，32 题组须至少 31/32 有效。所有失败、截断和完整分母保留。新候选若组不合格则不进入跨家族和历史主矩阵。Qwen3 复用原 64 题资格，只补新增 independent 32 题。若新增组不合格，历史主诊断解释暂停并报告该能力结果。推理因子模块仍保留 on/off 的全部计划单元，因为模式造成的能力变化正是该模块的研究对象，不能筛除 off 失败。

单次技术 smoke 仅覆盖新增 loader/模板和首轮历史读出路径，不计入资格或科学结果。不得用 smoke 成功替代上述资格。Qwen 起始 batch 为 8，两个新候选为 2，OOM 只减 batch，并记录每条结果的实际 batch。全部 gate 单元必须已经终止后才能判定通过，尚未运行的单元不会提前解锁主实验。

## 4. 模块 A：跨家族固定子集

沿用 R3 主网格 `prior=.18/.32/.68/.82`、`r=.66/.78/.92`、`likelihood_v3/frequency_v3`，共 24 数值×模板单元、6 个互补先验 family。模板和正文提示逐字复用冻结 R3，保留 `origin_r3_id`。每合格新候选 1728 唯一调用：

| 部分 | 条件 | 调用数 |
|---|---|---:|
| 信息方向 | S hidden，positive/zero/negative，A=0/1，3 attribution，scoped+compatible | 432 |
| 依赖与归属 | S,A visible，positive 的 2 合法观测和 independent 的 4 观测，3 attribution，scoped+compatible | 432 |
| 共享基线 | A hidden，positive/independent，2 个 S，3 attribution，scoped+compatible | 288 |
| 指令诊断补充 | independent visible，self/other，完整 2 reminder × 2 instruction，扣除已共享的 scoped+compatible | 576 |
| 合计 | 全部均为固定 R3 主矩阵子集 | 1728 |

主要输出为各 checkpoint 的 attribution gap、同配对 self/other 方向误差、MAE、更新残差、逐模板/agree/conflict结果、输出覆盖和全设计缺失界。independent 指令比较使用完整八格。原 R3 Qwen 记录只作为相同子集的既有参照，不将新环境与旧环境差异解释成模型身份的纯因果效应。任何“跨家族一致”结论必须按实际通过者和实际结果逐项给出。

## 5. 模块 B：真实自生成历史

### 5.1 首轮和 oracle

每个历史模型生成 48 个独立首轮记录：12 个数值设置 × 2 模板 × 2 个 S。初始用户提示给出全部概率规则和已观察 S，要求自由报告对 H 的概率 `p0`。用户可见的最终答案为唯一 `FINAL_JSON: {"p_h1":概率}`。分析中的初始方向 `C=1[p0≥.5]`，恰好 .5 时预先约定 C=1。原生思考使用模型默认模式，raw output 和 token ids 原样保存，不按正确性或置信度筛选。

第二轮始终保留首轮的全部证据 S。设首轮消息为 `M=f(S, prior, r, template, model, U)`，U 为与 H 独立的生成随机源。因此在这些已知变量及 S 下，M 和 C 不提供新的 H 信息。本模块不假定模型生成的 C 具有传感器可靠度 r，也不套用旧 `A=T` 规则。

对每个合法首轮构造四种后续证据：没有新读数、复制 S 的记录 B=S、条件独立 T 的 B=0、条件独立 T 的 B=1。前两者的 oracle 为 Q(S)，后两者为 Q(S,B)。全部四个分支预先计划，不按 C、首轮正确性或是否同意 C 选分支。后两者才提供新的外生信息。

### 5.2 四个呈现臂

| 臂 | 实际传入 messages | 归属含义 |
|---|---|---|
| `native_self` | 原 user、真实产生的 assistant 最终可见答案、第二轮 user | 真正的两轮 assistant 历史，self 来源属实 |
| `replay_self` | 单条 user，逐字引用原 user 和同一 assistant 可见答案，再给相同第二轮任务 | 同一内容的 assigned self 归属 |
| `replay_other` | 与上一臂相同，仅所示作者改为 another assistant | assigned other，保留真实 producer provenance，不称实际另一模型生成 |
| `replay_neutral` | 同一引用内容，作者 undisclosed | neutral replay |

native 与 replay 使用相同的 `parent.final_output` 字节，不能先润色、删除错误或重新生成。首轮提示请求最终单行，但固定 parser 仍允许 marker 前的可见说明，若实际出现则全文保留在 final_output 并注入各臂，不据此剔除首轮。原生内部 think 不作为用户可见回答重新注入任何臂，但 raw output 完整保留。若官方模板在历史 assistant message 上再次改写可见内容，须在执行前明确并核对渲染后的 payload。native 只用 truthful self 标签，不在其 assistant 角色上强加“其实另一个模型”的矛盾标签。

`native_self − replay_self` 识别历史呈现方式的总效应。`replay_self − replay_other` 识别真实生成文本固定时的标签效应。二者不是同一个效应，不把两者相加或相互替代。完整同角色消息的重新加载在标准 transformer 推理中不是独立的“失去真实记忆”对照，因此不增加这个重复臂。

每个首轮最多 4 证据 × 4 臂 =16 后续调用。每模型 48+768=816，最多两模型 1632。第二轮固定 scoped、compatible、默认原生模式和 12288 cap。所有后续 children 的 ID 包含 parent item/model/record hash、臂和证据设置。

### 5.3 失败与估计目标

首轮须正常 EOS、所需 think 闭合、单字段严格读出才产生 children。非法或截断的首轮保留，不重试或补抽，后续标记 `not_constructed_parent_failure`。报告所有 48 个计划父记录的成功率，另给合法父记录上的每一臂终止/有效率。

历史效应的目标集合为预先规定生成流程实际获得的合法父记录 cohort，按 family 等权。生产失败时具体子提示尚未形成，不能把它装作一个已知提示上的缺失概率或生成虚构 transcript。对已形成 cohort 的第二轮缺失，照常计算固定 cohort 的全设计界。该条件目标和父层覆盖必须一起报告。

原始选择方向用 `d_C=2C−1`，新证据更新方向用 `d_B=2B−1`，两者显式区分。主要指标是各臂 MAE、同 cohort 的 history contrast、replay attribution gap、重复/独立证据相对同臂无新增证据的更新及规范增量。按 C 与 B 一致性分组属于生成后描述，不代替预先固定的主要平均。

## 6. 模块 C：原生模式 × 措辞 × 预算

仅 Qwen3 同一 checkpoint。因素为 `enable_thinking=True/False`、原 R3 compatible/restrictive 两种用户措辞，以及 `max_new_tokens=4096/12288`。三个因素独立交叉，不修改原始用户措辞来模拟 native 开关。

全部使用 scoped reminder、positive 和 independent 的 S,A visible 合法观测、self/other、两个 R3 模板和完整网格。调用量是 24 × (2+4) × 2 × 2 × 2 × 2 =2304。全部为此次环境下的新调用，只复用题目，不拼接旧 high-cap 结果充当新因子格，因而不复验整个旧 R3。

同一数值/观测/模板单元包含 2 attribution × 2 mode × 2 wording × 2 budget =16 格。聚合主效应和交互使用同一完整 16 格集合，分项给各自清楚的分母；另对全部计划 16 格计算缺失界。优先报告 mode 对 MAE 的平均效应，以及 mode 对 attribution gap 的交互，随后给 wording、budget 和相应交互。positive 与 independent 分开报告，不把不同比例规则混平均。

固定措辞与 cap 下的 mode 差异可解释为 native thinking 配置的总效应。cap 固定不等于实际计算量固定。实际生成长度、空 think、可见解释及截断率另列，不以它们进行生成后的选择或作为因果中介。off 不需要 think 闭合，on 必须闭合。禁止事后把含概率的截断输出修复为有效。

## 7. 读出、统计和缺失

第二轮及静态任务沿用 `unique-final-json-v1`，唯一最终行只有 p_h1。首轮同样用 `probability` 读出，唯一最终行只包含 p_h1，概率为非 bool 的有限 [0,1] 数。C 由已保存的有效概率确定，不从文本追加解析。重复字段、多个 marker、末尾文本均无效。始终要求正常 EOS，并按模型/运行模式要求原生 think 闭合。

保留旧 11 列 SQLite schema。metadata 保存完整 item、最终可见答案、parsed_answer、实际 mode、预算、模板 kwargs、tokens、终止原因、模型来源、渲染提示和 SHA256。每个 terminal record 只 INSERT，不 REPLACE。

沿用固定 family 等权，先在 family 内平均，再跨有完整单元的 family 等权。报告每个 family 及模板，避免用均值替代逐对分布。无 bootstrap、p 值或自动等价判定。均值负误差不能写成所有实例不足更新。误差分解使用同一配对集合，并输出严格符号计数及 ±.02 描述类别。

已形成提示的缺失概率属于 [0,1]。绝对误差范围为 [0,max(q*,1−q*)]。缺失界固定全部计划 unit 的 family 权重，按唯一原始记录合并系数后求界。共享首轮、无新增证据基线和旧子集引用不得重复当独立观测。完整集合点估计与全设计界目标不同，分别标注。科学零结果、反向结果、能力失败和截断都不是追加运行的触发器。

## 8. 调用量与执行顺序

| 阶段 | 最大新增调用 |
|---|---:|
| 两个新候选 gate | 192 |
| Qwen independent 资格补充 | 32 |
| 两个合格新候选跨家族主矩阵 | 3456 |
| Qwen 与首个合格新候选真实历史 | 1632 |
| Qwen native 因子矩阵 | 2304 |
| 总计 | 7616 |

调用量不包括少量技术 smoke。科学预算是上限，gate 失败和无法形成的历史 children 会减少调用数。固定矩阵完成即停止。八卡按模型和 generation profile 分配，保持每个 manifest item/model 唯一归属，可独立运行 cross-family、native-factorial 与 history-stage1，history-stage2 等待对应首轮记录冻结。

先冻结模型、协议、源码及 gate/main 固定矩阵，执行 gate。之后只按既定资格和候选顺序生成执行计划。history children 的生成规则预先冻结，parent terminal 记录完成后对其数据库快照封存，再确定性生成 children manifest，不能依据效应修改分支。输入 tokens + cap 必须不超过原生 context；history 对长 payload 做真实 tokenizer 预检，不允许静默裁剪。

## 9. 产物与待闭合项

生成器 `source/build_extension.py`，分析器 `source/analyze_extension.py`，runner `source/run_extension.py`。固定 manifest 为 gate、gate-independent、cross-family、native-factorial 和 history-stage1；history-stage2 由冻结父输出生成。数学检查覆盖 oracle 枚举、矩阵调用量、16格主效应、共享记录缺失系数、历史 oracle 不依赖 C 和父失败不伪造 children。技术检查只覆盖新增实现路径。

生成器和分析器已实现，`source/test_extension.py` 的 10 项 CPU 检查通过，固定五份 manifest 已生成。测试数值均为明确的数学 fixture，不计入科学结果。静态 manifest 再生必须字节一致，否则拒绝覆盖。公开 gate API 为 `analyze_gate(manifest_path, db_path, model)`，只有所有计划记录终止后才可能返回通过。历史 CLI 仅消费冻结首轮的已有有效概率和 final_output，不重解析 raw output。

冻结前必须闭合：两个新 checkpoint 下载和哈希、原生模板与输出闭合要求、Qwen on/off 实际模板区别、history 可见 assistant 字节在渲染后是否保持、单字段 parser 与 payload 限制，以及 gate/主矩阵/分析命令共同封存。模型是否通过资格由未来实际输出决定。当前没有新增实验结果，也不预填任何支持性结论。
