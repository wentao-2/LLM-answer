# I16 第三轮：归属差、规范误差与提示条件的独立网格验证

制定日期：2026-10-08。用户已明确选择鉴别路线并要求启动补充实验。本轮受已完成的前两轮启发，属于前瞻性新验证；不能追溯称为前两轮预注册。此前全部数据、失败、协议及冻结源码保持原样。能力及主矩阵、分析、候选模型和预算在第三轮科学输出前同时封存。

## 问题、预测与可解释范围

核心问题：相同证据下的self−other报告差究竟反映self超出规范更新、other低于规范更新，还是随提示改变的计算误差？检验旧结果在新数值网格、两种新证据表述及具备基础能力的其他checkpoint上是否保持。

预先比较三项预测：（1）在正、零、负通道中，回答方向和大小随已知信息价值改变；（2）S可见时冗余A的规范增量为零，而独立A仍应引起相应更新，包括一致和冲突观测；（3）compatible输出指令相对restrictive可能减小归属差与规范误差。保留反向或零结果，不能只挑支持这些预测的条件。

归属仍是独立赋值的外生作者文字，单条user提示不含真实自生成历史。native thinking与最终可见解释是不同层次；输出措辞改变同时包含长度、词汇和要求变化，不能称为纯内部思考干预。空思考和解释长度是生成后诊断，不认定心理承诺、神经机制或因果中介。新模板仍属合成概率任务，不称为自然任务泛化。

## 固定设计

H为二元状态，先验p；二元S的对称可靠度为r。positive中A=S，negative中A=1−S，zero中A是独立公平硬币；independent中A=T，T与S给定H独立、可靠度同为r。所有规则在生成前确定，无选择或按一致性停止。规范后验由完整隐变量枚举独立验证。

S隐藏时只提供A，正/零/负通道分别产生正向、零和反向证据。S与A都显示时只使用合法观测；A隐藏时只提供S及生成规则。independent保留一致与冲突。A隐藏基线的共享引用始终只是一条原始记录，缺失界先合并该记录的所有系数。

主网格p=.18/.32/.68/.82，r=.66/.78/.92，共12数值设置，互补先验成对后6个family。两种新sensor表述为完整条件概率表likelihood_v3及精确频率frequency_v3；后者明确频率不是额外观测。共24个数值×表述单元。与前两轮主矩阵的(p,r)组合及提示均不重叠，检查结果另存准备记录；能力网格复用开发数值，不冒充新数值确认。

每个合格配置2736条唯一主调用：生成机制部分1872条，使用scoped+compatible；wording另增864条。机制部分含S隐藏3通道×2个A×3归属×24=432，S/A显示12合法观测×3归属×24=864，A隐藏4通道×2个S×3归属×24=576。归属为self/other/neutral，匹配self/other仅改变作者文字。

wording限定S/A显示的positive与independent、self/other，交叉scoped/legacy及compatible/restrictive，共1152分析单元；其中288条与机制部分共享，无重复调用。所有相同数值/观测/表述的对照完整配对，不混合非法世界。

## 能力资格与四个候选

每配置64条无归属题：p=.25/.75、r=.65/.90、两新模板，positive/zero/negative（S隐藏）和baseline（仅S）各16。四组分别有效率≥95%且有效值family等权MAE≤.05才合格；16条的分母意味着每组实际必须16/16有效。只依据此无归属资格决定入主矩阵，不查看归属效果选模型。失败原样报告，无择优补跑、放宽或新增候选。全部失败则止于gate并报告限制。

| 配置 | 只读模型路径 | 原生模式 | 输出token上限 |
|---|---|---|---:|
| qwen3-8b-thinking | /workspace/models/Qwen/Qwen3-8B | enable_thinking=True | 12288 |
| qwen35-9b-thinking | /workspace/models/Qwen3.5-9B | enable_thinking=True，文本输入 | 12288 |
| mistral-7b-instruct | /workspace/models/lto-table2/Mistral-7B-Instruct-v0.1 | native instruct | 4096 |
| olmo-7b-instruct | /workspace/models/lto-table2/OLMo-7B-0724-Instruct-hf | native instruct | 2048 |

两个Qwen是同家族不同checkpoint，不能计作跨家族复现。Mistral/OLMo仅通过资格才进入主矩阵。不同原生模式及上限并非等计算量比较。此前Mistral复制示例、OLMo格式无效等失败完整保留；本轮新读出不重新解释或重算旧数据。格式不可测不等于无推理能力。若Qwen3不通过、其他通过，按原资格运行其他合格配置，但不得冒称成功复制旧主配置。

## 冻结读出与执行

unique-final-json-v1：最后唯一一行必须是FINAL_JSON:后接只有p_h1键、值在[0,1]的JSON对象。可在此前写解释；不得有多个marker、重复key、额外key、尾随正文、NaN/bool等非概率。所有模型必须EOS，thinking模型还须完整闭合think后才解析。截断思考不作为答案，不事后宽松抽数字。restrictive要求只给最终行、compatible允许先推理，两者最终marker要求完全相同。

这是前瞻性读出修订，不是只更换模型的旧协议复现。原生chat模板应用后tokenize使用add_special_tokens=False，避免重复BOS；Qwen3.5使用AutoModelForImageTextToText但只传文本。所有2800题×4tokenizer进行长度检查；唯一Qwen3.5的16token“Write the word ready.”调用只验证新loader，不计科学证据。

BF16、greedy、num_beams=1、SDPA、batch4；OOM仅减batch，不改变科学参数。输入上限1600且禁止截断；每次核验输入+输出不超过模型原生上下文。四卡按manifest索引mod4分片，每片gate每模型16条，main每合格模型684条。记录实际batch、token、模型来源、EOS、闭合及完整原始输出。seed只决定顺序。

总预算最多4×64+4×2736=11200条终止调用。每候选最多2800条，上限输出合计2800×(12288+12288+4096+2048)=86016000 token，输入预算最多17920000 token；未合格模型只消耗gate。不为补足预算扩样。耗时按实测吞吐更新。运行失败每片最多3次；停止于固定矩阵完成、全部资格失败或需修复的明确运行故障，绝不因科学效果不好无限延长。

## 主要分析

每配置独立报告完整分母、有效/无效/截断、条件MAE、有符号残差、平均绝对差、模板与六family逐项结果。六family仍是固定设计，不报告bootstrap、p值或自动等价结论。基础无归属资格与主矩阵规范表现分开报告，gate通过不保证归属条件都合格。

沿用第二轮预先指定的方向化更新、规范增量及残差，S可见条件与同协议A隐藏基线配对。报告全部正/零/负/独立、一致/冲突及self/other/neutral，wording完整2×2交互。所有空思考、截断和阴性保留。两模板分开展示，不以少数大计算错误概括普遍归属偏好。

新增主要诊断为同一self/other完整配对的规范误差分解：d=2A−1，e_self=d(p_self−oracle)，e_other=d(p_other−oracle)，gap=e_self−e_other。分别报告这三项及绝对误差，不能只凭正gap称self过度更新。预设±.02容差用于描述self接近规范/other不足等类别；它只是概率尺度的描述阈值，不是显著性或心理机制标签。主线判断必须同时读数值误差、信息价值和措辞交互。

对有缺失概率的所有完整设计线性指标，已知值固定，未知原始概率限制[0,1]，计算最坏上下界；先合并共享记录系数。完整配对估计与这些敏感性界分列，不能当置信区间或补造回答。容差类别只报告有效配对计数和缺失分母，不对缺失做确定分类。

## 保存、推进与停止

代码独立位于validation_v3，gate为runs/validation-gate，main为runs/validation-main。两阶段manifest及协议在gate科学输出前封存；run/source与config/manifest/protocol均哈希验证。复用工具的phase mechanism-gate/mechanism及design_snapshot/MECHANISM_PROTOCOL.md是历史文件名，内容为本第三轮协议，不代表第二轮run或旧协议。

逐样本SQLite FULL提交，128条或每模型结束backup，严格合并；已提交终止记录不重算或覆盖。NFS nolock通过原子controller.claim及hostname/boot_id/start ticks防重复。不能删除外主机claim或按缓存PID杀进程。90分钟真实无进展才处理验证身份的自有worker，最多3次失败预算，不无限recover。冻结后软件修复必须另建有兼容迁移记录的新run。

gate每小时有界检查，完成后审计/同步备份、依据资格运行advance_validation.py冻结主run并启动，然后同一自动化改为两小时。只下载周期backup，不普通复制活跃主SQLite。main全部ID集唯一终止、四片结束、严格merge/analysis及最终下载验收后，生成独立第三轮报告并删除本轮自动化。所有gate失败同样保存报告并停止。无效、截断、零/反向效果不触发修题、换seed、补跑或追加模型。
