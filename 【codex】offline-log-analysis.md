# 离线逐轮日志分析

本工具只读本地 `main3.py` 服务端 stdout 日志，解析 `YYYY-MM-DD HH:MM:SS,mmm | {JSON}` 中 `event=turn` 与 `event=task` 的记录。输入日志保留在操作者本地；工具不联网、不执行日志内命令、不写文件，只向 stdout 输出一份 JSON 摘要。

```sh
cd CoreGeek
python3 analyze_logs.py --input /absolute/path/to/server.log --pk PK编号 --half challenger > /tmp/nightwatch-summary.json
```

`--pk` 与 `--half` 必填，`--half` 可为 `challenger` 或 `defender`。它们是操作者提供的元数据；日志本身没有足够字段核实 PK、上传包 SHA256、完整提交 SHA、终局或平台结论。不要从文件名推断这些值。

摘要按队伍类型、队伍 ID、决策会话与 buildId 分组。身份字段缺失时标 `identityUnknown`，不闭合夜晚。每组报告原始帧、唯一帧、排除的重复帧、首末轮、重复/倒序/内部缺口及被截断段计数。重复轮的全部记录都排除出统计，倒序仍保留告警。白天/夜间 `processing` 与 `decide` 分别给总览和逐日的样本数、p50、p95、最大值、超过 500ms 数。百分位采用 nearest-rank；耗时是服务端计时，不是裁判端到端耗时。没有声明预期轮次范围时，首尾完整性为未知。缺失、null、错误类型和数值 0 分开计数。

经济摘要包含金币范围、逐段列明起止轮次/长度/日/昼夜的空指令区间及最长长度、寻路搜索/计算/扩展计数、`defensePlanning.unsafeDayWork` 及规划截断轮数。空指令仅认非负整型 `commandCount=0`；缺失、null、布尔/浮点/负数等错误类型分别计数。重复轮排除，缺帧及日/昼夜边界切段。基地摘要包含 HP/等级范围、升级券使用**请求**数与相邻帧可见的等级上升数。HP独立读取 `bases.our.health`；等级仅在 `ourStructures.items` 可见同 ID 基地时读取。结构列表截断时，已知 HP/等级仍保留，未见基地的等级为未知；防线全集可比性为未知。这些只是日志可见事实；发出 `use` 指令不等于平台执行成功。

`tasks.instances` 按 build、队伍、session 与 instance 隔离，联合 `turn` 和 `task` 的固定元数据。两路具有相同请求指纹且内容一致时，同一轮同一实例的 `submitAnswer` **请求**只计一次，并保留证据来源。统计先跨来源按指纹分组：同指纹重复不增加范围，同指纹计数冲突取该组最小/最大值，不同指纹各组相加。无指纹记录再按可能与已知请求重合的保守范围合并；缺共同指纹或记录冲突时不强拼，精确数为 `null`，另给上下界、原始观察记录数和 `submit_association_unknown`。同指纹一条有提交、一条无提交时范围为 0–1；两个不同指纹各提交一次时范围为 2–2。结束事件分别按 `decision.taskEnd.instanceId` 或 `endedTaskInfo.instanceId` 归属，错误码也只按 `answerResult.association` 或结束元数据关联；无法关联的原始错误不称为当前实例错误。摘要只保留求解状态/固定原因、命令结果类别、修正/截断标志、错误码及观察轮次；每实例 timeline 最多 64 轮，按轮稳定合并来源与固定元数据，超限标 `timelineTruncated`。不输出题面、答案、命令、prompt、模型输出、指纹全文或异常描述。覆盖率列出两类记录数、带指纹记录数、task-only 实例、重复/冲突、倒序及 task 轮次缺口。身份缺失时单独分组，不强行关联。结束、`completed`、错误类别、金币或分数变化都不构成成功证据，`successStatus` 始终为 `unknown`。旧 `task_detail` 仍被忽略，其正文不会进入摘要。

`pressure.nights` 从 `decision.pressureShadow.prediction` 与 `verification/recentNights` 按夜次汇总双方。夜间伤害事件 `event_observed` 与次日 `final` 属同一夜，优先用 final；只有事件时取该夜最新观测，保留局部证据。明确受伤可由事件证明；无伤仅在 `final.complete=true` 且该侧 `actualBaseDamage=false` 时成立。缺尾帧、缺验证或身份不明时实际状态为未知。预测已发出但夜晚尚未到来，也会列为不可评分的夜次。正式 `assessment` 与 `persistenceBaseline` 分开评分；`unknown` 预测列为不可评分，不算漏报或预测失败。汇总给出双方及合计的命中、漏报、误报、无伤正确和不可评分个数，不计算缺证据下的准确率。

每夜 `firstNightRobotDifference` 只取夜首帧并冻结，并随 final/recentNights 保存：`rawDifference` 为“朝我方机器人总数减朝敌方机器人总数”，`inferredOpponentAdditions=rawDifference+ourAppliedAdditions`。当前程序没有召唤路径，因此仅在 `program_does_not_summon` 和 `current_program_only` 范围内记录我方已生效新增数为 0；外部输入或部署差异仍标未知。双方自然生成总数相同是用户提供的前提；请求相对首批伤害结算的时序未由官方资料确认，所以结果标为 `conditional_observation` 并保留 `timing_unverified`，不做机器人类型归因。负差保留原值但标 `inconsistent_observation/negative_difference`，推导新增数为未知。夜首缺失、机器人字段缺失、未知目标或观察上限触发时保持未知，不从夜中或黎明回填。离线工具在只有黎明 final 记录时仍可恢复该夜观察，并优先 final。

`criticalWalls` 只汇总同 ID、同位置、同等级墙体在相邻帧中的可见失血，以及受损墙一格内可见目标机器人的有界计数。输出区分基地邻近墙和侧墙，记录受损墙帧、连续受损、相邻机器人观测、修复、升级及消失等事实。相邻不证明攻击因果，墙消失不证明被摧毁或形成可通行缺口，`criticalWallBreach` 和因果归属保持 `unknown`。缺基地时近基地/侧墙分类为未知；缺机器人、墙/机器人超过 512 上限或缺帧时，累计精确值为 `null`，同时保留已观察下界、固定未知原因和 `coverageComplete=false`。黎明相邻帧墙失血计入墙损及几何分类；由于黎明机器人列表已清空，机器人关联精确值为 `null` 并标 `dawn_robot_association_unknown`，前一夜末邻近数量仅作为 `dawnPriorAdjacentRobotObservations` 上下文。黎明是否观察及墙集合是否完整随 final 输出。位置变化、升级或缺帧不会延长连续受损；摘要不输出坐标或机器人 ID。

历史投资对照只取目标夜晚前白天最后一条匹配 `baselineNight` 的 `historyInvestment`，报告发出轮次及距入夜轮数。仅当黄昏记录、目标夜晚全部 60 帧、下一黎明帧、同基地同等级与连续 HP 轨迹具备时，从黄昏到夜首开始累加夜间相邻帧可见失血；缺帧、倒序/重复、回血、升级或基地缺席均标 `unknown`。防线变化按结构 ID、类型、等级、HP 和非负整型位置识别；位置缺失或格式无效使可比性未知，但保留可完整观测的基地失血值。已知防线变化标 `changed_unquantified`，不将旧夜与新夜称为可比。历史状态、未知原因及防线变化仅输出已知枚举；`no_observed_damage` 且损失为 0 是有效观测。这不是反事实估计，也不判定旧投资方案的预测命中。没有完整 SHA、包 hash 和终局信号时不得据此宣称 S2 验收通过。

输出仅包含预定的计数、枚举原因、分组标识和有限数值；不回显原始行、题面/答案、prompt、token、executeCmd 或异常正文。非逐轮事件、无法识别的行、半场与日志队伍类型不符的帧分别计数。源日志可能含敏感材料，处理与分享原文件仍需遵守项目规则。
