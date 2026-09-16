# S2防线恢复首批：内网执行提示

日期：2026-09-16。仓库icesword007/NightWatch，分支main，构建nightwatch-s2-defense-r1。源码提交68e24f7e1ce143f0bf953e496339adbcf7885f28，后续送测提交仅增加文档；实际参赛记录拉取后完整HEAD，不用buildId代替SHA。

## 执行目标

阅读AGENTS.md、本文和【codex】intranet-validation.md，拉取main，确认68e24f7是祖先；工作区有修改先报告，不覆盖/reset。按手册用CoreGeek打包上传，保留运行目录布局，排除.git/缓存/本地环境文件而不删除工作区文件。程序标准库无新依赖，本地307项测试通过，内网确认Python版本及实际包SHA。

本轮只验证新能力：首夜损失→次日固定墙补建→第二夜表现，以及真实波次统计/预测。S1基线7debfef已验收，不重复全面取证；无须一开始就跑1300回合，但自然继续的日志全部保留。本批预测仅辅助诊断和后续资源规划，不自动调整升级采购、扩大墙数或改变任务策略。

## 对手与场次

- 先选有望自然打过首夜、出现毁墙并继续至第二夜的常规对手。可参考此前4249/4323风格，但不假设对手当前版本/强弱未变。记录选择原因与我方实际阵营，不用发起方推断。
- 首场后最多追加两场，每场针对明确缺口：真实毁墙恢复、另一阵营镜像、第三夜预测核验。若一场覆盖多项，不重复刷；尽量覆盖双方阵营，无法安排说明限制。
- 参考预测在第一夜只能增长未知；两夜相邻完整观测后才有增长估计，至少到第三夜夜首轮（正常R331）才可初步核验这一预测。不要用同夜事实冒充提前预测。如果始终活不到第三夜，报告未覆盖，先定位第二夜，不无限换对手刷样本。
- 发现启动/异常、确定重复缺陷或达到场次上限即提交结果，不为填满所有观察项拖延反馈。

## A. 核心：毁墙补建是否正确且有价值

每场按固定墙坐标列短表：曾真实存在的回合/HP→被毁首次确认→次日recoveryTargets→施工者/采石/建造→次轮墙出现及反馈。至少核对R70、R130/131、第二天关键施工点、R200/201、R260/261（存在才记录）。

- 字段：decision.economyPlanning.fortification中的targets、completed、observedFixedWalls、missingConfirmedWalls、recoveryTargets、attemptsToday、batchTargets、builderId、phase、skipReason；联看controlledRoles、ourStructures、commands和次轮actionFeedback。
- 足迹最多6个固定目标、每批最多2墙；每日有界施工机会。累计建造次数可跨日超过6，不能据此误报；实际固定目标集合不能无限增加。
- 只有曾真实观测存在、跨日白天缺失的墙恢复。同日缺失、现存低血墙、未知build反馈、明确失败或永久不安全目标不自动重开。低血墙沿已有WallFixer逻辑，不能用build覆盖回血。
- 未补建先区分缺石料、路线/占格、安全、持券/资金保护、施工者缺席或回岗窗口，不能只因没build就报bug。
- 评估副作用：工人是否挤占原本有效升级链，是否未按时回岗，是否堵炮位/通路；记录实际操作武器的controller与坐标，距离须为1。施工者死亡/复活是否能恢复分工。
- 对比第二夜基地/墙/炮手/武器损失与有效攻击窗口。旧603316/603444/603631只作参考，地图、对手、波次不同不能直接把胜负改善归因补墙。

## B. 核心：波次与预测能否可信取证

机器人规则为全图可见、夜首轮统一出现、数量逐日增加、召唤叠加。只按targetTeam为我方统计，不按基地附近范围；按类型和同夜ID去重，跨夜允许ID复用。内网独立统计全量日志，与decision.defensePressure比较。

- 每夜给day、首末观测轮、各类数量、完整性、是否有重复ID/未知targetTeam/512-ID截断。夜首轮正常R71/R201/R331；迟到或缺夜按实际完整性标记，不把击杀后活体减少当波次减少。
- 预测字段forecast/forecastHistory：targetDay、basisDays、sampleCount、method、counts、growth、uncertainTypes、actualCounts、absoluteError。白天看当夜，夜间看下一夜；从目标夜到来前的日志取预测，再与实测比，防止事后回填。
- 相邻两个完整夜才按类型最新量+非负增量估计；下降类型标uncertainTypes，参考取最近两夜较大值。首夜/partial/缺夜标unknown，不是下界或上界保证。natural与召唤不可拆分就保留combined_observed，不猜增长公式。
- 记录各类型预测偏差及方向；新BOSS/大型机器人出现单列，别只看总数。核对previousNightLoss与实际战损；缺夜首轮或黎明观测必须保留不完整状态，不拿修复/复活后状态冒充夜末。
- assessment.reasons中的known_deficit/rising_pressure/unknown/no_observed_deficit只表示可观察准备缺口与压力参考，不是保证活下来。当前版本不因压力增长自动买券/道具，未发生自动采购不算bug。

## C. 关键回归与顺带证据

全量扫描我方turn事件的首末回合/数量/唯一回合/缺口/重复、非零errors、HTTP/traceback/JSON异常及服务端处理耗时。数据源不可得明示；服务端不是裁判端到端。正常任务/经济/夜间操控有代表链即可，不为重新验证已通过S1强刷所有场景。

任务效率、新闻invalid_candidate为后续批次，本版未改求解器；可保留可执行建议，不把旧误报重新当缺陷。背包是controlledRoles.items[i].backpack物品计数字典，无backpack.items。LLM命令在final-only后被拒是截止保护。task_detail/news_detail独立event；按PK/team/round/build及日志顺序关联，turn不含requestFingerprint。反馈通常对应前一回合动作。

## D. 反馈到GitHub

- 本轮S2恢复/波次结果主要追加#11；经济副作用追加#9，任务真实回归追加#10。独立新根因可新建简短issue并关联，不复制旧长报告；不自动关闭issue。
- 每条约1500中文字符内，只写增量：实际完整SHA/build/包差异、PK/我方阵营/对手、观察窗口、通过/失败/未覆盖/证据不足、关键回合链、事实与推断分开、下一步建议。给最终覆盖矩阵：毁墙补建、资金/回岗保护、第二夜、完整波次、第三夜预测校验、异常回归。
- 原始题面/LLM交互/工具输出/答案/认证信息只留内网，用GLM 5.2分析；issue只写允许披露的摘要和内部证据代号。没有观察到的项不写通过。
- 记录上传、排队/对战、日志可分析、反馈各时间点和人工耗时（含时区）；未知标未知，日志首末跨度不是完整周转时间。
