# r9 统一内网验证参考

实际执行版本、完整SHA、场次及停止条件始终以最新 https://github.com/icesword007/NightWatch/issues/12 为准，每次开工刷新正文和updatedAt。构建 `nightwatch-s2-integrated-r9`；包含r8三项T48只读增量及R248双方压力旁路。历史r7/r8提示不自动执行。

## 统一范围

- T48缺品采购路线、逐字段推导审计、本轮金币/角色占用：按[原r8字段验收参考](【codex】s2-integrated-r8-intranet-prompt.md)的增量及验收章节核对，旧文档的r8版本限制不适用本批。仍actionEnabled=false，路线可达/证据可追溯不代表语义真值或允许献祭。
- r9 `decision.pressureShadow`：current为本轮双方观察，prediction为日间预测/夜间冻结，verification为最近结束夜次核验，recentNights最多3夜；session/team及elapsedMs位于旁路根。只观察，不参与行动、预算、召唤或防守决策。
- 完整Turn机器人按targetTeam统计，基地距离为到2×2占地的切比雪夫距离，非路径时间。墙/机器人消失不是摧毁/击杀，天亮清场不算击杀。ID上限512只影响有界历史追踪，完整当前数量仍统计；超限及缺帧必须保留unknown。
- 首版heuristic_v1只延续上一完整夜基地是否受损：elevated/low/unknown，不是校准概率或召唤收益。白天防御变化/观测缺口降unknown；无日间预测为unavailable；关键墙风险始终unknown。
- 按PK/半场/session/team/night/issuedRound关联冻结预测与实际。入夜后不能反写，夜后命中/误报/漏报/正确负例须按完整性解释；提前终局无下一日核验时保留未核验，不伪造负例。人工日间分析预测必须先记录时间与目标夜，事后分析不能冒充事前预测。

## 回归与报告

继续核T60任务归属/工具回注/裁判反馈、T46自然应急采购、三塔/回防/经济、decide和serverWriteComplete耗时；旁路elapsedMs不含输入解析和完整日志写出。

先验证提取脚本路径和类型：task_detail.text.*.value、submittedAnswers.items[*].text，turn.task.tool与turn.decision；shops.vendor/weapon为直接映射。完整数据可获取，不用分析截断后的16项推断总体。每轮130，前70白天后60夜间。提交不等于任务成功，收入支出须按动作反馈和下一帧闭合。无法闭合写未知，不猜补项。

每场核完整SHA、包SHA256、文件清单/大小、实际上传启用和首帧buildId。排除缓存/.DS_Store/凭据，不覆盖本地修改，不改参赛源码/依赖。错版、协议异常、持续无有效动作或明显回防/预算退化时停止下一场并报告。胜场不等于S2验收通过。

#13经济/T48路线资金/耗时；#14任务/T48证据审计；#15版本/防守/双方压力预测与核验。各给短增量及证据位置，完整日志/题面/答案/Token只留内网；不关闭issue，不自行追加场次。
