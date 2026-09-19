# r8增量验证参考（必须以最新#12放行为准）

构建nightwatch-s2-integrated-r8。r7在途验证不切版、不混局；先完成/因既定停止条件结束并回传。重新读取#12正文及updatedAt，只有#12明确允许r8完整SHA部署时才能切换。禁止改源码、覆盖本地改动、reset或刷场。

## 增量及验收

- T48缺品采购路线：读取procurement.route及routeEvaluated，而非顶层已有用品route。核到店、按不同品种各1次购买、到祭坛、窗口等待、献祭1轮、返炮位及2轮余量。单品多件可一次买，采购与献祭不能同轮。已有/缺品路线共享4候选、256扩展、原deadline内10ms，截断明确unknown。无自然候选就记未覆盖。
- T48证据审计：现有新闻调用可返回每字段derivation（kind/explanation/unresolved/timeBasis），旧响应兼容。decision.evidenceAudit只有固定状态及来源计数；同文不同ID只算一个文本来源，更多来源不等于独立证明。first_observed/relative缺锚点时anchor_unknown；模型自称absolute也仅absolute_anchor_unverified。记录自然模型返回新字段的比例、解析拒绝、提示长度/截断，不因direct标签宣称真值。
- T48本轮承诺：procurement.currentAllocation限定current_response_only。核真实分配的金币预留与余额，拒绝候选不扣款、未到账收益不计入；开拓者动作按actor计，炮台owner不是炮手；任务/筹资预留另列。fallback或上下文错配应unknown，不伪装无冲突。

这三项均只读；actionEnabled=false、pending_validation、semanticValidationEvaluated=false、spendingBudgetEvaluated=false保持。没有本轮冲突、路线可达、引文可追溯都不等于应采购/献祭。未来完整预算、语义真值、未来占格仍未确认。不制造候选、金币或用品，不主动触发真实献祭。

## 版本、回归与报告

完整40位SHA、完整64位包SHA256、清单/大小、实际上传启用、首个可取得日志buildId闭合。排除缓存、.DS_Store和凭据；错版立即停。场次/半场额度及停止条件只看#12，按PK/阵营/对手/范围独立记录，不把胜局当S2通过。

回归r7 T60任务归属/真实执行与裁判反馈、T46自然采购库存使用、三塔/回防/多日失血、经济到账/路径预算及decide/serverWriteComplete分布。自然缺字段明确未知，不向旧版本追索新字段。详情及真实题面/答案/认证留内网；issue仅短脱敏增量。

#15统一版本身份和防守，#14任务与T48引用/推导，#13资金/路线/性能，互相引用不复制整份报告。区分已核实/推导/证据不足/未覆盖。不编辑关闭#12或缺陷，不自动进入下一版本。
