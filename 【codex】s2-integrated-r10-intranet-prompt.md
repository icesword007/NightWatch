# r10 统一送测说明：回防、基地应急升级、压力旁路

本页只解释版本与验收字段，不独立授权开场。唯一当前执行入口为 https://github.com/icesword007/NightWatch/issues/12 ，每次刷新完整正文和updatedAt；完整SHA、场数和停止条件以该正文为准。构建标识为 `nightwatch-s2-integrated-r10`。

## 增量

- R256回防：仅炮手寻路在A*同等总代价下优先深入，扩展预算保持256；旧筹资计划无当前可执行候选时不再预留角色和炮位。新增turn.decision.defensePlanning，区分postStatus、assignedPosts/requiredPosts、routeResults、fundingReservedRoles/Posts、unsafeDayWork、returnCandidates。
- R254基地应急升级：有条件白天储备对应等级基地券，夜间按受损与到位时间余量决定是否使用；保护即时自救、任务、火力与原经济工作。失败/死亡/等级变化/新局会清理储备。新增turn.decision.baseUpgrade。首批不增加主动墙升级，不夜间跨图买券，不保证同轮升级与攻击的结算顺序。
- R253只读旁路：pressureShadow.prediction.algorithmVersion为shadow_v2，首次观测同基地同等级HP下降即可记录当前夜event_observed、firstDamageRound及baselineResult；历史基线与正式assessment分开。正式未知不能写成安全，缺天亮不妨碍正例漏报评分；不参与动作、预算或进攻决策。

## 首场验收重点

1. 核完整源码SHA、实际上传tar.gz SHA256、包内源码/BUILD_ID与平台启用链；包名前缀或zipball hash不能代替内容核验。保留现有未提交文件，不自行改程序/依赖/脚本。
2. 回防P0：重点每夜前至少40轮至入夜后15轮，逐角色位置/HP/计划/候选/实际命令/反馈/到岗时间、炮塔攻击/冷却。经济停工不等于必须立即返岗；用实际可达路线与时限判断。空分配、path expansion_limit、deadline、旧筹资占用分别归因，fortification.gunnerStands不是完整岗位分配。
3. 基地升级：用库存、券等级、购买/使用命令及反馈→次轮同基地ID的等级/HP核实。记录是否因备券干扰普通工作、到岗或即时救援。未自然触发只记未覆盖，不擅自重跑刷触发，不以末帧HP推断结算顺序。
4. 旁路：检查当前夜事件是否及时输出、双方证据是否独立，历史baseline与assessment/currentPressure分开解释；同等级HP下降和升级回血分开。提前终局保留已观察正例；不伪造完整夜负例或宣称准确率提升。
5. 原有T60/T46/T48只观察自然触发，保留任务实例分母、真实奖励账与不完整语义证据，不扩任务或宝藏执行。时间字段按turn.timingMs实际路径读，不以末帧耗时代表全场。

## 证据边界

本地636项功能回归和受控多轮反例已通过，不代表PK665935原局根因唯一确定，更不代表S2通过。实战日志、完整题面、真实答案和凭据保留内网；issue仅脱敏增量。#13经济/性能/包身份，#14任务，#15回防/基地/旁路；不改/关闭#12或其他issue，不删除历史证据。
