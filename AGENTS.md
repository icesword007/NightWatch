# NightWatch 项目约定

- `CoreGeek/`：既有 vanilla-v2 演示程序，共7个受版本控制文件（main3.py、pyproject.toml及src/agent下5个Python文件）；其中main3.py也与项目上级目录的原始赛方SDK不同。2026-09-13已获准按A批S0计划修改并本地验证，不将其协议定义和策略视为赛方规则。
- 原始赛方SDK位于外层项目的 `CoreGeek/main3.py`，不在此仓库内。进入后续实现阶段前，任何复用需依据正式任务书和接口文档重新核验。
- 仓库根目录对应本地 `code/NightWatch/`，不收录上级目录的研究文档。
- 不提交 `.DS_Store`、密钥、token、密码或本地环境文件。
- 新目录先约定用途与命名；删除文件或目录前须用户确认。
- 当前上传验证以文件清单和内容一致性为准，不代表参赛程序已通过运行验证。


## A批编码范围与验证

- 依据外层design/【codex】implementation-plan.md批次A，仅实施S0；主会话验收后再安排下一批。
- CoreGeek/src/agent/存放参赛模块；main3.py为入口，run.sh为拟新增启动脚本，上传包根目录须核验。
- CoreGeek/tests/存放test_*.py；CoreGeek/tests/fixtures/存放最小派生JSON与带【codex】前缀的来源说明。真实日志不直接提交，派生样例逐项说明语法修正，不修改赛方原始材料。
- 不放临时材料；测试临时文件使用系统临时目录。任何文件删除或目录清理先获用户确认。
- 在CoreGeek目录先核对Python版本，再执行：PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v；批末同时运行git diff --check。尚无测试时记录基线为空，不声称既有测试通过。
- 禁止安装全局依赖；当前优先标准库。开发会话直接使用现有checkout，主会话不并发编辑源码；已有AGENTS.md未提交修改须保留。
- 本批已由主会话完成本地验收，用户于2026-09-13授权主会话提交并推送代码与内网验证手册；内网验收前不关闭issue。真实平台验收由内网安排，不能用本地响应代替。


## 内网Agent执行规则

- 首先阅读根目录[内网验证手册](【codex】intranet-validation.md)，该手册为内网S1候选接管入口，不需要外层design或原始SDK即可按步骤验证。
- 内网职责仅为拉取明确版本、打包上传、对战、分析及在本仓库issue反馈；不自行改源码、策略、依赖或启动脚本。平台适配差异先报告并记录，不能用原SHA代表改过的程序。
- 根目录生成Markdown统一使用【codex】前缀；本AGENTS.md固定文件名例外。测试来源说明引用外部材料仅为历史来源，不是内网执行依赖。
- 原始日志留内网，只向issue提交符合内网要求的最小证据摘要；不得上传密钥、token、任务私密内容或完整日志。
- 已跟踪文件有本地修改时不得覆盖或回滚，应报告差异。远端issue是反馈证据，不自动授权本地修改范围。


## B批与取证修订（用户2026-09-13授权）

- 按外层implementation-plan.md批次B实现公共状态、动作资源约束、寻路边界与主动截止；同时补S0有限角色/双方基地状态、UTC时间日志与对应内网手册。A批S0未决验收并入下一次版本内网验证，不单独安排对战。
- 可新增src/agent/state.py、actions.py及tests/test_state.py、test_actions.py、test_grid.py；扩展现有protocol/brain/server/grid与测试。保持单一开发会话，不启动C/D策略或仿真。
- 实际HTTP决策链必须使用本批公共能力，不能只增加无人调用的框架。未知裁判规则保留边界；不要为未启用策略虚构完整模拟。
- 暂不提交或推送，主会话验收后统一安排下一版本。保持原有缓存与未跟踪文件，不删除、不全量暂存。


## C批经济建造与首夜防守（用户2026-09-13授权）

- 在已通过本地验收的B批工作区上继续C批，不覆盖/回退B批未提交文件。新增src/agent/economy.py、defense.py及对应test_economy.py、test_defense.py、test_team_turn.py；可修改既有模块与内网手册。
- 范围为采集出售采购使用、武器/基地升级、Medicine/WallFixer、三类武器基本操控、炮手到岗与目标稳定性、基地被毁后继续合法行动。真实任务求解、主动宝藏、Bomb/Dizzy、召唤和直接进攻不在本批。
- 规则冲突须标来源与影响；程序日志记录试验格/动作反馈供下一合并版本核验，不能把自测假设称平台事实。不能默认围墙有效或让失败建造无限重试。
- 本地验证命令沿用B批；主会话验收前不commit/push、不触发内网对战，缓存保持原状。


## D批任务闭环与S1统一交付（用户2026-09-13授权）

- 原Sol + medium开发会话继续在B+C工作区实现D批，新增src/agent/tasks.py及tests/test_tasks.py，可扩展既有模块和联合测试。任务类型、工具调用与反馈字段以外层官方文档为来源；无真实题目时只能认定管线完成。
- 包含任务领取/求解/提交/离点生命周期、平台prompt与executeCmd、新闻线索记忆及任务与防守联合仲裁。不扩展S2宝藏、召唤或仿真。不得在本机执行来自任务或LLM的任意命令；平台执行命令的生成边界和验证方式应明确。
- 开发会话不提交推送；主会话验收后统一提交并推送S1候选及自包含内网手册，内网验证由用户安排。S1平台验收需双方首夜与真实任务完整通过证据，不能仅看金币变化。保留issue #4未决项。


## S1实战修复（2026-09-13，R057）

- 用户授权在be36ab6基础上定位并修复任务收敛、交替工具循环、全队墙试验限额，以及有证据的经济/防守缺口；原开发会话独占源码。沿用既有目录与验证命令。
- 内网补充分析并行进行；未证实原因不得写成已修复，不扩展S2/仿真。开发会话不提交推送，主会话验收后统一安排复测版本。


## S1联合修复（R061）

- 在本地6a11a94上修复提前出售、采购使用与炮位到岗期限、任务提交与必要第三炮位协调，并加入有限命令/结果指纹与决策原因。沿用既有源码/测试目录；规则先补软件设计。
- 诊断必须与当次响应绑定，不读共享可变状态制造串回合记录，不把内部诊断字段泄露到裁判响应，不记录敏感正文。开发会话不commit/push；主会话验收后统一交付。


## S1采矿与联合筹资（R076，用户授权执行）

- 依据外层design/【codex】economy-improvement-plan.md，在a21aeb4上改进有界稳定选矿与两工人共享金币筹资；沿现有源码/测试目录，可修改economy/brain/state与必要诊断及内网手册。不扩展S2/仿真，不改任务求解、攻击协议或低血撤岗。
- 原Sol+medium在现有checkout独占源码；先失败反例后实现，保留守岗与单人资金链，验证完整兑现和双方回岗、真实到账、候选拒绝及失效撤销。沿用全量测试、diff检查与启动脚本语法检查。
- 开发会话不commit/push，不触发对战；主会话验收后统一安排。保留缓存/未跟踪文件。


## S1资金链正确性与性能（R082，用户授权）

- 基于1929f76，执行外层design/【codex】funding-correctness-performance-plan.md：请求内路线复用、有界规划与交付预算、阵营等价与持券使用链修复、有限诊断。沿现有源码与测试目录，可改economy/brain/grid/state/actions/server及相关测试手册，不改投资偏好、攻击协议、LLM或低血撤岗。
- 原Sol+medium独占源码，先真实DecisionEngine反例后修复，记录寻路次数与耗时对照、完整回归；主会话独立验收。不得跨请求缓存污染或提前消费预测金币，不以增大超时解决搜索问题。
- 开发任务不commit/push、不发issue或启动对战；保留缓存/未跟踪文件。


## S1-r7路径与任务有界修订（R091，用户授权）

- 基于fabcf23，执行外层design/【codex】s1-r7-repair-plan.md。原Sol+medium独占源码，沿现有源码/测试目录。资金邻格搜索局部失败与兑现诊断、任务输入定位约定和有证据的截止边界允许修改；防守只读定位，确定缺陷先交主会话核实。
- 保持预算、合法性、完整结果缓存边界；不盲调投资/布局/低血撤岗，不扩展S2。全量测试、diff和启动脚本语法检查沿用既有规范。开发会话不commit/push或发起对战，主会话独立验收。


## S1主动围墙防线（R095，用户授权）

- 在本地r7提交1cb41a3上实施外层design/【codex】s1-fortification-plan.md，统一候选r8包含r7。原Sol独占源码，可新增src/agent/fortification.py与tests/test_fortification.py，沿既有目录修改brain/economy/defense/state及相关测试/手册。
- 用有界主动采石/定向多墙施工替代旧单墙试验，联合检查炮手站位与通路、期限，保留夜间守岗与已启动fund/持券；不做复杂仿真、主动进攻、无证据低血撤岗或任务答案硬编码。主会话验收前不commit/push，不发起对战。


### R095任务详细取证补充

- 用户确认平台只能下载程序日志；主会话将任务专用详细日志纳入本批定位能力，依据外层design/【codex】s1-task-detail-logging-plan.md。默认任务相关event=task_detail记录白名单交互正文及截断/实例关联，仅留内网，原摘要与裁判响应仍不含正文扩展。此为原“日志不记录正文”规则的限定例外；无关凭据/headers/环境/整包请求禁止记录。
- 可修改server及必要请求级trace/测试，必须有界、不串并发、响应后记录。内网issue仍只传脱敏分析；不得将真实任务正文入仓库。


## S1-r9经济兑现与任务效率（R100，用户授权）

- 基于58254f9执行外层design/【codex】s1-r9-repair-plan.md。原Sol+medium独占源码；沿既有src/agent与tests目录，允许修改economy/brain/tasks/state/fortification/server及对应测试和内网手册，不新增目录。
- 修复旧采矿计划跳过单人兑现；压缩任务交互、保留重要证据、复用经验证的环境线索并按实际轮数收尾；围墙仅优化小批量备料和暂时不安全目标的有界复查。保持总预算、夜间岗位、真实资源、请求隔离和日志边界。
- 不硬编码真实题目路径/答案，不从命令结果随意拼答案，不为追求提交数跳过证据；不取消防守截止，不扩展S2，不重构整个调度器。证据不足的策略参数先向主会话报告。
- 开发任务不commit/push、不发issue或对战；主会话独立验收后统一提交，推送另按用户指令。内网补证与开发并行，未取得证据不能写成已验证。


## Issue归档迁移约定（2026-09-15，用户授权）

- 用户批准因消息体过长将旧open issue #4/#5/#6/#8关闭归档；这是迁移，不表示缺陷修复或S1验收通过。关闭前保存全部正文与评论快照，远端保留旧内容，并在关闭评论中说明原因；关闭原因使用not_planned以避免标为已解决。
- 后续由内网Agent对真实剩余问题建立独立issue：经济兑现、任务完整求解、炮手防线与defender首夜分别跟踪，按实际根因再拆分。每单一项明确验收目标，链接原issue，不复制旧全文；r8实证与r9本地修复状态分开。
- 团队建议正文约2000中文字符以内、单条增量评论约1500字符以内（不是平台硬限制）。正文只保留当前结论/验收，评论追加本次增量，不同时重复贴正文与评论。历史原文留旧issue，完整日志留内网；超长证据按独立子问题拆分。
- 新issue在实际复测通过后关闭；不得将此次迁移例外扩大成“未通过也可按完成关闭”。新编号由内网创建后回传映射。


## S1-r10修订及S2本地合并（R118，用户授权）

- 原Sol+medium本批唯一源码工作区切回`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch`，main基准5c22d6155cb61500232b6412abada04536f9d3bc。不得沿用S2目录；S2分支f369b58保持不动。
- 实施外层design/【codex】s1-r10-and-s2-integration-plan.md：修复任务路径线索污染、接取前回防窗口，针对及时可达的后方炮位被短路径压过进行反例定位后最小修订。经济持券/修墙问题等待内网实证，不盲调参数。
- 主会话独立验收并提交S1修复后，用户已明确授权将codex/s2-day-recovery合入main。本地合并、必要冲突修复与完整验证均在授权范围；不推送，不删除工作树/分支，不回滚。开发任务仍不自行commit/merge/push，集成由主会话统一安排。


## S2跨日恢复专用工作树（R107，用户授权，覆盖旧阶段范围限制）

- 唯一开发目录：`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch-s2-recovery`；唯一开发分支：`codex/s2-day-recovery`；基准`5c22d6155cb61500232b6412abada04536f9d3bc`。每次执行命令显式指定该目录或其CoreGeek，先核对pwd/branch/HEAD。
- 任务虽仍绑定外层CCN-Comp，请勿沿用旧源码路径`code/NightWatch`。该主工作区仅为r9验收/修复保留，本S2任务禁止修改它。
- 仅实施外层`design/【codex】s2-day-recovery-plan.md`：先盘点已有能力，再用真实连续回合反例修复跨日释放/恢复、角色复活和次日施工缺口。不得扩展宝藏/投资/直接对抗或宣称S1通过。
- 共享todo/review保留在外层design，可按原规范追加；不得复制平行清单。主会话已跑新工作树基线238项通过。
- 不commit/push/merge/rebase/cherry-pick/reset，不切main，不删除分支/目录。完成后留变更给主会话独立验收；紧急S1修复由主会话另行明确调度，不自动回旧目录。


## S2新闻证据基础（R110，用户授权继续选批开发）

- 在既有S2工作树codex/s2-day-recovery、e874497之后继续，范围扩展为新闻/传闻的来源、当前session去重、跨日与跨局有效性及有界诊断；依据外层design/【codex】s2-news-evidence-plan.md。本段覆盖此前“仅跨日恢复”的范围限制。
- 原Sol+medium独占S2源码，main送测树不得修改；不增加新闻LLM调用、价格预测交易、宝藏采购/献祭/改道，不改变S1任务/防守仲裁。不新建目录，不引入依赖；使用既有模块和测试目录。
- 开发任务不提交、推送、合并、切分支或删除；交付由主会话验收。日志只在内网保留有界新闻证据，不记录整包请求、凭据或无关正文；issue只反馈脱敏摘要。


## S2新闻LLM候选解读（R113，用户授权继续开发）

- 在S2树codex/s2-day-recovery、a1eae92之后执行design/【codex】s2-news-interpretation-plan.md；覆盖R110禁止新闻LLM的阶段限制，仅允许有界普通额度新闻解读与带证据的候选记忆。
- 可新增CoreGeek/src/agent/intelligence.py和CoreGeek/tests/test_intelligence.py，沿现有目录修改state/brain及必要日志/测试；不新增目录，不改SDK或依赖。普通新闻分析每天至多2次，任务求解优先，禁止混用工具结果。
- 仍禁止宝藏采购/献祭/改道、囤矿、投资或防守调参；LLM解释不是平台事实，不执行其命令。主main保持r9；开发任务不提交推送/合并/切分支/删除，主会话独立验收。


## S1-r10与S2三批集成开发（R122，用户授权，覆盖旧工作树指向）

- 当前main集成唯一工作区为`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch`；正在将S1-r10提交`59918afbaf9e4825b1e21b58425b2c95e1016a2b`与S2分支`codex/s2-day-recovery`/`f369b58bf234afd962044c14e79dfd3f6ed6eca0`语义合并。
- 旧`code/NightWatch-s2-recovery`工作树只读保留，不再是当前命令或编码入口，不修改、删除或清理。
- 冲突须同时保留S1任务路径/接取窗口/炮位修订与S2跨日恢复/新闻证据/有界LLM候选解读，不允许整文件选单边。新闻候选仍不驱动角色动作，不扩展宝藏、投资或防守参数。
- 开发会话可`git add`明确解决的冲突/集成文件，但不commit/push/rebase/cherry-pick/reset/abort/切分支/删除；主会话最终独立验收并提交merge。


## R123 当前集成版本约定

- 主会话已独立验证S1-r10与S2三批合并结果，288项完整测试通过，构建nightwatch-r10-s2；本次创建本地merge提交，不推送。当前后续工作入口为本main树，旧S2工作树与分支只读保留。
- R122中的冲突解决操作仅适用于当次集成，不授权后续开发会话自行合并或提交；新源码任务须主会话另行下发。S1/S2实战验收尚未通过，保持issue开放。


## R124 本轮送测授权（2026-09-15）

- 用户已授权推送当前main候选版并由内网执行验证。本条覆盖R123的当轮“不推送”限制，不授权其他开发、回滚或清理。
- 内网以本仓库【codex】r10-intranet-run-prompt.md及【codex】intranet-validation.md执行送测；前者规定本轮优先级。不得使用外网机器绝对路径作为内网工作目录，使用实际克隆位置。


## R128 当前S2开发授权（覆盖历史main与旧S2目录指向）

- 本批唯一源码目录/分支：code/NightWatch-s2-defense / codex/s2-defense-recovery，基准7debfef73494be215ee7032c70792f3921f5a884。所有命令显式workdir到该树。main和NightWatch-s2-recovery只读，不混改。
- 原Sol+medium按外层design/【codex】s2-defense-recovery-plan.md实施T38：固定目标毁墙跨日补建、逐夜波次观察/有界预测及资源缺口；可在CoreGeek/src/agent及tests现有目录加必要模块/测试，不另建目录，不改SDK/环境/依赖。
- 不commit/push/merge/reset/切分支/删除；主会话独立验收。旧规则禁止补建的阶段限制由本批明确授权覆盖，仍保留6格足迹、2墙批次、逐目标每天有界尝试、现有失败/安全/持券/回岗边界。新预测不是裁判增长公式，也不授权无限投资或强制购券。


## R131 当前S2首批送测（2026-09-16用户授权）

- 用户授权将已验收S2首批68e24f7快进合入main并推送，本轮只送测，不启动下一批开发。覆盖R128的当轮禁止合并/推送及唯一开发树限制；原S2工作树与分支保留。
- 内网使用实际克隆目录，以本仓库【codex】s2-defense-r1-intranet-prompt.md为本轮优先级，配合【codex】intranet-validation.md。不执行外网绝对目录指令，不照搬旧r10提示重跑完整S1验收。
- S1已在7debfef完成功能验收（双阵营首夜、工程完整成功、失败恢复、726回合全量日志无已发现异常）；本版只作关键回归。裁判端到端时延不可得、本版部分提交未覆盖的边界保留。S2整体十天验收尚未完成。


## R132 当前S2任务效率开发授权

本工作树唯一开发入口code/NightWatch-s2-tasks，分支codex/s2-task-efficiency，基准0bce507。覆盖R131暂停与历史目录指向。按外层design/【codex】s2-task-efficiency-plan.md由原Sol+medium开发工具周期预算、本任务有界失败证据、求解指引和JSON边界。main与旧S2树只读；不变更截止/路径安全/SDK/经济/防守/新闻，不新增目录或依赖。开发任务不commit/merge/push/reset/切分支/删除；主会话验收后本地提交。本轮不推送，内网仍测试nightwatch-s2-defense-r1。


## R134 本批本地验收

任务效率首批已由主会话独立314项回归与4/5/6/20剩余回合连续工具/提交实验验证；本次允许主会话在codex/s2-task-efficiency创建本地提交。main/origin送测版0bce507保持，不合并、不推送。实际任务成功率收益待内网，S2整体未验收。


## R136 当前整体防线与三火箭开发授权

本树唯一开发入口code/NightWatch-s2-layout，分支codex/s2-rocket-layout，基准baab47e；覆盖历史工作目录和6格墙/混合阵容阶段限制。原Sol+medium按外层design/【codex】s2-rocket-layout-plan.md开发整体墙/塔/炮手/通道布局、连续分段施工、三火箭及紧急/群体收益选点。可在现有agent/tests目录新增layout模块及测试，不另建目录，不改SDK/依赖/环境。其他4树只读，不改任务/新闻等既有能力，不降级覆盖旧塔。开发不commit/merge/push/reset/切分支/删除或向其他任务补发消息；主会话验收后可本地提交。本轮不合main/推送，交付R137，验收R138保留。


## R138 本批本地验收

主会话独立343项通过及残血L2/L3合法开火、动态塔位不误建额外反例通过。本次允许主会话在codex/s2-rocket-layout创建本地提交；main0bce507仍为内网送测版，不合并/推送。双向完整施工测试预置两塔及材料，只验证后续施工/回岗流程，不证明真实首日资源必然足够。14格布局、后排三火箭与溅射收益实战效果待验证，S2整体验收未完成。


## R139 当前增量送测授权与入口

用户授权将94ce77e（包含baab47e任务效率）快进main并推送；覆盖R138的当轮不合并/推送限制。本轮不启动其他S2开发。内网本轮优先读取【codex】s2-layout-r1-intranet-prompt.md及【codex】intranet-validation.md，构建nightwatch-s2-layout-r1，记录实际上传完整HEAD。旧s2-defense-r1提示只适用0bce507旧局，旧局结果与新版严格分开；不照搬6墙/混合阵容口径。不使用外网绝对目录作为内网工作路径，旧工作树与缓存保留。


## R141 S2投资目标与计划稳定性（用户授权继续开发）

新增`code/NightWatch-s2-investment/`为同仓库linked worktree，分支`codex/s2-investment-targets`，基准e084aed1c492bd51f0f43bb0f2eaa90d7ef0005c；仅源码、测试和手册，不放临时材料，删除须授权。原Sol+medium独占此树，按design/【codex】s2-investment-targets-plan.md实施。main正在送测，旧五树只读。本批只改现有投资物品的可执行目标选择与已启动目标稳定性，不调采购优先级/预算/波次公式，不新增墙升级采购、新闻宝藏或任务能力。开发不commit/merge/push/reset/切分支/删除，主会话验收后可本地提交；本轮不合main或推送。


## R143 本批本地验收

主会话独立360项回归及未绑定/封死/临夜超期额外反例通过，允许主会话在codex/s2-investment-targets本地提交。本轮不合main或推送；main e084aed仍在内网验证。实际投资收益与S2整体待实战，旧工作树和缓存保留。


## R145 S2多工人投资协同（用户授权继续开发）

继续使用`code/NightWatch-s2-investment/`与`codex/s2-investment-targets`，基准d1e0183ee88daed501cfe6e3818584333acfacdc，不新增工作树。原Sol+medium独占源码，按design/【codex】s2-investment-targets-plan.md的R145续批实施同目标投资冲突协调；覆盖R141不做跨工人目标排他的本批限制。main e084aed及其余旧树只读。仅同目标重复采购/使用与在途分工释放，不改投资品种/额度/储备阈值、布局/任务/新闻/攻击/SDK，不新增依赖目录。开发不commit/merge/push/reset/切分支/删除，主会话验收后可本地提交，本轮不合main/推送。交付R146、验收R147。


## R147 投资协同本地验收

主会话独立367项回归、6回合双目标购券使用链和owner释放矩阵通过，允许主会话在本隔离分支本地提交。本轮不合main/推送；main e084aed继续作为内网版本。


## R151 S2围墙升级闭环（用户授权继续S2开发）

原Sol+medium沿code/NightWatch-s2-investment/codex/s2-investment-targets，基准f72170214a2359b931f02d79d7bbd4a7a3c0ef35独占开发；main f721702与远端e084aed均保持。按design/【codex】s2-wall-upgrade-plan.md补围墙升级采购与兑现，覆盖此前不新增WallUpgrade采购的批次限制。保留现有物品相对优先级、合法路线/回岗/同轮预算/目标owner，不调布局、任务、波次公式、SDK或全局预算。不新增目录依赖、不改其他工作树。开发不commit/merge/push/reset/删除；主会话独立验收后可本地提交，本轮不自动合main/推送。交付R152，验收R153；内网待补证不是本批故障依据。


## R153 围墙升级本地验收

主会话独立374项含HTTP回归及共享20金/全墙L3/敌基地消失固定蓝图/41×32双工人14墙边界通过。允许主会话本地提交本批；不合main、不推送。main f721702与内网e084aed保持，旧树保留，真实收益待验证。


## R155 S2任务成功率首批修订（用户授权）

新增code/NightWatch-s2-task-success/为同仓库linked worktree，分支codex/s2-task-success，基准main/f72170214a2359b931f02d79d7bbd4a7a3c0ef35；仅同仓库源码/测试/手册，沿原命名，无临时材料，清理须授权。原Sol+medium独占，按design/【codex】s2-task-success-plan.md实施确定性有界任务入口读取及有界收尾纠正。覆盖旧批次不自动生成沙箱读取命令的范围限制，仅允许本文列出的只读入口操作；不自动修改LLM shell、不执行本地原始SDK示例。

main f721702、投资分支776e8ac（墙升级）及其他旧树只读；本批不混入墙升级，不改经济/布局/攻击/新闻/SDK/依赖/环境/截止期限。可在现有agent与tests目录新增task_input.py及对应测试，不新增目录。开发不commit/merge/push/reset/切分支/删除，主会话独立验收后可本地提交。本轮不自动合main/推送。交付R156，验收R157；仅泛化行为修订，不声称原PK故障已全部复现或实战成功率已提升。


## R157 任务入口与收尾本地验收

主会话独立386项含16HTTP通过，真实生成读取命令→下轮正文求解→可靠部分提交连续链、否定/引用入口不触发、文件名字面量及慢单目录超时不报成功反例通过。允许主会话本地提交；main f721702、墙升级776e8ac保持，不合并/推送。无内层JSON强制、无自动答案、无任务/回防截止放宽，实战成功率待验证。


## R159 S2统一集成送测（当前执行规则）

用户已明确授权将投资目标/多人协同、围墙升级776e8ac与任务成功率5810d6d合入main，主会话处理集成冲突、完整验证、提交并推送。本条覆盖上述历史批次不合并/不推送限制。当前构建nightwatch-s2-integrated-r1，内网以【codex】s2-integrated-r1-intranet-prompt.md和通用手册当前轮次要求执行。停止旧版补证；旧版本资料仅历史参考。旧树/缓存/日志保留，不擅改策略，不以本地通过或推送作为实战验收，不自动关闭issue。


## R160 S2已有应急道具使用（用户授权继续S2）

新增code/NightWatch-s2-emergency/为同仓库linked worktree，仅同仓库源码/测试/手册，不放临时材料，清理仍须授权。分支codex/s2-held-emergency，基准e7ff4c563b51a44e047721eda6ae30700477dd77；main正在内网验证，保持不动。原Sol+medium唯一开发者，按design/【codex】s2-held-emergency-plan.md实现夜间已有Bomb/DizzyWeapon的保守原地救援。覆盖旧阶段禁止应急道具使用限制；不新增采购/筹资、不中断活动任务、不抢当前可发出的武器攻击、不扩展宝藏或投资策略。允许现有agent/tests目录新增emergency.py/test_emergency.py，必要actions/defense/protocol/server适配，不新增依赖。开发不提交/合并/推送/切分支/reset/删除/派生任务，交付R161由主会话R162独立验收后可本地提交；本轮不自动合main或推送。


## R162 已有应急道具本地验收

主会话独立406项含16HTTP通过9.929秒；额外缺/空/多/越界目标拒绝、两阵营三火箭use后次轮原炮手恢复攻击、完整列表乱序稳定通过。允许主会话在本隔离分支本地提交；main和origin/main保持e7ff4c5送测，不合并/推送，不清理旧树/缓存。没有采购、主动任务抢占或正常炮火替换，实际库存来源与生存收益未验证。

## R165 S2宝藏结构化条件与只读评估（用户授权继续开发）

新增code/NightWatch-s2-treasure/为同仓库linked worktree，仅源码/测试/手册，沿命名约定、不放临时材料，清理须授权。分支codex/s2-treasure-conditions，基准e7ff4c563b51a44e047721eda6ae30700477dd77；main送测和应急db579b2保持只读，互不混入。新Sol+medium（01a0ab24-5617-70f1-9d03-2899e3ea7a95）唯一开发，按design/【codex】s2-treasure-conditions-plan.md实施。覆盖R113禁止候选内地点/配方/时间字段，仅允许带引用的结构化假设与本地约束检查；模型结论仍pending_validation，不自动买物品、移动、献祭、囤矿，不增新闻调用额度，不改任务/经济/战斗策略。允许现有agent/tests新增treasure.py/test_treasure.py与必要intelligence/state/brain/server修改。开发不commit/merge/push/reset/切分支/删除/派生任务；交付R166、主会话R167验收后可隔离提交，不自动合main或推送。


## R170 本次统一送测扩展修复授权

用户明确补充：刚发现的问题修复后一并送测。新Sol+medium切至code/NightWatch/main，在db579b2与4f6eb55待提交合并结果上，按design/【codex】s2-integrated-r2-repair-plan.md修任务入口文法/有界任务根、逐夜累计与预测冻结。主会话暂停源码编辑，唯一开发者为新Sol；旧树只读。开发不commit/merge/push/reset/删除；主会话验收后一并提交推送。此条覆盖R169“本次不修”限制，不扩经济优先级或攻击策略。


## R172 统一r2本地验收与当前内网入口

新Sol已完成R171并idle。主会话独立430/430含16HTTP通过10.412秒，原实际题面两变体、后续新ID累计/实际评分、黎明完整定稿反例通过，diff/cached diff/bash-n通过。构建nightwatch-s2-integrated-r2包含任务入口/有界任务根、波次累计修复、已有应急使用与宝藏条件只读评估。用户已授权提交推送，当前内网唯一入口【codex】s2-integrated-r2-intranet-prompt.md，旧条目中的暂停送测限制由本条覆盖；实战未验收，不关闭issue。


## R178 回合效率首批开发授权

本批唯一源码入口为`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch-s2-round-efficiency`，分支`codex/s2-round-efficiency`，基准`42be0e34bf303b8acc5b81d55eb40baefab5aac0`；本条覆盖历史工作区指向。新Sol+medium独占此树，按外层`design/【codex】s2-round-efficiency-plan.md`实施稳定墙段批量备料与低收益新维护采购门控，构建标识`nightwatch-s2-efficiency-r1`。`main`和全部旧工作树只读。

不得修改固定回防12回合或炮手窗口、普通采矿/采购批量、任务reader与deadline、布局、攻击、新闻、应急、SDK、依赖或环境。门控仅限制新Medicine/WallFixer采购，不阻断已有物品或有效在途承诺，并保留多工人排他和目标筛选。开发不commit/merge/push/reset/切分支/删除/派生任务；交付R179，主会话R180独立验收，本轮不自动合main或推送。


## R180 回合效率首批本地验收

主会话独立446项（含HTTP）通过30.699秒，矿刷新原反例确认完整重算后batch1并向墙施工。稳定批次按容量、路线和原12轮停工窗口计算，环境/计划外动作/库存变化使缓存失效；不改变回防窗口值。受控第10采矿消失，旧版6墙/42移动/3采集区段，新版10墙/33移动/1区段；不代表平台收益已验收。

新Medicine/WallFixer按有效恢复HP与完整承诺人员回合过滤低收益路线，普通门槛1 HP/人员回合，紧急例外及已有物品/有效承诺保留；现金、出售和联合路线逐目标评估，夜间购药按购买和使用两轮计算。构建nightwatch-s2-efficiency-r1，仅本地验收与隔离提交；main送测42be0e3不变，不自动合并推送。夜间清场离岗、动态回防和普通采购批量未实施。


## R183 本批统一送测与固定内网入口

用户授权36beb9f合入main、完整验证并推送；此前R180不合并推送为历史批次限制，现覆盖。构建nightwatch-s2-efficiency-r1。内网每次执行前读取固定https://github.com/icesword007/NightWatch/issues/12最新正文，按批次去重，完整SHA/场次/停止条件以该正文为准；旧版本提示仅历史。反馈写对应#9/#10/#11，不编辑或关闭#12。本轮不自动开发第3/4项，不以推送代替实战验收。


## R184 夜间清场释放人手开发授权

本树继续作为唯一源码入口，分支`codex/s2-round-efficiency`、基准`d594671fa7f89841c3fe68749bee0293e6ea8935`，按外层`design/【codex】s2-night-clearance-release-plan.md`实施T51。`main`正在执行R183内网验证，其他工作树同样只读；构建标识更新为`nightwatch-s2-night-release-r1`。

本批只在夜间全图攻击我方的存活机器人清零且观测充分后，逐轮解除纯炮手保护和旧返岗计划，使worker复用现有合法经济动作、pioneer复用现有任务流程；夜首帧零、缺失机器人字段、未知阵营或仍有远处我方来敌均不得释放，威胁再现时恢复原防守。保持真实昼夜与夜间禁建，不修改白天12轮回防、攻击、reader、普通采购批量、SDK、依赖或全局优化策略。

开发按TDD提供真实连续DecisionEngine反馈、收益对照及保守反例；不修改固定issue #12，不启动新对战。不得commit/merge/push/reset/切分支/删除/新目录/派生Agent；交付R185后停止，等待主会话R186独立验收。


## R186 夜间清场释放本地验收

主会话独立451项含HTTP通过30.842秒，原三轮闲守场景现连续三次collect，独立实际离岗后再现来敌/缺失观测返岗及任务accept→prompt→submit链通过，重放/列表重排一致、静态检查通过。本批仅本地验收，可隔离提交，不合main或推送；#12仍R183已送测版本，夜间清场实战收益待后续验证。


## R187 PK627784规划退化修复授权

本批唯一源码入口切回`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch`的`main`，基准`d594671fa7f89841c3fe68749bee0293e6ea8935`；按外层`design/【codex】s2-planning-regression-repair-plan.md`实施T52，构建标识`nightwatch-s2-efficiency-r2`。原`code/NightWatch-s2-round-efficiency` / `77c2176`夜间释放树及其他工作树只读，不合入本批。

只修围墙安全前缀重复全图遍历、实际耗时循环无截止控制及其造成的后续动作饥饿；保持批量备料、安全通道、固定12轮窗口、全局4秒与经济75%预算，不以加时、取消安全检查或固定2批回退。预算耗尽不得记永久unsafe/failed、不得把空初始化视为布局完成或消耗临时重试；reader、LLM答案、攻击、夜间释放和普通采购批量均不在本批。

按TDD提供真实DecisionEngine红例、结构性遍历上限、可控clock严格截止、连续动作与现有批量回归。开发不commit/merge/push/reset/切分支/删除/新依赖目录/派生Agent，不修改issue #12或启动对战；交付R188后停止，等待主会话R189独立验收。


## R189 T52本地独立验收

主会话独立451/451（含16 HTTP）通过15.953秒，diff与bash检查通过；只读安全审查及3项预算事务性测试通过。41x32同图d594为119次检查/1.271秒，修复28次/0.311秒，batch仍10；旧42be为17次/0.187秒/batch2。保留通道、不同炮位与可重试状态，未扩大总预算。仅本地提交nightwatch-s2-efficiency-r2，不推送，不混入77c2176夜间释放，不关闭issue；PK627784完整实战因果与收益待原局补证及后续送测。


## R191 任务文件查找范围修复（用户批准）

本批唯一源码入口为`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch-s2-task-input`，分支`codex/s2-task-input-scope`，基准`20e4ca6424dba72e7120060e07c87683402b7781`；按外层`design/【codex】s2-task-input-scope-plan.md`实施T53，构建标识`nightwatch-s2-task-input-r1`。`main`、`77c2176`夜间释放树及其他工作树只读。

文件名入口先完整扫描`/tmp/selfEvolutionTask`任务根域，唯一合法匹配即读取；任务根不存在或完整零匹配才回退`cwd`。任一域歧义、非普通同名目标、权限错误或时间/深度/条目预算耗尽均拒绝，两个域共享原1秒与4096条目预算；不扩大深度、正文或命令4096字符上限。显式路径和后续任务链保持不变。

按TDD覆盖优先根、回退、扫描不完整、歧义、重叠根、共享预算与命令长度。开发不commit/merge/push/reset/切分支/删除/新依赖目录/派生Agent，不修改issue #12或启动对战；交付R192后停止，等待主会话R193独立验收。


## R193 T53本地独立验收

主会话459/459含16 HTTP通过16.131秒；只读审查未发现新增确定缺陷，19项reader专项通过。原失败夹具现仅输出任务根正文；额外验证第一域命中仍遇深度中止拒绝、多根重名及乱序拒绝、完整零匹配回退、共享条目预算和祖先cwd跳过已扫描子根。默认最大512字符target命令4053字符。仅隔离本地提交nightwatch-s2-task-input-r1，不合main/推送或更新#12。main 20e4ca6继续送测，夜间释放77c2176保持独立；实战收益未验收。


## R195 统一增量合并送测（当前规则）

用户已授权将已验收4fdf34c任务根优先读取与77c2176夜间清场释放合入main并推送，保留20e4ca6规划性能修复。构建nightwatch-s2-integrated-r3；主会话处理集成与完整验证。此前各批不合并推送为历史限制，本条覆盖。夜间仅完整观测下我方目标清零才逐轮释放，有威胁或未知即恢复；读取任务根优先、完整无匹配才cwd回退，原预算共享。未开发白天动态回防或普通采购批量。内网仅按固定#12当次完整SHA/场次/停止条件执行；推送不代表实战验收，不关闭缺陷、不删除旧树/缓存。


R195集成验证：主干464/464含16 HTTP通过15.942秒。task_input与fortification分别和4fdf34c一致；defense/economy/protocol与77c2176一致，brain同时保留夜间释放与围墙独立截止预算。独立读取反例、歧义与深度拒绝/共享预算回归通过，静态检查通过。实际平台收益待#12新版复测，不降低S2验收标准。


## R196 P1工人动态回防开发授权

本批唯一源码入口为`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch-s2-day-work`，分支`codex/s2-day-work`，基准`3e0f12343b8ef68620f55a98b5ad489acf45207b`；按外层`design/【codex】s2-day-work-plan.md`实施T54/P1，构建标识`nightwatch-s2-day-work-r1`。`main`及全部旧工作树只读；P2任务求解待R198验收后另派。

白天工人按最终工作动作、真实独占炮位返程和2轮可解释安全余量决定是否继续，不把旧12轮常量机械改小。施工、采矿、交易与防守必须共享同一岗位分配；路径未知、预算耗尽、占格冲突或工作承诺不能完整兑现时沿原保守回防。开拓者任务截止/召回、夜间有敌防守、夜间清场、全局4秒与经济75%预算保持。

按TDD覆盖连续额外工作和三岗准时到达、远路提前返岗、阻挡/冲突/失效/未知/预算边界、跨日镜像与41x32性能。开发不commit/merge/push/reset/切分支/删除/新依赖目录/派生Agent，不修改issue #12或启动对战；交付R197后停止，等待主会话R198独立验收。


## R198 P1本地独立验收

原Sol修订交付后已completed/idle。主会话独立473/473含16HTTP通过19.597秒；未知路径最终经济域与basic_probe兜底均关闭放行，旧batch须本轮规划成功，完整采矿/出售/返岗及其他精确岗位投影回归通过。静态检查通过。T54本批本地验收通过，可隔离提交；main仍3e0f123内网送测，不合main/推送/更新#12。实际收益待内网，P2按既有授权由主会话另行派发。


## R199 P2任务失败证据与无效重复收敛开发授权

本批继续使用`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch-s2-day-work`与`codex/s2-day-work`，基准`adf5d33ab01cd450ea419d93522d0b01a9f547bc`；按外层`design/【codex】s2-task-feedback-plan.md`实施T55/P2，构建标识`nightwatch-s2-task-feedback-r1`。`main`及全部旧工作树只读，P1已验收语义保持不变。

失败工具结果仅在现有证据预算内保留头尾；当前任务同一命令连续两次明确非零退出后，第三次执行前最多提供一次有界纠正机会。命令按原文逐字符比较，不改写、不正规化；不同命令重置连续失败链。截止与final-only优先，不增加回合、额度或LLM调用；`exitCode:0`只表示工具进程完成，不声明业务成功。

不得新增答案/API/凭据、跨任务失败证据、通用解析器或关键词判错，不改P1动态回防、reader、任务deadline或模型shell执行协议。开发不commit/merge/push/reset/切分支/删除/新依赖目录/派生Agent，不修改issue #12或启动对战；交付R200后停止，等待主会话R201独立验收。


## R201 P2本地独立验收

Sol最终回合completed/idle，主会话独立482/482含16HTTP通过19.596秒，静态检查通过。独立连续反例确认第三次明确失败原命令不执行、重放不重复纠正、改命令与exit0不误拦；长输出当前context/history/失败观察均保留尾错误，原8192预算及短文本保持。只读审查无剩余确定阻断。T55本批本地验收通过，可隔离提交；P1行为保留。main3e0f123与#12不变，未合并推送，不声称任务实战成功率已提升。原Sol停止，不自动续开发。


## R203 P1/P2统一合并送测

用户明确授权将已验收adf5d33与5f600f3合入main、验证并推送，覆盖此前隔离不合并/推送限制。main已快进两批；构建nightwatch-s2-integrated-r4，执行入口固定issue #12，以最新完整SHA和批次为准。主会话仅统一标识/手册，无额外策略修订。R195原局约3秒规划与检查通过未提交尚待根因证据，不列为本批已修。旧树/缓存保留，不关闭issue、不重启开发任务。

R203主干验证：482/482含16HTTP通过19.843秒，diff/cached diff/bash-n通过；与5f600f3相比策略源码无差异，仅server/test_server统一构建。两批均为main祖先，无冲突。实际平台效果待内网，旧R195问题继续补证。


## R204 普通采购批量筹资开发授权

本批继续使用`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch-s2-day-work`与`codex/s2-day-work`，基准`7a1a88740f055c5d1653f9113866186d9f6c1d00`；按外层`design/【codex】s2-procurement-batching-plan.md`实施T56，构建标识`nightwatch-s2-procurement-r1`。`main`正在r4内网验证，全部其他工作树只读。

仅对尚未出发的新非紧急资金链比较立即兑现与当前矿点有限追加采集；完整计入出售、采购、使用、真实回岗与延迟成本，追加收益不改善则沿原行为。批次目标固定且受矿量、容量、真实需求、路线与P1回防约束；紧急维护、关键建设/升级、已有fund/joint/持券及时兑现，不改采购相对优先级、CPU/寻路预算或其他策略。

开发不commit/merge/push/reset/切分支/删除/新依赖目录/派生Agent，不修改issue #12或启动对战；交付R205后停止，等待主会话R206独立验收。


## R206 T56本地独立验收

原开发任务临时Astra完成续行修订，回合01a0b0bc-cd0b-7c21-bc61-beb70ad20693已completed/idle，cursor137。主会话独立509/509含16HTTP通过20.865秒，diff/cached diff/bash-n通过，续行需求缩减专项只读审查无阻断。真实连续反例确认副墙999HP后不再追加采矿，3铜整批出售并完成buy/use，副目标预留随兑现链释放。固定初始批量上限不滚动扩张，紧急与关键需求维持原链。

T56本批本地验收通过，可隔离提交；构建nightwatch-s2-procurement-r1。连续用例21工人动作降至13、14次移动降至7；首次兑现晚1轮、第二次提前8轮，实战收益待验。main7a1a887、issue #12保持r4，不合并推送、不启动对战，不自动继续开发。按用户要求原任务恢复Sol+medium。


## R207 普通采购批量筹资合并送测授权

用户明确要求版本提交远端并更新issue12。主会话获准将已验收0f7ffde快进main，统一构建nightwatch-s2-integrated-r5，完整验证后提交推送，并先保存远端原文再全量刷新#12。本条覆盖T56历史隔离不合并/推送限制；不扩大策略开发，不重启Sol，不清理旧树/缓存，不关闭缺陷或自动开局。

R207主干验证：509/509含16HTTP通过21.297秒，diff/cached diff/bash-n通过。与已验收0f7ffde相比策略源码及经济测试无差异，仅server/test_server统一r5标识；已快进无冲突。固定#12以R207完整SHA为准，平台收益待验。


## R208 任务收尾格式纠正开发授权

本批继续使用`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch-s2-day-work`与`codex/s2-day-work`，基准`365c8ccdcaa67f4e73df3cc7c79ac9c5398418fb`；按外层`design/【codex】s2-task-envelope-recovery-plan.md`实施T57，构建标识`nightwatch-s2-task-envelope-r1`。`main`正在r5内网验证，全部其他工作树只读。

仅为非法LLM信封增加一次明确、有界的格式纠正机会；纠正结果仍须通过既有严格parser，合法answer沿原deferred/allocator/submitAnswer链提交，合法abandon沿原终止链，纠正期间禁止command。剩余1轮仍保留下一轮合法answer同轮提交机会，剩余0停止；未知截止仍只允许本次纠正。不得提取或猜测答案、接受裸JSON、扩大任务截止/命令预算或改变正常command、P2重复失败保护。

开发不commit/merge/push/reset/切分支/删除/新目录依赖/派生Agent，不修改issue #12或启动对战；交付R209后停止，等待主会话R210独立验收。


## R210 T57本地独立验收

原Sol+medium交付回合01a0b753-5f51-7cc3-b6e9-00108871aab9已completed/idle，cursor143。主会话独立516/516含16HTTP通过21.094秒，diff/cached diff/bash-n通过。首次516项全绿后独立审查发现format-only与允许command的提示矛盾，已交原Sol关闭；再次独立核对纠正提示周期0/无command示例，普通prompt与365c8cc逐字一致，剩1回合纠正后截止当轮原样提交、重放一致。其他边界只读复验未发现确定阻断。

T57本批本地验收通过，可隔离提交，构建nightwatch-s2-task-envelope-r1。只证明一次格式纠正及合法提交链可用，不保证模型遵守纠正或全部answer_error/abandoned已修。main365c8cc/r5及issue #12保持内网验证，不自动合并推送或继续新开发，旧树/缓存保留。


## R211 经济规划性能定位与最小优化

本批继续使用`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch-s2-day-work`与`codex/s2-day-work`，基准`ce7bab3c8aec9d4b8e0077202704e0bc6c25d38f`；按外层`design/【codex】s2-economy-performance-plan.md`实施T58，构建标识`nightwatch-s2-economy-perf-r1`。`main`/r5、其他工作树与固定issue #12只读保持。

必须先用真实`DecisionEngine`/`propose_economy`、41×32大图和可控计数确认请求内重复开销，再只在`economy.py`现有上下文/helper内做最小消除。不改全局寻路算法，不增加跨轮缓存，不提高4秒/3秒/1600及扩展上限，不减少候选、安全校验或改变采购/施工优先级。缓存必须包含投影起点、目标、角色、占位/预留等路线输入，不得把截止或局部扩展结果当完整可达性。

先红测试，再最小实现；覆盖三塔多墙、普通/拥堵大图、批量采购新建与续行、乱序/移动/障碍变化、截止/局部扩展/无路/临夜/新请求隔离。无时压可控钟下动作、计划绑定、路线代价必须等价；性能证据分开寻路调用、实际扩展、缓存命中与Python计算，合成fixture不得宣称完全复现平台原局。开发不commit/merge/push/reset/切分支/删除/新目录依赖/派生Agent，不更新#12或开局；R212交付后停止等R213。


## R213 T58本地独立验收

原Sol+medium回合01a0b764-953e-7a71-8343-1b4247ca3ee9已completed/idle，cursor145。主会话独立523/523含16HTTP通过21.348秒，diff/cached diff/bash-n通过；只读代码审查无确定阻断。独立14墙基准实测A*636→517，扩展15062→9468，move(1,0)、fund:WallFixer:10020:20010与13轮估计不变。六组旧ce7bab3源码/新源码无时压逐字段提案及原pathSearches一致，覆盖输入乱序、临夜、无资金、障碍、多vendor/shop。

T58本地验收通过，可隔离提交，构建nightwatch-s2-economy-perf-r1。原pathSearches仍计候选访问，新增pathComputations/pathCacheHits/pathExpansions记录真正A*开销；不增加预算、候选剪枝或跨轮缓存。原平台约3秒问题未获完整原始帧，实际收益待验。main365c8cc/r5与#12保持，不自动合并推送/开局或继续开发，旧树及缓存保留。


## R214 三塔建设优先与资金保护开发授权

本批继续使用`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch-s2-day-work`与`codex/s2-day-work`，基准`40b934483a43bc891871604b8024cc61e98ec53d`；按外层`design/【codex】s2-three-tower-priority-plan.md`实施T59，构建标识`nightwatch-s2-tower-priority-r1`。`main`/r5、其他工作树与固定issue #12保持只读。

仅在第三座火箭塔有合法目标、可完成的资金与建造回岗路线及足够白天窗口时，让塔的既有承诺先于普通采矿、投资和非紧急维护；保护实际所需共享金币与builder，不假定未出售矿石为现金，不新增跨工人筹资。即时Medicine及正在受击墙的紧急维修、任务/防守占用和P1动态回防继续优先。不可行时保留原经济，不永久锁定资金或角色；不改布局、预算、夜间禁建或其他策略。

先以真实连续DecisionEngine反例复现遮挡及双工人抢金，再最小修订；覆盖现金/可售矿/不足资金、普通fund/mine/持券、既有塔计划、三塔完成恢复、失败/占位/期限/无工人、紧急/跨日、重放乱序。沿现有源码/测试/手册目录开发，不新增目录依赖。开发不commit/merge/push/reset/切分支/删除/派生Agent，不更新#12或开局；R215交付后停止等R216独立验收。


## R216 T59三塔优先本地验收

原Sol最终回合01a0b7a4-3ff2-7e33-9000-cb390900ef64已completed/idle，cursor151。主会话独立540/540含16HTTP通过22.009秒，diff/cached diff/bash-n通过。两轮独立审查发现并交回修复Medicine被采购抢占、重复买券阻挡即时修墙、新塔投影复用原图缓存、远处持券者覆盖相邻有效owner四项回归，最终反例及角色反序通过。投影单通道现拒绝错误回防，原图仍命中缓存；T58性能回归保持。

本批本地验收通过，构建nightwatch-s2-tower-priority-r1，可隔离提交。三塔目标在合法资金/工人/施工与回防窗口下优先，普通旧计划让位，即时救命保留；新紧急预选仅白天缺塔，立即use优先且保持有效owner稳定。未改布局或全局预算。连续合成用例售矿补第三塔并完成14墙仅证明本地链路，不代表原r4两塔零墙因果或平台收益已验证。main365c8cc/r5、#12和全部旧树保持，不自动合并推送、开局或清理。


## R218 r6统一送测

用户授权全部已验收T57–T59合入main，主会话快进bef9a5c并更新集成构建nightwatch-s2-integrated-r6、验证后提交推送。唯一执行入口固定issue #12，仓库s2-integrated-r6提示同步；停止追补r4，最多challenger再defender两场，完整SHA/包hash/部署与首帧buildId必须一致，错版停止。历史不合并推送限制由本条覆盖；旧树与缓存保留，不自动关闭issue或开发其他策略。


## R219 T60任务失败恢复与证据联合批次

本批唯一源码树仍为`/Users/icesword/Documents/Project/CCN-Comp/code/NightWatch-s2-day-work`、分支`codex/s2-day-work`，基准r6 `8f1bba16953c86f75f4b2401dfd73e51bfda0826`；`main`继续内网送测，其他树只读。按外层`design/【codex】s2-task-recovery-evidence-plan.md`实施工程失败恢复、答案依据/提交和API有效重试，构建`nightwatch-s2-task-recovery-r1`。

先追踪平台结果至下一命令/答案/提交的完整数据流并写受控红测，只修复有证据的局部信息损失、过期混用、工具状态误判或无效重试；需要时加有界脱敏元数据日志。不自动改写shell或硬编码题目、凭据、端点、答案；不扩大任务期限/工具额度，不改SDK/经济/防守。开发不commit/merge/push/reset/切分支/删除/新增依赖目录/派生任务，不修改#12或开局。交付R220并更新共享todo/review后停止，待主会话R221独立验收。


## R221 T60本地独立验收

原Sol回合01a0b9ba-c6b1-7123-b53f-6b7c217e890c已completed/idle，cursor154。主会话独立546/546含16HTTP通过22.207秒，diff/cached diff/bash-n通过；209组Unicode/小limit/长文本截断长度与省略数核验通过。长缺kind信封→一次格式纠正→合法尾答案原样提交及重放一致；工具归属更替/迟到/重放只读复验无阻断，新增元数据无正文。可隔离提交，不自动合main或推送。

本批修复尾部纠正信息丢失，长输出中段仍有界省略、只增明确未知与重查提示。CRLF/API/占位答案的原局语义失败未证明修复；连续测试为合成工具/模型反馈，不是执行真实check或真实网络API，更不构成程序强制完成依据门槛。r6主干保持，#12已安排现有日志定位，新字段尚未部署。
