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

- 首先阅读根目录[内网验证手册](【codex】intranet-validation.md)，该手册为内网S0接管入口，不需要外层design或原始SDK即可按步骤验证。
- 内网职责仅为拉取明确版本、打包上传、对战、分析及在本仓库issue反馈；不自行改源码、策略、依赖或启动脚本。平台适配差异先报告并记录，不能用原SHA代表改过的程序。
- 根目录生成Markdown统一使用【codex】前缀；本AGENTS.md固定文件名例外。测试来源说明引用外部材料仅为历史来源，不是内网执行依赖。
- 原始日志留内网，只向issue提交符合内网要求的最小证据摘要；不得上传密钥、token、任务私密内容或完整日志。
- 已跟踪文件有本地修改时不得覆盖或回滚，应报告差异。远端issue是反馈证据，不自动授权本地修改范围。
