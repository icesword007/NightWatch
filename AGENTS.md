# NightWatch 常设工程规则

## 范围与目录

- 中文沟通，称呼用户喵总。仓库CoreGeek是改造后的参赛程序，不能当赛方原始SDK或规则；外层正式资料只读。
- `CoreGeek/src/agent/`存参赛模块；`CoreGeek/main3.py`入口，`CoreGeek/run.sh`启动。`CoreGeek/tests/test_*.py`及fixtures存受控测试；不提交真实日志、题面答案、凭据、.DS_Store或缓存。
- `CoreGeek/analyze_logs.py`只读分析本地服务端逐轮日志，`CoreGeek/tests/test_log_analysis.py`用合成记录验收；`【codex】offline-log-analysis.md`记录用法与字段边界。分析结果只向stdout输出脱敏JSON，不在仓库保存输入或原始内容。
- 仓库生成Markdown使用【codex】前缀；AGENTS.md例外。新目录先约定用途。临时材料用系统临时目录，不混入仓库。

## 分工与操作边界

- 原Sol+medium独占主会话明确派发的开发树源码；主会话只读验收、文档和获授权集成。每批开始核对pwd、branch、HEAD、worktree list；没有新派发不自动重启旧批次。
- 源码任务默认不commit/merge/push/reset/切分支/删除/派生任务，不操作其他工作树。主会话验收后按授权提交/集成；不全量git add，不覆盖未提交改动。
- 删除文件/目录/缓存/工作树、git回滚、修改.env/密钥/CI/CD、数据库迁移、全局依赖/系统配置、公开发布须用户明确授权。已有历史授权不等于可任意扩展。
- 本地协调的当前目录/版本/待办见外层design/【codex】current-status.md及todo-list.md，不把AGENTS当版本流水。内网执行不依赖外层文件。

## 实现与验证

- 先证据/受控反例，再最小修复；不附带重构、格式化、吞异常或删测试。协议问题完整逐字段对照官方文档；不凭猜测硬编码任务答案、端点或认证。
- 默认标准库，不新增依赖；平台Python 3.11兼容。题面/LLM shell只由平台执行，不能在本机执行任意外来命令；测试用受控合成材料。
- 在CoreGeek目录执行：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'`。HTTP测试需本地端口权限，受限失败不等于源码失败，按工具审批重跑适用项。
- 批末`git diff --check`、`git diff --cached --check`、`bash -n CoreGeek/run.sh`。只在源码改动/失败/未决风险需要时重复完整回归；纯文档修改验链接/内容即可。
- 原开发最终回归与主会话独立验收保留。说明本地/平台边界，提交记录只写一次验证摘要，不复制到多份AGENTS。

## 内网执行

- 唯一当前任务入口：https://github.com/icesword007/NightWatch/issues/12 。先读该正文，再按需读【codex】intranet-validation.md；仓库旧版本提示只归档，不自动续跑。
- 内网仅拉取指定版本、核对差异、打包上传、按指定场数对战并分析反馈，不自行改源码/策略/依赖/启动脚本，不覆盖本地修改。
- 完整SHA、包SHA256、平台实际启用与首帧buildId一致才验收；错版停止。日志、完整题面、答案、TOKEN和凭据留内网，issue只写脱敏证据。
- #17经济性能、#18任务/新闻、#19防守/旁路；旧#9/#10/#11及#13/#14/#15仅作迁移历史，关闭不表示修复。#12由主Agent全量刷新，内网每次先重读最新正文，不修改/关闭；不开额外场次，不自动关闭缺陷。命令发出、进程exit0、胜场都不是功能全部通过。

## 当前保留边界

- 三塔为既定布局目标；普通投资不得遮挡合法可完成的补塔，保留即时自救/紧急修墙、合法路线与回防。
- 外层任务信封严格解析，语义正确性仍依赖实际任务/模型与裁判；不猜答案，不扩任务工具或截止预算。新闻/宝藏候选不是平台事实，已有宝藏能力仅评估；新增采购/献祭/进攻/仿真须明确新批次。
- 已有能力/历史授权由Git与主会话批次记录追溯，不在本文件追加流水。
