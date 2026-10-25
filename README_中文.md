# SGDB 复现说明

最终检测配置为 `SGDB_V2`（complete_support V2）。
主稿结果：LOCO 98.19 / 94.52，VID 全50任务 93.56 / 92.76（AUROC / F1，%）。

按英文 README 的 Quick start 操作即可：Python 3.12 + requirements-cpu.txt。
无需 GPU、原图、检测器训练、baseline 推理或阈值重新拟合。

- reproduce.py：复算 18 种配置、990 个任务行，核对论文715行和配对区间。
- replay_facts.py：重放9,031张图的评分、判定及解释证书。
- replay_language.py：复核70条保存的语言输出，不调用LLM。
- verify_bootstrap.py：独立检查重采样组展开及成对AUROC计算。

逐图预测、正常角色、分组、冻结规则/参照/阈值、分条件FPR和完整变化清单均在仓库中。
选择新的输出目录运行；不覆盖 results。

research_source 收录真实研究算法模块，已有15个小型最终语义解码器权重一并保留。
整套视觉前端权重及外部OCR/LLM模型没有包含在本次发布中，预期哈希见资产清单。
此次已验证的是冻结输出/事实级复现。原图端到端推理还需要相应模型和原始数据。

VID 按跨条件文件序号分组；不将其称作已验证的物理实例或视频分组。
公开测试曾参与开发，本次配对区间用于条件稳定性分析。
外部方法的监督与本地实现差异见 comparison_conditions.csv。
