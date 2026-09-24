# 外部测试集（held-out evaluation set）

本目录维护 FORGE 的外部验证测试集。存在理由：主测试集 Mem2ActBench 的全部 400 题在方法迭代过程中被反复使用（架构消融、组件裁决、缺陷诊断均在其上做分数选择），无法自证泛化；学界公认补救是用一个开发期间从未参与任何决策的外部集做盲测。本目录的数据集在生成之后**冻结**：只用于最终跑分（含多维指标），任何方法调整不得参考其结果。

## 数据集

### toolace_holdout/（第一外部集，同管线 held-out）

- 构造方式：Mem2ActBench 官方构建管线（`Pdev/refs/repos/Mem2ActBench/` 的 01→05 五步脚本）+ ToolACE 源数据中**未被主集使用的样本**（已用源 ID 从官方中间产物 `processed_data/conversation_sequence.csv` 反推排除）
- 格式：与 `toolmembench_small` 完全一致（`qa_dataset.jsonl` + `toolmem_conversation.jsonl` 双文件、同字段同协议 given-tool），四系统与全部评分脚本零改动直接跑
- 构建 LLM：DeepSeek-V4-Flash（官方原版用 Qwen3-Next-80B 与 Kimi-K2；构建模型不同是已声明的偏差，对四系统一视同仁不影响横向比较）
- 生成记录：见 `toolace_holdout/BUILD_LOG.md`（源样本清单、管线参数、种子）
- 已知局限：与主集同管线同分布——堵"分布贴近性"，不堵"管线偏差"，需与异源外部集搭配

### bfcl_multiturn/（第二外部集，异源，规划中）

- 目标：Berkeley BFCL V3/V4 multi-turn 类别（missing parameters / 跨轮记忆场景，工业标准），改造为同格式
- 状态：占位，未构造

## 验证协议（写死）

1. 外部集上只跑：四系统 + 全塞 + 先知，全部沿用主集配置，零调参
2. 报告多维指标（F1、错误三分、可追溯、长值保真、每题输入 token、延迟），与主集结果并列呈现
3. FORGE 在外部集的相对排序若与主集一致 → 泛化性证据；若不一致 → 如实报告并分析
