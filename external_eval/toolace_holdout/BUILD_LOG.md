# toolace_holdout 数据集构建说明

本数据集由 Mem2ActBench 官方构建管线（五步脚本）在 ToolACE 源数据的未用样本上生成，与 toolmembench_small 同格式、同协议。

## 构建方法

1. 源池与划分：ToolACE 格式化源共 7,924 条。已用于主集的 6,246 条（以官方中间产物 conversation_sequence.csv 的 14,094 条源记录反推）全部排除，剩余 1,678 条从未使用。以固定种子 20260925 从中抽取 600 条作为本数据集的源，清单见 heldout_source_ids.txt，源记录见 toolace_heldout_source.jsonl
2. 构建管线：官方 01（会话构建）→ 02（事实与冲突抽取）→ 03（排序）→ 04（QA 构造）→ 05（规范化），管线脚本位于上游仓库
3. 构建 LLM：DeepSeek-V4-Flash（上游原版使用 Qwen3-Next-80B 与 Kimi-K2；构建模型不同是已声明的偏差，对全部被比较系统一致，不影响横向比较）
4. 产出：本目录下的 qa_dataset.jsonl 与 toolmem_conversation.jsonl，格式与 toolmembench_small 完全一致，四系统与全部评分脚本零改动直接运行

## 使用纪律

本数据集生成后冻结：只用于最终跑分（四系统 + 全量供给 + 先知，多维指标），任何方法调整不参考其结果。
