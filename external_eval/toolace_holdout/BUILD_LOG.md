# toolace_holdout 数据集构建说明

本数据集由 Mem2ActBench 官方构建管线（五步脚本）在 ToolACE 源数据的未用样本上生成，与 toolmembench_small 同格式、同协议。生成后冻结。

## 构建方法

1. **源池与划分**：ToolACE 格式化源共 7,924 条。已用于主集的 6,246 条（以官方中间产物 conversation_sequence.csv 的 14,094 条源记录反推）全部排除，剩余 1,678 条从未使用。以固定种子 20260925 从中抽取 600 条作为本数据集的源，清单见 heldout_source_ids.txt，源记录见 toolace_heldout_source.jsonl。
2. **构建管线**：官方 01（会话构建）→ 02（事实与冲突抽取）→ 03（排序）→ 04（QA 构造）→ 05（规范化），管线脚本位于上游仓库。全部五步的调用方式与本地适配见 build/ 目录：run_01.py（薄包装，上游 01 只支持连续编号，而本集 600 条源为散布编号，故直接喂入源文件、复用上游的提示词与解析）、03_conv_sort.py 与 04_qa_construction.py（适配副本，改动见各文件头部注记）、run_05.py（包装，输出文件名对齐 toolmembench_small）、build_all.sh（串行总控）、spot_check.py（抽查，见第 5 条）。
3. **构建 LLM**：DeepSeek-V4-Flash，全部五步统一（上游原版使用 Qwen3-Next-80B 与 Kimi-K2；构建模型不同是已声明的偏差，对全部被比较系统一致，不影响横向比较）。
4. **与官方管线的三处必要偏差**（其余步骤与参数均照原样）：
   - 官方仓库未随代码发布 02 步所需的提示词文件 prompts/resolve_fact_group原始.md（官方 prompts/ 目录仅含两个文件）。该提示词按 02 代码的输入输出契约重建（输入为按属性分组的陈述列表，输出为 sorted/discarded/summary/reasoning 四字段），重建文本见本地补文件，语义上忠实于代码的校验逻辑。
   - 03 步的随机打乱在官方脚本中无种子，会在布局上不可复现；适配副本固定种子 20260925。
   - 04 步官方脚本内硬编码了作者自用的 API 配置（会覆盖环境变量）；适配副本改为读 FORGE_API_BASE/FORGE_API_KEY 环境变量。
5. **质量抽查**：固定种子 20260925 抽 25 题做五项机械检查（字段完整、explicit 金标值在可见会话中在场——数值做数值等价、数组逐元素、参数泄漏、会话链完整、查询卫生），结果 25/25 通过（spot_check 报告见 build/spot_check_report.json，含全部 25 题的人读摘录）；另对抽中样本做语义细读，查询自然度、参数隐式化、inferred 标注合理性均合格。

## 产出（冻结）

- qa_dataset.jsonl：89 题 / 164 个金标参数。难度分布 L1=59、L2=18、L3=5、L4=7；接地类型 explicit=110、inferred=37、default=21；单源题 77、双源聚合题 12。
- toolmem_conversation.jsonl：41 个长会话。
- 格式与 toolmembench_small 完全一致（同字段、同 given-tool 协议），四系统与全部评分脚本零改动直接运行。

## 使用纪律

本数据集生成后冻结：只用于最终跑分（四系统 + 全量供给 + 先知，多维指标），任何方法调整不参考其结果。
