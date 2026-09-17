# PROMPT — m2a 开发指导

本目录存放指导项目推进的文档：开发提示词、技术决策记录（decisions/）、技术笔记（notes/）。

## 项目定位

m2a 是一个 memory-to-action harness：在长期记忆与工具调用之间提供确定性的编译式接口。
给定用户查询、工具 schema 与记忆库，产出 schema 合法、参数值可溯源的工具调用。

- Benchmark：Mem2ActBench（ACL 2026, arXiv 2601.19935），数据与论文见其官方仓库
- 评估指标：参数级 F1 / BLEU-1 / Tool Accuracy（主实验 given-tool 模式），TSA / EM / Arg_F1（工具选择压力测试）
- 运行环境：本地 vLLM + Qwen2.5-7B-Instruct（temperature=0），embedding 用 BGE-M3

## 开发准则

1. **确定性与概率性分工**：能确定性判定的逻辑（类型检查、默认值填充、逐字复制、必填项校验）不交给 LLM；LLM 只负责自然语言理解与意图提取
2. **可溯源**：每个生成的参数值必须携带来源标注（记忆条目 ID / schema default / 模型推理），评估与调试依赖它
3. **可测试**：确定性模块（绑定、冲突、回退）必须有单元测试；LLM 参与的环节用小规模固定样本回归
4. **消融友好**：每个机制一个开关，实验脚本可单独关闭任何一环
5. 文档不写增量式备注；决策记录进 decisions/（背景/选项/结论/理由）

## 目录约定

```
m2a/           核心包：schema / state(写入侧) / act(读取侧) / eval
experiments/   基线封装、全量实验、消融脚本
tests/         单元测试
Pdev/          本目录：指导文档
```

## 当前阶段

M0：evaluator 实现与基线校准（LTMemory 复现对齐论文 7B 数字：F1=26.71, TA=87.25）。
