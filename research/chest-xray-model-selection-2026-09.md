# CheXpert 胸片模型选型报告

更新时间：2026-09-20

## 结论

针对当前项目的 CheXpert 多标签胸片分类任务，推荐采用以下路线：

1. **主模型：EVA-X-S**
   - 官方仓库直接报告了 CheXpert 官方验证集上的 mAUC：**90.1**。
   - 22M 参数，公开预训练权重和分类代码。
   - 输入配置为 224x224，适合 RTX 4060，工程改造成本低。
   - 适合作为第一版可复现、高质量、多标签分类骨干。

2. **高上限模型：CheXFound ViT-L + GLoRI**
   - 使用超过 100 万张胸片进行自监督预训练。
   - ViT-L/16，512x512 输入，并配套 Global and Local Representations Integration（GLoRI）分类头。
   - 公开源码和检查点入口，适合在冻结骨干或小批量微调条件下做高质量实验。
   - 模型较重，官方代码按 Linux、多 GPU 环境测试，不适合作为 RTX 4060 上的第一条训练线。

3. **最新模型：CheXficient**
   - 2026 年公开的胸片视觉语言基础模型。
   - 使用 DINOv2-base 图像编码器和 BioClinicalBERT 文本编码器，公开了 Hugging Face 权重。
   - 论文覆盖零样本分类、疾病预测、分割、报告生成等 20 个基准。
   - 但公开模型卡主要展示图文零样本接口，尚不能据此断言它在本项目的 14 类监督式 CheXpert 分类上一定优于 EVA-X-S。
   - 建议作为第二阶段的零样本、特征集成或对照实验。

## 候选模型比较

| 模型 | 类型 | 与本赛题匹配度 | 公开资源 | RTX 4060 适配性 | 判断 |
|---|---|---:|---|---:|---|
| EVA-X-S | 胸片专用 ViT-S/16 | 很高 | 权重、分类代码 | 很好 | 首选 |
| CheXFound ViT-L + GLoRI | 胸片专用视觉基础模型 | 很高 | 源码、检查点入口 | 较低 | 高上限实验 |
| CheXficient | 胸片视觉语言双编码器 | 中高 | Hugging Face 权重 | 较好 | 最新实验分支 |
| MedSigLIP-448 | 通用医学图文编码器 | 中高 | 官方权重、微调示例 | 中等 | 可做集成 |
| CXR Foundation / ELIXR v2 | 胸片 embedding 模型 | 中高 | 官方权重和 notebook | 较好 | 旧方案，不作为新主线 |
| CheXagent | 胸片视觉语言生成模型 | 中等 | 权重、推理代码 | 较低 | 适合报告生成，不适合主分类器 |
| DenseNet121 | ImageNet 传统基线 | 中等 | torchvision | 很好 | 仅用于 baseline |

## 关键证据

### EVA-X

EVA-X 是专门针对 X-ray 图像预训练的视觉 Transformer，公开了 Ti、S、B 三种规模以及完整分类代码。官方分类结果中，CheXpert 官方验证集的 EVA-X-Ti mAUC 为 89.6，EVA-X-S mAUC 为 90.1；训练分辨率为 224x224。EVA-X-S 有 22M 参数，EVA-X-B 有 86M 参数。

来源：

- https://github.com/hustvl/EVA-X
- https://arxiv.org/abs/2405.05237

### CheXFound

CheXFound 在 CXR-1M 上进行自监督预训练，使用 ViT-L/16 和 512x512 输入，并通过 GLoRI 融合全局特征和疾病相关局部特征。论文声称其在 CXR-LT 24 的 40 类发现分类上超过已有方法，并在少标签和分布外任务上表现良好。官方仓库提供了训练、评估、解释性分析代码以及检查点入口。

来源：

- https://arxiv.org/abs/2502.05142
- https://github.com/RPIDIAL/CheXFound

### CheXficient

CheXficient 是 2026 年公开的胸片视觉语言基础模型。论文报告其仅使用 1,235,004 对胸片和报告中的 22.7% 样本进行预训练，同时达到与全量数据模型和其他大规模模型相当或更好的结果。公开 Hugging Face 模型为约 0.2B 参数的 DINOv2-base + BioClinicalBERT 双编码器，支持图文相似度和零样本分类。

来源：

- https://arxiv.org/abs/2602.22843
- https://huggingface.co/StanfordAIMI/CheXficient

### MedSigLIP

MedSigLIP 是 Google 的医学图文编码器，包含 400M 参数的视觉编码器和 400M 参数的文本编码器，输入分辨率为 448x448。官方明确建议它用于不需要文本生成的医学图像分类、零样本分类和语义检索，并支持针对下游任务微调。其公开 CheX-ray 零样本平均 AUC 为 0.844，但该数字与监督微调后的 CheXpert mAUC 不可直接比较。

来源：

- https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card
- https://huggingface.co/google/medsiglip-448

### CXR Foundation

Google 的 CXR Foundation / ELIXR v2 曾报告在 CheXpert 五项任务上的数据高效分类平均 AUC 为 0.898，支持从胸片提取 embedding 后训练轻量分类器。但官方当前模型卡已将其标记为 legacy，并建议新项目使用 MedSigLIP，因此不建议把它作为本项目新主线。

来源：

- https://developers.google.com/health-ai-developer-foundations/cxr-foundation/model-card

## 本项目的实施建议

### 第一阶段：可复现主线

- 使用 EVA-X-S 预训练权重。
- 替换最后分类头为 14 个独立输出的多标签 head。
- 使用 `BCEWithLogitsLoss`。
- 对每个标签计算 `pos_weight`。
- 对不确定标签分别比较 U-Zero、U-One、U-Ignore 和自训练策略。
- 输入先使用 224x224，确认训练流程稳定后再测试 448x448。
- 指标同时记录每类 AUROC、AUPRC、F1、Recall，以及 macro/micro 平均值。

### 第二阶段：高上限

- 使用 CheXFound ViT-L 作为冻结或半冻结 backbone。
- 优先冻结 backbone，只训练 GLoRI 或轻量 14 类分类头。
- RTX 4060 上使用 512x512、batch size 1 或 2、AMP 和梯度累积。
- 如果显存和速度不可接受，回退到 EVA-X-S。

### 第三阶段：集成

- 训练 EVA-X-S 和 CheXFound 两个独立模型。
- 在验证集上进行逐类 logit 或概率加权。
- 每个疾病单独寻找集成权重和分类阈值。
- 不要直接平均所有输出，先检查两个模型的错误相关性。

## 重要风险

1. **外部预训练数据规则**
   - CheXpert、MIMIC-CXR、NIH-CXR 等公开数据可能已经被候选基础模型用于预训练。
   - 如果比赛规则禁止外部数据或要求从头训练，必须先确认是否允许使用这些预训练权重。

2. **指标不可直接横向比较**
   - EVA-X 的 90.1 是 CheXpert 官方验证集的监督微调 mAUC。
   - MedSigLIP 的 0.844 是零样本评估平均 AUC。
   - CheXFound 和 CheXficient 的论文重点基准不完全是同一个 CheXpert 任务。
   - 因此不能简单把不同论文中的数字排成一个绝对排行榜。

3. **数据泄漏与患者划分**
   - 训练、验证和测试必须按患者划分。
   - 对 CheXpert 的不确定标签不能默认全部当普通正样本或负样本。
   - 任何外部预训练模型都应在本项目固定验证集上重新测量，而不是直接相信论文平均指标。

## 最终选择

如果现在只选一个模型开始训练：**EVA-X-S**。

如果目标是尽可能提高最终分数：**EVA-X-S 主线 + CheXFound 冻结骨干实验 + 逐类概率集成**。

如果目标是研究最新模型：加入 **CheXficient** 和 **MedSigLIP**，但把它们视为实验/集成模型，而不是直接替换 EVA-X-S。
