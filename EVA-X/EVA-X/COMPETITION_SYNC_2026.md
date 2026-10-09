# 2026 胸部 X 光比赛同步说明

更新时间：2026-09-23

用途：本文件是比赛信息与 EVA-X 项目的同步入口。请在 EVA-X 文件结构对话中读取本文件，并先汇报需要调整的点和修改建议，再开始改代码。不要覆盖当前已有的本地改动。

官方页面：https://www.ncccu.org.cn/index/Paper/case1.html

## 1. 比赛约束

以下信息以 2026-09-23 官方页面为准。

### 1.1 任务与数据

- 任务：胸部 X 光片多标签分类。
- 训练集：18,513 次检查，带完整标签。
- 验证集：859 次检查，带标签。
- 测试集：1,414 次检查，标签由主办方保密。
- 图片格式：JPG/PNG，尺寸不固定。
- 一个 `Study_id` 可能包含一张或多张胸部 X 光片；最终预测以检查/Study 为单位。
- 页面列出的字段包括：`Subject_id`、`Study_id`、`Predict_class`、`ViewPosition`、`PatientOrientation`、`Rows`、`Columns`、`Support Devices`。

### 1.2 十个目标类别及固定顺序

比赛目标不是 EVA-X 官方 CheXpert 的五类，也不是 CheXpert 的十四类。分类头必须使用下面的十类顺序：

```text
1  Enlarged Cardiomediastinum
2  Pneumothorax
3  Consolidation
4  Pneumonia
5  Edema
6  Cardiomegaly
7  Atelectasis
8  Lung Opacity
9  Pleural Effusion
10 No Finding
```

建议集中定义：

```python
COMPETITION_LABELS = [
    "Enlarged Cardiomediastinum",
    "Pneumothorax",
    "Consolidation",
    "Pneumonia",
    "Edema",
    "Cardiomegaly",
    "Atelectasis",
    "Lung Opacity",
    "Pleural Effusion",
    "No Finding",
]
```

页面用 `Predict_class` 描述类别 ID，实际数据可能是长表而不是一行十个标签列。数据开放后必须先打印文件列表、表头、样例行和每个 `Study_id` 的行数，再实现转换。

### 1.3 评价与提交

- 主指标：10 类 Macro-AUC。
- 辅助指标：Macro-F1；只有主指标相同或被判定为相近时才用于二次排序。
- 每日最多提交 2 次。
- 初赛提交测试集预测 CSV；复赛还要提交模型供验证。
- 页面提交示例仍残留“眼底图像/青光眼”文字，与胸片任务不一致。`submission.csv` 的最终列名、每个 Study 是否输出 10 行以及概率格式，必须以数据开放后的实际模板或组委会通知为准，不能照抄该残留示例。

### 1.4 推理环境与限制

- GPU：NVIDIA T4 16GB x1。
- CPU：8 核；内存：32GB；SSD 至少 100GB。
- 容器 CUDA 12.8、cuDNN 9；Python 3.12；PyTorch 2.10.0、torchvision 0.25.0。
- 模型文件压缩前总大小不超过 500MB。
- 单张图像端到端延迟不超过 100ms。

当前本地 EVA-X-S 通用 checkpoint：

```text
eva_x_small_patch16_merged520k_mim.pt
大小：307,569,543 bytes
SHA256：135D70A6988B5AACFE4848E1C2A0D524B2C076536FCACCDCE88B636D302316C2
```

这个文件本身低于 500MB，但最终提交不能直接使用含优化器、日志和训练状态的完整 checkpoint。需要单独导出模型权重，并在 T4 上实测模型大小、加载方式和端到端延迟。

### 1.5 数据与模型使用规则

页面明确写明：只允许使用主办方提供的训练数据集；允许数据增强；不允许使用外部标注数据；还要求作品原创、不得抄袭代码和方案。

页面没有明确说明是否允许外部预训练权重，也没有明确禁止开源模型。规则解释存在歧义：

- EVA-X 开源结构和代码：需要遵守 MIT 许可证，并在报告中注明来源。
- EVA-X-S 通用自监督权重：作为候选方案，但应向组委会确认是否允许，并披露来源。
- EVA-X-S CheXpert 五类微调权重：不作为正式提交默认方案，因为它已经使用外部 CheXpert 标签做过监督微调。
- 不得通过改文件名、重存 checkpoint 或删除元信息来隐藏权重来源。

建议向组委会确认：

> 是否允许使用基于外部无标签胸片进行自监督预训练的 EVA-X-S 通用权重？下游分类头重新初始化，训练和微调只使用本赛题提供的数据，不使用任何外部标注数据。

## 2. 当前 EVA-X 与比赛的差异

以下判断基于当前 `classification/` 目录，不要把官方 CheXpert 示例直接当成比赛实现。

### P0：必须调整

1. **新增比赛数据集适配器**

   当前 `classification/utils/datasets.py` 只分派 `chestxray`、`covidx`、`chexpert`。应新增独立的 `competition` 数据集类型，不要修改 CheXpert 类来硬塞比赛格式。

2. **处理 Study 级别和多图**

   `Study_id` 可能对应多张图片，而评价和提交是检查级别。第一版建议：

   ```text
   image -> EVA-X-S -> image logits
   同一 Study 的多张 image logits -> mean-logit pooling
   study logits -> sigmoid -> 10 类概率
   ```

   后续再比较 mean probability、max pooling 和 attention pooling。

3. **按 Subject_id 防止泄漏**

   训练/验证划分必须以 `Subject_id` 为组，不能随机按图片或 Study 拆分同一患者。需要保存 split 文件并检查患者重叠。

4. **十类分类头**

   EVA-X 官方 CheXpert 配置的 `--nb_classes` 是 5。比赛应改为 10，并使用上面的固定顺序。使用通用 EVA-X-S 权重时，新建 10 类 head；加载五类分类 checkpoint 时必须丢弃旧 head。

5. **损失与指标**

   仍使用多标签：

   ```text
   logits: [batch, 10]
   loss: BCEWithLogitsLoss 或带明确 mask/pos_weight 的变体
   probability: sigmoid(logits)
   ```

   不要使用 Softmax。评估必须返回 10 个 per-class AUC 和严格的 Macro-AUC。当前 `engine_finetune.py` 在 AUC 计算异常时返回 0，并在平均时排除 0，这可能掩盖无阳性/无阴性类别并抬高指标；比赛适配代码应显式报告缺失类别并阻止错误的 Macro-AUC。

6. **目标标签解析**

   如果实际训练文件是长表，`Predict_class` 需要按 `Study_id` pivot/multi-hot 成 `[10]` 标签。必须确认：

   - 同一 Study 多行是否代表多张图片、多个阳性类别或两者混合；
   - 一个 Study 没有某个类别行时是否表示阴性；
   - `No Finding` 与疾病标签共现时以原始标签为准，并记录异常；
   - 标签 ID 是否从 1 开始，并映射到 0 到 9 的模型列。

7. **独立的提交生成器**

   增加 `predict.py`/`make_submission.py`，严格按照真实测试模板生成文件，保持 `Study_id` 顺序，输出原始概率而不是只输出 0/1。提交逻辑不能依赖训练脚本中的 CheXpert 特殊分支。

### P1：强烈建议调整

1. **不要沿用 CheXpert 专用硬编码**

   `classification/utils/dataloader_med.py` 当前默认五类，并写死了 CheXpert 的 `-1` 替换、`Frontal/Lateral` 过滤、路径前缀和训练集特殊重复采样。比赛应使用新 Dataset 类。

2. **先做图像质量审计，不删除困难样本**

   添加检查：坏图、空图、路径不存在、重复图、Study/Subject 泄漏、尺寸/通道、亮度极端值和字段缺失。只删除确认损坏的文件；AP、旋转、过暗、过亮、设备存在等有效困难样本应保留。

3. **元数据先不接入主模型**

   `ViewPosition`、`PatientOrientation`、`Rows`、`Columns`、`Support Devices` 是数据字段，不在十个目标类别列表中，但可能造成设备或医院捷径。第一版使用纯图像 EVA-X-S；之后做消融：纯图像、图像加视角、图像加全部元数据。只有多种 Subject 级划分和多个 seed 都稳定提升，才考虑后融合。未知值应有 `UNKNOWN`，但不要默认以 20% 随机屏蔽字段。

4. **不确定标签策略以比赛数据为准**

   不要把 CheXpert 的 `-1` 规则直接搬过来。若比赛只有 0/1，就不需要 U-Zero/U-One；若存在 `-1` 或空值，应先统计每类分布，再比较 U-Zero、U-Ignore 或软标签。

5. **支持单卡 Windows 训练**

   官方 `vit_s.sh` 使用 4 张 RTX 3090、`batch_size 256` 和分布式启动，不能原样运行在 RTX 4060/4060 Ti。建议新增单卡配置：

   ```text
   EVA-X-S, 224x224
   batch_size 2-4（按显存调整）
   AMP
   gradient accumulation 4-8
   num_workers 2 起步
   seed 42/123/2026
   ```

   单卡评估不能假设 `model.module` 一定存在；当前 `train.py` 的 `--eval` 分支需要兼容普通 model 和 DDP model。

6. **先做依赖验证**

   当前本地改动已经为 Apex 和 `torch.load(weights_only=...)` 做了部分 Windows/PyTorch 兼容处理，但 `models_eva.py` 仍依赖 `xformers`。先用一个 batch 验证 `torch`、`torchvision`、`timm`、`xformers`、EVA-X 权重加载和前向输出，不要凭空升级/降级整个环境。

### P2：基线稳定后再做

- 多 seed 概率平均或模型集成。
- 按类别计算 `pos_weight`，并与普通 BCE 做消融。
- 只在验证集调每类 F1 阈值；AUC 使用概率，不依赖阈值。
- 研究 `No Finding` 的逻辑一致性约束，但不能未经验证强行覆盖原始标签。
- 224x224 稳定后再试高分辨率；先确认 T4 的 100ms 单图延迟。
- Grad-CAM 只用于错误分析和报告，不是训练必需项。

## 3. 建议的实现顺序

```text
1. 数据开放后建立 data_inspect.py，输出文件、列、样例和类别统计
2. 确认长表/宽表、Study 多图关系和 submission 模板
3. 建立 Subject 级 train/val split，并做泄漏检查
4. 建立比赛 Dataset：读取图片，按 Study 聚合标签，返回 image/study_id/labels
5. 用 EVA-X-S 通用权重 + 10 类 head 跑一个 batch 和 1 个 epoch
6. 修复严格 Macro-AUC、checkpoint、日志和 best model 保存
7. 在 RTX 4060 上跑纯图像 baseline
8. 对 3 个 seed 做验证，保存 per-class AUC、Macro-AUC、Macro-F1 和配置
9. 比较元数据后融合和多图聚合
10. 导出仅模型权重，实测 T4/本地推理延迟和文件大小
11. 生成 submission.csv，并用官方模板/小样例做格式校验
```

## 4. 交给 EVA 文件结构对话的汇报要求

请读取本文件后，按以下格式回复，不要直接跳到大规模训练：

1. 当前 EVA-X 文件结构中哪些文件需要改，逐文件说明原因。
2. 哪些官方文件应保留不动，哪些功能应新建独立文件。
3. 当前代码中会阻止 10 类比赛任务运行的硬编码、路径假设和 DDP 假设。
4. 10 类标签、Study 多图聚合、Subject 划分和提交格式的具体实现建议。
5. RTX 4060 本地训练与 T4 推理的配置建议。
6. 规则上仍需向组委会确认的事项，尤其是通用自监督预训练权重。
7. 分阶段修改清单、每一步的验证命令和预计输出。

汇报完成后，优先实现数据检查和 Dataset/标签转换，不要在实际比赛数据尚未开放前修改官方 CheXpert 数据加载器，也不要把十类任务误写成五类、十四类或单标签分类。

