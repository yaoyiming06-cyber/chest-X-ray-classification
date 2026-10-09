# Windows 训练手：比赛实操与代码修改交接

日期：2026-10-09。发送方身份：Mac / Codex 代码手。接收方身份：Windows / RTX 4060 训练手及 Windows 端 AI。

目标：先在 CheXpert 上建立十标签可复现流程，再按比赛规则适配正式数据。本文给出当前决策、文件修改范围、执行顺序与验收要求。下面标记为“待实现”的文件、配置和命令是开发接口约定，当前不能直接运行。

## 1. 先读这一页：当前状态与执行优先级

2026-10-09 拉取 `1c2c520` 后：除根目录文档和环境检查外，已存在完整 EVA-X 源码、`EVA-X/EVA-X/classification/train.py`、`classification/competition/` 数据/损失/评估/预测组件，以及根目录 CheXpert 标签准备、患者划分和 Windows 批量实验工具。具体比赛接口见 [competition README](../EVA-X/EVA-X/classification/competition/README.md)。

仍待对齐：本文 E0-E6 统一策略配置、固定十标签患者划分、三种子结果矩阵及后续消融接口。现有比赛适配提供逐类混合策略及两阶段自训练，不等于六组统一策略实验已全部实现；官方提交格式仍待确认。本次 Mac 端只核查代码与文档，未验证 Windows GPU 运行或结果。

此前空 gitlink 状态已由远程更新修复，`EVA-X/EVA-X` 现为普通受版本控制文件，不需要为此执行 submodule 初始化。requirements 中的固定版本与 Windows 兼容性仍须在训练机实测。

执行顺序固定为：

```text
P0 环境与数据审计
  -> P1 固定患者划分 + 纯图像 E0 跑通
  -> P2 六策略 E0-E5 + 验证结果矩阵
  -> P3 E6 混合策略 + 前两套策略三种子复验
  -> P4 图像增强 / blank / 类别加权独立消融
  -> P5 分层审计 + 可选元数据模型、专家模型
  -> P6 分辨率、集成、TTA、校准、阈值
  -> P7 冻结方案、最终测试与比赛提交
```

阶段名优先于聊天中的“第二轮/第三轮”；E0-E6 专指标签实验，增强使用 G 编号。不要求先实现后面所有功能才开始 E0。

## 2. 当前确定的实验规则

| 项目 | 约定 |
| --- | --- |
| 输出 | 十标签，顺序见下一节；原始十四观察项保留用于审计 |
| 划分 | 在可用训练数据内按患者 80:10:10；比赛有官方划分时优先遵守 |
| 数据范围 | 第一主线只用正面片，保留 AP/PA；侧位片是后续独立范围 |
| 主模型 | EVA-X-S 通用预训练 + 新任务头，核对权重来源与比赛许可 |
| 工程备用 | DenseNet121；作为独立实验线，不混进 EVA-X 策略排名 |
| 初始输入 | 224 x 224；归一化遵循骨干对应预训练配置并记录 |
| 基础增强 | G1：轻微旋转 ±5°、轻微亮度/对比度；第一轮全组相同 |
| 验证预处理 | 确定性缩放与归一化，无随机增强 |
| 第一轮权重 | 关闭，所有类别权重为 1 |
| 第一轮 blank | 九个病变 blank -> 0；原始四状态仍保存 |
| No Finding | 原字段 1 -> 1，0/blank -> 0；出现 -1 时报告错误并核查 |
| 随机种子 | 初筛 42；复验 42 / 2026 / 3407，患者划分始终固定 |
| 主指标 | 暂定十类 macro AUROC；正式比赛明确后在筛选前登记替换 |
| 辅助指标 | 每类 AUPRC、Precision、Recall、F1、有效样本数和覆盖率 |

三个种子改变模型初始化、训练数据顺序和增强随机性，不重新随机划分患者。No Finding 不由其余九类推导。

外部数据保留既有路线：CheXpert -> NIH 对照 -> 允许时 MIMIC 扩展 -> VinDr 定位分析 -> Shenzhen/Montgomery 结核补充。不同标签体系先映射并记录，外部数据不能都当成“十类金标准测试集”；结核二分类也不能替代十类评估。官方五类专家参考结果与内部十类报告弱标签结果分开报告。

## 3. 代码修改地图（职责规划，不是现有文件清单）

以下 `src/cxr` 路径是拉取前的接口规划，不要求另建一套平行框架。优先把 labels/data/policies/losses/metrics 职责映射到 `EVA-X/EVA-X/classification/competition/` 现有模块，训练与骨干适配扩展现有 `classification/train.py` 和模型代码；先检查根目录 `scripts/`、`tools/` 再新增入口。配置与返回值是目标接口，迁移时须适配现有调用方并补测试，不能直接替换现有 Dataset 契约。

| 文件 | 需要实现或修改的内容 |
| --- | --- |
| `src/cxr/labels.py` | 唯一标签顺序、四状态常量、策略名称；其他文件导入而不复制列表 |
| `src/cxr/data.py` | CSV 与图像读取、患者/study ID、字段映射、训练目标和固定评估目标分别生成 |
| `src/cxr/policies.py` | -1/blank 映射、二分类目标、CE 目标、掩码、教师软目标和类别权重统计 |
| `src/cxr/transforms.py` | G0-G3、训练/验证不同流程、RGB 通道与归一化一致 |
| `src/cxr/model.py` | EVA-X/DenseNet 骨干适配、各分支任务头、元数据可选融合、统一十概率接口 |
| `src/cxr/losses.py` | masked BCE、逐分支 CE、权重、有效样本归一化、混合损失 |
| `src/cxr/metrics.py` | AUROC/AUPRC、阈值选择、分组统计、患者 bootstrap、NA 处理 |
| `src/cxr/training.py` | AMP、梯度累积、训练循环、验证、断点恢复、checkpoint 选择 |
| `scripts/inspect_chexpert.py` | 四状态计数、路径、损坏图、重复图、患者信息、元数据审计 |
| `scripts/split_patients.py` | 固定患者划分、清单与哈希、交集和重复图泄漏检查 |
| `scripts/train.py` | 加载 YAML、构建训练/验证 loader 和模型、写入完整运行信息 |
| `scripts/cache_teacher.py` | 为训练患者生成不确定标签软目标缓存 |
| `scripts/evaluate.py` | 固定真值评估、概率保存、分组报告；最终测试不执行阈值拟合 |
| `scripts/tune_thresholds.py` | 仅在验证预测上拟合阈值/校准，保存来源信息 |
| `scripts/run_experiments.py` | 按实验清单顺序运行、遇到失败停止、成功后才跳过已完成组 |
| `scripts/summarize_experiments.py` | 生成三种 10 x 6 矩阵、总体排名、种子均值/标准差 |
| `scripts/predict.py` | 执行冻结推理链，按比赛 sample_submission 校验行列 |
| `configs/base.yaml` | 公共实验条件；除实验变量外各组继承相同值 |
| `configs/experiments/*.yaml` | E/G/B/W/M/X 的实验变量，不写本机路径 |
| `configs/local.windows.yaml` | 本机数据、权重与输出路径；不进入 Git |
| `tests/test_policies.py` | 四状态映射、No Finding 例外、伪标签替换、类别权重 |
| `tests/test_training_contract.py` | 掩码梯度、头维度、预测范围、患者隔离与断点一致性 |

新增 Python 包时包含 `src/cxr/__init__.py`。脚本通过从脚本路径解析项目根目录加入 `src`，或建立明确的可编辑包安装方式；不能依赖某台机器的 PYTHONPATH。选一种并写入 README。

本机路径只改 local 配置；策略、种子和增强只改 experiment 配置；增添策略改 policies + losses + 对应测试；改骨干只改 model 适配；更换数据集改 data 适配和字段映射。原始 CSV 不覆盖。

## 4. 固定数据、目标与模型接口

唯一标签顺序：

```python
LABELS = (
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
)
```

数据读取返回 `sample_id`、`patient_id`、`study_id`、图像、原始十目标及元数据。缺失字段用 Unknown；缺失患者 ID 不按图片悄悄划分，先核查路径或来源。路径分隔符用 pathlib 处理，保存来源图像唯一 ID。

标签策略转换返回：`bce_targets[B,10]`、`bce_mask[B,10]`、`ce_targets[B,10]`、`ce_mask[B,10]`。同一分支只激活一种损失；无效 BCE 目标填有限值 0，CE 无效目标用约定 ignore_index，避免 NaN 乘零仍污染梯度。

模型返回 `logits[label_name]`，二分类头形状 `[B,1]`，MultiClass 头形状 `[B,3]`，状态顺序固定 `[negative, positive, uncertain]`。统一 `predict_proba` 总是返回 `[B,10]`：二分类用 sigmoid；三分类用 `sigmoid(z_positive - z_negative)`。

E4 有 28 个 logits，但对外仍只有十个阳性概率。所有病种可共存；仅三分类头内部使用 softmax，不对十个输出一起 softmax。

## 5. 配置接口样例（待实现，当前不可直接运行）

```yaml
experiment_id: E0_ignore
seed: 42
labels:
  - Enlarged Cardiomediastinum
  - Pneumothorax
  - Consolidation
  - Pneumonia
  - Edema
  - Cardiomegaly
  - Atelectasis
  - Lung Opacity
  - Pleural Effusion
  - No Finding
data:
  dataset: chexpert
  view_filter: frontal
  split_seed: 42
  split_ratios: [0.8, 0.1, 0.1]
  eval_policy: blank_zero_exclude_uncertain
  metadata_inputs: []
model:
  backbone: eva_x_s
  initialization: general_pretrained
labels_policy:
  default_uncertain: ignore
  per_label: {}
  blank: zero
  no_finding: original_one_vs_zero_blank
  soft_target: 0.5
loss:
  weighting: none
  weight_clip: [1.0, 20.0]
  reduction: mean_of_active_label_means
augmentation:
  preset: G1
  image_size: 224
training:
  optimizer: adamw
  learning_rate: 0.0001
  weight_decay: 0.01
  max_epochs: 10
  batch_size: 8
  accumulation_steps: 4
  amp: true
  num_workers: 0
selection:
  primary_metric: macro_auroc
  threshold_objective: macro_f1_per_label
```

学习率、轮数和增强强度是起步候选；在 smoke test 后一次性登记正式值，再统一跑全部 E 组。每次保存合并本机配置后的 resolved 配置，并检查未知键、非法策略、错误标签顺序，不能静默忽略拼写错误。

本机配置样例：

```yaml
paths:
  data_root: D:/Datasets/chest-xray/CheXpert-v1.0-small
  raw_csv: D:/Datasets/chest-xray/CheXpert-v1.0-small/train.csv
  split_manifest: outputs/splits/chexpert_v1/manifest.json
  pretrained_checkpoint: D:/Models/eva_x_small_patch16_merged520k_mim.pt
  output_root: outputs
```

YAML 合并优先级为 base -> experiment -> local -> CLI 显式覆盖；local 只允许覆盖路径和机器资源字段，不能悄悄改变策略、患者划分或评估规则。num_classes 从标签列表推导，不另维护第二份数字。

## 6. 按任务逐步开发与验收

### P0：环境、依赖、数据审计

- [ ] 拉取共享仓库前先检查 `git status`，保留本地改动；不要用 reset --hard。
- [ ] 运行已有的 `scripts/check_environment.py`，记录 Python、PyTorch、驱动、CUDA、显存与实际 import 错误。
- [ ] 将环境脚本区分基础必需库与模型可选库；xformers/medpy/libauc 缺失不能直接阻止纯 DenseNet baseline。EVA-X 启用何种算子则校验对应依赖。
- [ ] 核查 EVA-X 源码。保留 Git 原状，选择补齐有效子模块或独立依赖路径方式，并记录来源 commit；不假定只安装 timm 就能创建官方模型。
- [ ] requirements 若无法安装，先记录首个包冲突、Python/平台约束，再生成本机验证过的依赖组合；未经验证不宣称 CUDA 环境已经可用。
- [ ] 实现审计入口；统计每类 1/0/-1/blank、图像/患者/study 数、缺失路径、损坏文件、重复图、AP/PA 和字段覆盖率。
- [ ] 人工确认比赛指标、标签语义、允许的外部权重/数据和提交格式；未知规则写入运行记录，不编造。

验收：能够读取小样本并输出审计报告。所有正式训练前阻断路径错误、患者 ID 缺失与非法标签；疑似标签噪声只标记，不自动改写。

### P1：固定划分和 E0 工程 baseline

- [ ] 按患者划分并保存固定清单；训练/验证/测试患者交集为空，跨集合重复图检查通过。
- [ ] 先实现 Ignore、blank-zero、No Finding、masked BCE 和统一十概率接口；不要求先做专家模型。
- [ ] 保存每个有效分支平均损失，再对有效分支求平均。全批所有分支都被屏蔽时跳过更新并计数，不能制造 NaN。
- [ ] 实现 AMP 与梯度累积；最后不足累积步数的梯度窗口按实际步数归一化，不丢掉尾批。
- [ ] Windows DataLoader 入口使用 `if __name__ == "__main__":`，num_workers 从 0 开始，后续加速不改变样本清单。
- [ ] 用 32-64 张训练图跑一次小样本前向/反向和一轮验证，验证与测试不出现随机增强。
- [ ] checkpoint 包含模型、优化器、scheduler、AMP scaler、随机状态、epoch/step、配置、标签顺序和划分哈希；resume 恢复训练，finetune 只加载模型，两者区分。

验收：损失有限、被屏蔽目标无直接损失梯度、十概率范围在 [0,1]、同一图像确定性推理一致、恢复训练使用相同配置与患者清单。正式 E0 开始后锁定公共预算。

### P2：六种统一策略

| 实验 | 九个病变分支配置 | 修改位置 |
| --- | --- | --- |
| E0 | ignore | policies 生成掩码；BCE |
| E1 | zeros | policies 将 -1 变 0；BCE |
| E2 | ones | policies 将 -1 变 1；BCE |
| E3 | soft | policies 将 -1 变 0.5；BCE |
| E4 | multiclass | model 建九个三类头；policies/losses 启用 CE |
| E5 | self_trained | cache_teacher + policies 只替换 -1；BCE |

- [ ] 明确 0/1 不因 -1 策略变化；blank 和 No Finding 按公共规则处理。
- [ ] E0 教师用训练患者拟合、验证选 checkpoint，eval 模式和确定性预处理缓存训练 -1 概率；不使用测试患者生成训练目标。
- [ ] 缓存由 sample_id + 标签关联，含 teacher checkpoint 哈希、划分哈希和标签顺序。缺少目标概率或哈希不匹配时报错，不能靠行号合并。
- [ ] E5 学生从同一种通用初始化开始，不默认接着教师权重训练。每个新 seed 使用对应教师并记录额外成本。
- [ ] 聚合生成 `auroc_matrix.csv`、`auprc_matrix.csv`、`f1_matrix.csv`，均为十行六策略列；缺失或失败实验标为 NA，不能补 0。

No Finding 在六组中使用同一标签规则，但共享骨干训练不同，预测和分数可以不同。此前“六列固定同一分数”的说法不正确；它仍须逐组评估，但不能称为六种 No Finding 不确定策略对比。

F1 主报告每模型每标签在验证上拟合阈值；另保存固定 0.5 的对照。验证上选阈值后的 F1 是调参结果，不宣称独立测试性能。AUROC 主排序、AUPRC 辅助审查；不同指标冲突按预先登记目标处理。

### P3：混合头与三种子

- [ ] 从 E0-E5 验证结果提出 `per_label` 策略配置；差异小时不硬选偶然最高分，不根据 -1 比例直接指定策略。
- [ ] 实现按分支混合 BCE/CE/软目标的 E6，再完整训练。输出十概率接口保持一致。
- [ ] 前两套标签配置使用 42/2026/3407 复验，保存均值、标准差。复用已完成同配置的 seed 42，不重复计入。

### P4：增强与损失消融

| 系列 | 变量 | 保持固定 |
| --- | --- | --- |
| G0-G3 | G0 确定性预处理；G1 基础增强；G2 加轻微平移缩放；G3 再加水平翻转 | 标签配置、权重、划分、尺寸、预算 |
| B0/B1 | 九病变 blank->0 / ignore；No Finding 不变 | 增强、权重开关、划分、预算 |
| W0/W1 | 权重关闭 / 开启 | 标签配置、blank、增强、划分、预算 |
| L0/L1（可选） | 标签平滑关闭 / 开启 | 单独定义作用于哪些确定/不确定目标，不混称图像增强 |

- [ ] 每种处理使用三个相同种子配对；资源有限先用 42 初筛，再补候选种子，标注完整/未完整复验状态。
- [ ] B 系列中评估真值和掩码固定：弱标签主报告 blank->0、-1 排除；明确 0/1 子集只作固定补充视图。
- [ ] 权重从该组训练有效目标计算，Ignore 不计被屏蔽项；软目标用 sum(y) 与 sum(1-y) 质量，保存原始比值和截断值。
- [ ] W1 可预先采用 [1,20] 截断规则；这是项目候选，不是医学标准。无有效正/负目标时先核查分支，不静默继续。
- [ ] MultiClass 权重采用预先定义的 CE 状态权重；不能把 BCE pos_weight 传给三分类损失。
- [ ] G3 单独检查左右标记、方向元数据与变换是否相容；强裁剪不作为默认增强。

单因素消融各自与同一个明确 baseline 比较。多个胜出改动组合后必须再训练，不能把独立提升数值直接相加；如 B1 改变权重统计，披露这个关联。

### P5：元数据、分层评估与专家（可选）

- [ ] 先审计字段，而不是假定存在。CheXpert 常见 CSV 用 `Frontal/Lateral` 和 `AP/PA` 表示视角；ViewPosition、PatientOrientation 的实际可用性需核查，映射成内部统一字段并保留 Unknown。
- [ ] ViewPosition 是类别，不是阴/阳性；分组 AP、PA、Lateral、Unknown。PatientOrientation 按真实类别分组；两者不是同一个字段。
- [ ] 当前正面主线不含侧位，第一阶段没有 Lateral 专家可训练。扩展侧位先另登记数据范围、预算和比较基准。
- [ ] 对 Support Devices 做固定分组审计，保留 1/0/-1/blank 来源；如合并组需声明。报告标签不能作为推理时已知输入或专家路由真值。
- [ ] 输出各组阳性/阴性数、AUROC/AUPRC 和患者 bootstrap 区间；不足有效样本的组记 NA，不按“方差大”自动建专家。
- [ ] 只有测试推理可获取元数据时，才比较 M0 图像、M1 图像+视角、M2 图像+视角+方向；缺失用 Unknown，并测试元数据缺失时行为。
- [ ] 条件模型可用各字段 embedding 与图像特征拼接后进入任务头；不能把类别编号当成有大小关系的连续数值。
- [ ] 专家按原固定清单过滤，患者不重划分；AP/PA 子模型均从同一种通用初始化训练，训练参数变化记录为独立搜索成本。
- [ ] 在同一共同验证样本集上比较通用模型、视角路由专家、通用+专家融合；Unknown 使用预先指定通用回退。样本单位固定，不拿 study 多视图模型与单图模型混报。

### P6-P7：提升、冻结与提交

- [ ] 分辨率依次比较 224 与 320/384，核对骨干 patch/位置编码兼容性和预处理；显存不足只调整物理 batch，保持有效 batch 并记录。
- [ ] 集成使用逐图十概率；模型顺序、权重、TTA 和校准流程由验证集确定，小验证集优先简单平均，防止十类各自过拟合权重。
- [ ] 校准和阈值在最终集成/TTA 后拟合；最终测试只应用保存参数。AUROC 比赛提交连续分数，F1 比赛按官方格式使用冻结阈值。
- [ ] 逐类独立最大 F1 对应 macro F1 目标；micro F1 或其他比赛指标需要匹配整体优化方式，不能用错误目标选阈值。
- [ ] 如约束 Recall >= 90%，在满足约束的候选中选择最高阈值以尽量减少假阳性；不是选择最低阈值。验证样本太少时报告不稳定性。
- [ ] 高频错误样本只自动生成复核清单；只对有证据的训练标签做版本化修订。错误率高不构成删样本或改标签的依据。测试真值不依据模型结果修改。
- [ ] 冻结配置、checkpoint、阈值、校准、TTA、标签顺序与输入字段。最终只在内部锁定测试及兼容的外部测试上报告性能；按患者配对 bootstrap 计算区间。
- [ ] 预测文件对照 sample_submission 校验 ID、顺序、行数、缺失、范围和列名。锁定测试与排行榜反馈不继续用于调参；若测试揭示 bug 而修复，必须披露重复使用测试的事实。

## 7. 最低限度的行为验收用例

这些是需要开发端实现的测试规范，针对实际风险，不是已执行的测试：

| 输入/场景 | 必须满足的输出/行为 |
| --- | --- |
| 单病变原值 `[1,0,-1,blank]`，Ignore+blank-zero | BCE targets `[1,0,0,0]`，mask `[1,1,0,1]` |
| 同输入，Ones+blank-ignore | targets `[1,0,1,0]`，mask `[1,1,1,0]` |
| 同输入，Soft+blank-zero | targets `[1,0,0.5,0]`，mask 全有效 |
| No Finding `[1,0,blank]`，B0/B1 | targets `[1,0,0]`，mask 全有效；遇 -1 报告非法状态 |
| SelfTrained 教师概率 `[0.2,0.8,0.6,0.9]`，blank-zero | 仅 -1 替换，目标 `[1,0,0.6,0]` |
| 被屏蔽 BCE 位置改变 logit | 该位置无直接损失梯度，其他标签仍训练 |
| 训练与验证含同一患者或重复图跨集合 | 划分验收失败，正式训练阻断 |
| E4 或 E6 预测 | 形状 `[B,10]`，三类分支概率等于 sigmoid(正logit-负logit) |
| 某评估类没有阳性或没有阴性 | AUROC 为 NA，显示样本数，跨组使用一致有效类别 |
| 测试模式尝试拟合阈值或校准 | 明确拒绝，要求验证拟合的参数文件 |

验收完成后记录真实测试命令与结果，不能仅写“代码看起来没问题”。标签策略、混合头和患者隔离是应优先覆盖的高影响逻辑。

## 8. Windows 操作与计划中的命令接口

以下命令现在可以执行（路径改为本机已有项目）：

```powershell
cd D:\Projects\chest-xray-classification
git status --short
nvidia-smi
.\.venv\Scripts\Activate.ps1
python scripts/check_environment.py
```

如果没有 `.venv`，先独立创建环境并按 PyTorch 官方方式安装匹配驱动/Python 的 CUDA wheel，再安装适用项目依赖。不要把 Mac 虚拟环境复制到 Windows，也不要把完整 requirements 安装失败误报为训练代码故障。

以下接口是开发目标，脚本完成并验收后才能执行；每个命令支持 `--help`，失败返回非零退出码：

```powershell
python scripts/inspect_chexpert.py --local-config configs/local.windows.yaml
python scripts/split_patients.py --config configs/base.yaml --local-config configs/local.windows.yaml
python scripts/train.py --config configs/base.yaml --experiment configs/experiments/E0.yaml --local-config configs/local.windows.yaml --smoke-test
python scripts/train.py --config configs/base.yaml --experiment configs/experiments/E0.yaml --local-config configs/local.windows.yaml
python scripts/cache_teacher.py --checkpoint outputs/E0_ignore/seed42/best.pt --local-config configs/local.windows.yaml
python scripts/run_experiments.py --config configs/base.yaml --suite configs/experiments/round1.yaml --local-config configs/local.windows.yaml
python scripts/summarize_experiments.py --input-dir outputs --output-dir outputs/reports/round1
```

`round1.yaml` 只列 E0-E5 和 seed42；按 E0 -> 教师缓存 -> 其余组执行。已有成功运行核对配置/划分/checkpoint 哈希后才能复用。不要仅因输出目录存在就跳过，也不要静默覆盖另一个实验。

评估与阈值接口同样待实现：

```powershell
python scripts/evaluate.py --checkpoint outputs/E0_ignore/seed42/best.pt --split val --local-config configs/local.windows.yaml
python scripts/tune_thresholds.py --predictions outputs/E0_ignore/seed42/val_predictions.csv --objective macro_f1
python scripts/evaluate.py --checkpoint outputs/E0_ignore/seed42/best.pt --split test --thresholds outputs/E0_ignore/seed42/thresholds.json --local-config configs/local.windows.yaml
```

第三条仅展示最终测试接口，P1/P2 不执行；正式冻结时对最终方案执行。五类专家评估使用独立 dataset/split 标识，不复用内部 test 名称。

## 9. AI 与人工职责、Git 交接

Windows AI 可以根据本文编写模块、测试、批量实验和指标报告；训练手负责实际运行、磁盘和显存安排、异常反馈。人工负责账号/协议、比赛规则确认与最终提交，医学标签修订需要专业复核。AI 不把未运行的建议写成已经提升的结果。

默认分工仍为 Mac 代码手、Windows 训练手；Windows 端也可按本方案修改代码。在独立 `codex/` 分支上做小块变更，与 Mac 不同时编辑同一文件。每个交接注明角色和模块；遇到冲突保留双方改动，不强制覆盖。

修改环境先提交依赖说明；修改数据先提交审计与划分；修改策略先提交 policies/losses/tests；跑通 E0 后再扩展其他组。每次完成可验证的一块就更新 README 和 HANDOFF_LOG，不等所有实验结束才记录。

提交 Git：代码、公共配置、测试、文档、无患者明细的小型实验统计。留本地：数据、原始标签、split 患者清单、模型权重、伪标签缓存、逐图预测、包含患者信息的日志和 local 配置。实现阶段在 `.gitignore` 增加 `configs/local*.yaml`，不要把本机设置提交。

结果统一保存在 `outputs/<experiment_id>/seed<seed>/`，每次至少有 resolved 配置、环境版本、划分哈希、模型初始化/checkpoint 标识、训练日志、逐类指标和验证阈值。汇总包含每组样本覆盖率，不能只回传一个平均 AUC。

回传模板：

```text
日期：YYYY-MM-DD
身份：训练手（Windows 端代码执行与训练）
接收方：代码手
Git commit / 分支：
本次修改文件及原因：
实际执行命令：
数据版本 / 划分哈希 / seed：
实验号及完成状态：
AUROC / AUPRC / F1 与阈值来源：
峰值显存 / 耗时：
验收命令及结果：
报错原文及日志位置：
下一步：
```

## 10. 给 Windows 端 AI 的直接工作指令

> 你是本项目 Windows 训练端协作者。先阅读 README、HANDOFF_LOG 和本文件，检查本机 Git 状态、源码与环境，不覆盖他人改动。按 P0/P1 先实现数据审计、固定患者划分、十标签 U-Ignore baseline 和关键测试，使用配置隔离本机路径。先小样本验收，再正式跑 E0；E0 未通过前不开始六策略、元数据或专家训练。后续按 P2-P7 分阶段实现，所有实验保持固定划分与明确评估真值。每次改动在日志写明身份、文件、命令和真实验证结果；回传指标及异常，不能用模型预测直接修改医学标签，不能用锁定测试调参。完整六策略结果必须包含十行六列 AUROC/AUPRC/F1，No Finding 分数逐模型真实计算。

相关依据：[标签实验详细设计](chexpert-label-experiment-plan.md)、[论文与标签器核查](../research/chexpert-uncertainty-source-check.md)、[EVA-X 使用约定](../research/eva-x-usage-guide.md)。执行顺序与工程接口以本交接为准；标签语义与文献边界参照来源核查。
