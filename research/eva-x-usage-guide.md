# EVA-X 使用方法整理

更新时间：2026-09-20

## 1. 官方资源

- EVA-X 官方代码：[https://github.com/hustvl/EVA-X](https://github.com/hustvl/EVA-X)
- EVA-X 论文：[https://arxiv.org/abs/2405.05237](https://arxiv.org/abs/2405.05237)
- EVA-X-S 通用预训练权重：[https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim.pt](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim.pt)
- EVA-X-S CheXpert 微调权重：[https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim_chexpert_ft.pth](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim_chexpert_ft.pth)

## 2. EVA-X-S 有两种用法

### 用法 A：直接使用官方 CheXpert 微调模型

适合：

- 快速验证 EVA-X-S；
- 先检查模型推理流程；
- 任务正好是官方 CheXpert 五类任务。

官方 CheXpert 五类标签是：

```text
Atelectasis
Cardiomegaly
Consolidation
Edema
Pleural Effusion
```

下载 `eva_x_small_patch16_merged520k_mim_chexpert_ft.pth` 后，使用官方 classification 代码的 CheXpert 配置，将 `--finetune` 或 checkpoint 路径指向该文件，并增加 `--eval`。

注意：官方权重对应官方 CheXpert 五类设置，不能直接当作 14 类分类器使用。

### 用法 B：使用通用 EVA-X-S 权重重新微调

适合：

- 自己的比赛数据；
- 14 类 CheXpert 任务；
- 比赛标签与官方 CheXpert 五类不完全一致；
- 需要控制训练、验证和不确定标签策略。

下载 `eva_x_small_patch16_merged520k_mim.pt`，加载到 EVA-X-S backbone，然后替换分类头：

```text
EVA-X-S backbone
        ↓
Linear(..., num_classes)
        ↓
BCEWithLogitsLoss
```

其中 `num_classes` 应与比赛标签数一致：

- 官方 CheXpert 竞赛任务：5；
- CheXpert 全部观察结果：14；
- 其他比赛：以比赛标签表为准。

## 3. 官方代码的安装方式

官方分类代码使用较老的训练环境，主要依赖：

```text
Python 3.8
PyTorch 1.12.1
torchvision 0.13.0
timm 0.5.4
DeepSpeed 0.6.5
Apex
xFormers
```

官方流程：

```bash
conda create --name evax python=3.8 -y
conda activate evax
git clone https://github.com/hustvl/EVA-X.git
cd EVA-X/classification
pip install -r requirements.txt
```

官方还要求安装 Apex 和 xFormers。

## 4. CheXpert 数据准备

### 4.1 下载数据

建议从 Stanford AIMI 官方页面申请和下载 CheXpert：

- [Stanford AIMI CheXpert](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays)
- [CheXpert 竞赛页面](https://stanfordmlgroup.github.io/competitions/chexpert/)

EVA-X 官方 README 给出了 Kaggle CheXpert small 镜像，但正式实验建议保留官方原始目录结构，不要只下载零散图片。

### 4.2 目录结构

解压后需要保留类似结构：

```text
CheXpert-v1.0-small/
├── train.csv
├── valid.csv
├── train/
│   ├── patient00001/
│   │   └── study1/
│   │       └── view1_frontal.jpg
└── valid/
    ├── patient64540/
    │   └── study1/
    │       └── view1_frontal.jpg
```

### 4.3 EVA-X 官方数据准备

官方 `classification/datasets/prepare_dataset.py` 会：

1. 复制 CheXpert 图片；
2. 复制 `train.csv` 和 `valid.csv`；
3. 创建 `classification/datasets/data_splits/chexpert/`；
4. 检查训练集和验证集图像数量。

官方代码默认生成：

```text
classification/datasets/
├── chexpert/
│   ├── train.csv
│   ├── valid.csv
│   ├── train/
│   └── valid/
└── data_splits/
    └── chexpert/
        ├── train.csv
        └── valid.csv
```

## 5. 官方 CheXpert 配置的关键参数

EVA-X 官方 `classification/train_files/eva_x/chexpert/vit_s.sh` 的核心设置包括：

```text
dataset: chexpert
model: eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE
input_size: 224
nb_classes: 5
loss: BCEWithLogitsLoss
batch_size: 256
epochs: 60
```

官方配置使用 4 张 RTX 3090，并将有效 batch size 设得很大。RTX 4060 不应直接照搬 `batch_size=256`。

## 6. RTX 4060 建议配置

建议从以下参数开始：

```text
input_size: 224
batch_size: 8 或 16
accum_iter: 2 或 4
num_workers: 2
AMP: 开启
模型: EVA-X-S
```

显存不足时按以下顺序调整：

1. batch size 降到 4；
2. 使用梯度累积；
3. 暂时冻结 backbone，只训练分类头；
4. 再考虑减小输入尺寸。

不建议一开始将输入尺寸提高到 448 或 512。先在 224x224 上完成稳定实验，再比较高分辨率收益。

## 7. 5 类和 14 类的区别

### 5 类任务

直接使用官方 CheXpert 配置即可：

```text
nb_classes = 5
train_cols =
[
  "Atelectasis",
  "Cardiomegaly",
  "Consolidation",
  "Edema",
  "Pleural Effusion"
]
```

### 14 类任务

不能只把 `nb_classes=5` 改成 `14`。还需要同步修改：

1. `CheXpert` 数据集类中的 `train_cols`；
2. 不确定标签处理逻辑；
3. 分类头输出维度；
4. 验证指标计算；
5. checkpoint 加载时的 head 处理；
6. 每类阈值和类别权重。

通用的 14 类列表应以当前 CheXpert `train.csv` 的列定义和项目实验协议为准，不要混用其他数据集的标签顺序。

## 8. 不确定标签处理

CheXpert 的标签通常包含：

```text
1   阳性
0   阴性
-1  不确定
空  未提及
```

EVA-X 官方 CheXpert 数据读取逻辑默认对五类标签采用：

```text
Edema / Atelectasis:
    -1 → 1
    空 → 0

Cardiomegaly / Consolidation / Pleural Effusion:
    -1 → 0
    空 → 0
```

这属于一种任务策略，不是唯一正确答案。你们的实验应至少记录：

- U-Zero；
- U-One；
- U-Ignore；
- 自训练或软标签策略。

## 9. 训练与评估建议

建议实验顺序：

### 实验 1：检查官方权重

- 下载 EVA-X-S CheXpert 微调权重；
- 在官方验证集上运行评估；
- 确认图片路径、CSV 路径、标签顺序和输出维度正确。

### 实验 2：通用 EVA-X-S 微调

- 加载通用 EVA-X-S 预训练权重；
- 使用比赛训练集；
- 训练新的分类头；
- 保存最佳验证集 checkpoint。

### 实验 3：标签策略对比

- 固定模型和数据划分；
- 只改变不确定标签策略；
- 比较每类 AUROC、AUPRC、F1 和 Recall。

### 实验 4：集成

- EVA-X-S；
- DenseNet121 baseline；
- 其他胸片预训练模型；
- 在验证集上做逐类概率加权。

## 10. 对当前项目的最终建议

当前项目使用 RTX 4060，并且仓库计划支持 CheXpert 多标签分类。建议不要直接运行 EVA-X 官方的 4 卡 Linux shell 脚本，而是：

1. 下载通用 EVA-X-S 权重；
2. 在现有 PyTorch 训练框架中接入 EVA-X-S backbone；
3. 先跑 5 类比赛任务；
4. 如果项目确实要求 14 类，再扩展数据集读取器和分类头；
5. 保留 DenseNet121 作为可复现 baseline；
6. 在固定患者级划分上比较两者。

## 11. 最容易踩的坑

- 官方 CheXpert EVA-X 配置是 5 类，不是 14 类；
- 官方脚本面向 4 张 RTX 3090，不适合直接用于 RTX 4060；
- EVA-X-S 通用预训练权重和 CheXpert 微调权重不是同一个文件；
- 加载 5 类 checkpoint 到 14 类模型时，必须丢弃并重新初始化分类头；
- CheXpert 必须按患者划分，不能随机按图片泄漏患者；
- 不要把论文中的 mAUC 直接当成自己数据上的保证分数；
- 外部预训练权重是否允许使用，要先核对比赛规则。

## 官方代码位置

```text
EVA-X/
└── classification/
    ├── README.md
    ├── train.py
    ├── datasets/
    │   └── prepare_dataset.py
    ├── utils/
    │   ├── dataloader_med.py
    │   └── datasets.py
    └── train_files/
        └── eva_x/
            └── chexpert/
                ├── vit_ti.sh
                └── vit_s.sh
```
