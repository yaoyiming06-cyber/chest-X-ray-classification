# 胸部 X 光片建模分类

## 0. 给 Windows 训练端的话

本项目采用双端协作：

```text
Mac / Codex：代码手
  负责写代码、改配置、维护文档、提交 Git 改动

Windows / RTX 4060：训练手
  负责配置 CUDA、下载数据集、运行训练、记录指标和反馈问题
```

当前确定的总路线是：

```text
CheXpert
  ↓ 跑通第一版多标签分类 baseline
NIH ChestX-ray14
  ↓ 验证第二个公开多标签数据集和跨数据集适配
MIMIC-CXR-JPG
  ↓ 在允许使用外部数据时扩大训练规模
VinDr-CXR
  ↓ 做病灶定位、Grad-CAM 和模型可信性验证
Shenzhen / Montgomery
  ↓ 做结核方向的小规模外部泛化测试
```

当前仓库还没有训练脚本和模型代码。现在处于：

```text
项目结构和协作方式已确定
Python 依赖已准备
数据集路线已确定
等待 Windows 训练端确认 CUDA 环境
```

## 1. 项目目标

目标是建立一个胸部 X 光片计算机辅助诊断（CAD）实验流程，完成：

1. 胸片图像读取与预处理。
2. 多标签疾病分类。
3. 类别不平衡处理。
4. 患者级训练集、验证集划分。
5. AUROC、AUPRC、F1、召回率等指标评估。
6. 分类阈值调优。
7. 错误样本和模型关注区域分析。
8. 跨数据集、跨医院分布的泛化测试。

第一版不追求复杂模型，先完成一个可复现、可验证、能正常提交结果的 baseline。

## 2. 已确定的数据集路线

### 2.1 CheXpert：第一主线，先用它练手

CheXpert 是 Stanford 发布的大型胸片数据集，包含约 224,316 张胸片和 65,240 名患者，提供 14 类胸部观察标签。标签来自放射科报告自动抽取，并包含阳性、阴性、不确定等状态。

官方来源：

- [Stanford AIMI CheXpert](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays)
- [Stanford ML Group CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [CheXpert 论文](https://arxiv.org/abs/1901.07031)

在本项目中的用途：

```text
主训练数据集
第一版多标签分类 baseline
学习 pos_weight 和不确定标签处理
建立统一训练、验证、预测流程
```

第一阶段先做：

- 14 个观察类别的多标签分类。
- 预训练 DenseNet121 baseline。
- `BCEWithLogitsLoss`。
- 根据训练集统计每个类别的 `pos_weight`。
- 使用 AUROC、AUPRC 和 F1 评估。
- 在验证集上调每个类别的分类阈值。

注意事项：

- 不能把同一患者的图像随机分到训练集和验证集。
- CheXpert 的不确定标签不能不加说明地直接当成普通正样本。
- 第一版可以先固定一种策略，再做不确定标签策略对比。
- 训练和验证划分必须固定并保存，后续实验不能随意更换。

### 2.2 NIH ChestX-ray14：第二主线，做快速对照

NIH ChestX-ray14 包含约 112,120 张正面胸片、30,805 名患者和 14 类疾病标签，图像主要以 PNG 形式提供。

官方来源：

- [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [Google Cloud NIH 数据集说明](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest)
- [ChestX-ray8 论文](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_CVPR_2017_paper.html)

在本项目中的用途：

```text
验证代码能否适配第二个多标签数据集
和 CheXpert 做标签、图像分布、泛化能力对照
在 CheXpert 下载或处理受阻时作为备用练手数据
```

注意事项：

- 标签主要由报告文本自动挖掘，存在噪声。
- 不要把 NIH 的结果直接和 CheXpert 的结果当成同一标准比较。
- 应分别保存数据集名称、标签映射和验证结果。
- 先完成 CheXpert baseline，再迁移到 NIH。

### 2.3 MIMIC-CXR-JPG：第三阶段，扩大训练规模

MIMIC-CXR-JPG 是 PhysioNet 上的大规模胸片数据集，包含约 377,110 张 JPG 图像和约 227,827 份影像研究/报告关联信息，提供从报告中得到的结构化标签。

官方来源：

- [PhysioNet MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- [MIMIC-CXR 论文](https://physionet.org/content/mimic-cxr/2.1.0/)
- [MIMIC-CXR-JPG 论文](https://arxiv.org/abs/1901.07042)

在本项目中的用途：

```text
扩大训练数据
做预训练或联合训练
测试模型在不同医院来源上的泛化能力
```

访问和工程要求：

- 需要按照 PhysioNet 要求完成账号、培训和数据使用流程。
- 需要预留明显大于当前 Mac 的磁盘空间。
- 标签体系需要和 CheXpert 或比赛标签进行映射。
- 第一阶段不使用 MIMIC，避免一开始把数据访问和工程复杂度引入项目。

### 2.4 VinDr-CXR：第四阶段，做定位和可信性验证

VinDr-CXR 提供约 18,000 张胸片，包含放射科医生对局部异常和全局疾病的标注，适合分类、病灶定位和检测研究。

官方来源：

- [PhysioNet VinDr-CXR](https://physionet.org/content/vindr-cxr/)
- [VinDr-CXR 论文信息](https://physionet.org/content/vindr-cxr/1.0.0/)

在本项目中的用途：

```text
检查模型是否真正关注病灶区域
做 Grad-CAM 可视化
利用异常框扩展到病灶检测
分析分类模型的假阳性和假阴性
```

注意事项：

- VinDr 的标签体系不完全等同于 CheXpert。
- 不能简单把不同数据集的同名标签直接拼接。
- 先完成分类模型，再做定位分析。
- 该数据集更适合验证和扩展，不是当前第一版的主训练集。

### 2.5 Shenzhen / Montgomery：第五阶段，做外部泛化测试

Shenzhen 和 Montgomery 是规模较小的结核相关胸片数据集，可用于测试模型在不同医院、设备和人群分布下的表现。它们还适合做肺野分割、肺部裁剪和结核方向的补充实验。

官方来源：

- [NLM Tuberculosis Chest X-ray Datasets](https://www.lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayDatasets.html)

在本项目中的用途：

```text
外部验证
结核方向的补充测试
跨医院泛化检查
肺野裁剪或分割实验
```

注意事项：

- 数据量小，不适合作为主训练集。
- 不能用它们的测试结果代替 CheXpert 主实验结果。
- 结果重点看泛化趋势和失败案例，不要过度解读单个指标。

## 3. 当前训练路线

### 阶段 A：Windows 环境确认

训练手先在 Windows PowerShell 中确认：

```powershell
nvidia-smi
```

应能看到 NVIDIA RTX 4060 和显存信息。

进入项目后确认 Python 环境：

```powershell
cd D:\Projects\chest-xray-classification
.\.venv\Scripts\Activate.ps1
python --version
```

安装 CUDA 版 PyTorch 时，以 PyTorch 官方安装页面生成的命令为准：

- [PyTorch Start Locally](https://pytorch.org/get-started/locally/)

验证 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期：

```text
True
NVIDIA GeForce RTX 4060
```

### 阶段 B：获取和检查 CheXpert

数据集放在 Windows 本地 SSD，不放进 Git。建议：

```text
D:\Datasets\chest-xray\CheXpert\
```

下载后先不要立刻训练，先完成：

1. 检查图像文件是否完整。
2. 检查 `train.csv` 或标签文件是否可读取。
3. 检查图像路径是否能和标签表对应。
4. 统计 14 个标签的阳性、阴性、不确定和缺失数量。
5. 检查患者 ID 是否存在。
6. 输出一份数据统计结果。

建议项目内使用配置文件记录本机路径，例如：

```yaml
dataset: chexpert
data_root: "D:/Datasets/chest-xray/CheXpert"
image_size: 224
num_classes: 14
```

不要把 Windows 的绝对路径硬编码到通用代码中；路径放在本地配置或命令行参数里。

### 阶段 C：第一版 baseline

第一版建议固定为：

```text
模型：预训练 DenseNet121
任务：14 标签多标签分类
输入：224 x 224
损失：BCEWithLogitsLoss
不平衡：每个类别单独计算 pos_weight
优化器：AdamW 或 Adam
设备：CUDA
验证：患者级固定验证集
指标：每类 AUROC、AUPRC、F1、召回率
```

多标签输出方式：

```text
模型输出：14 个 logits
训练损失：直接把 logits 传给 BCEWithLogitsLoss
预测概率：sigmoid(logits)
最终标签：使用验证集调好的每类阈值
```

不要使用：

```text
softmax
统一把所有类别阈值固定为 0.5
先 sigmoid 再传给 BCEWithLogitsLoss
随机按图片划分训练集和验证集
```

### 阶段 D：实验记录

每次训练至少记录：

```text
实验编号
数据集和数据版本
训练/验证划分版本
模型名称和预训练权重
输入尺寸
batch size
学习率
loss 类型
pos_weight 计算方式
不确定标签处理方式
训练轮数
最佳 epoch
每类 AUROC / AUPRC / F1 / Recall
总体验证指标
checkpoint 路径
异常或报错
```

第一版不要求追求排行榜分数，先保证：

```text
能训练
能验证
能保存模型
能加载模型
能生成预测
能复现实验
```

## 4. Windows 项目环境

Mac 和 Windows 不能共用同一个 `.venv`。两个系统分别创建自己的虚拟环境。

Windows：

```powershell
cd D:\Projects\chest-xray-classification
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

验证常用库：

```powershell
python -c "import torch, torchvision, pandas, sklearn, cv2, timm, albumentations; print('imports ok')"
```

RTX 4060 初始参数建议：

```text
输入尺寸：224 x 224
batch size：8 起步，显存允许再试 16
num_workers：0 起步，确认稳定后再试 2 或 4
AMP：CUDA 混合精度
```

如果出现显存不足：

1. 降低 batch size。
2. 降低输入尺寸。
3. 开启混合精度。
4. 再考虑梯度累积。

## 5. 推荐目录

```text
chest-xray-classification/
├── README.md
├── HANDOFF_LOG.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── chexpert_baseline.yaml
├── src/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── metrics/
│   └── utils/
├── scripts/
│   ├── inspect_chexpert.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── notebooks/
├── data/              # 本地数据，不进 Git
├── checkpoints/       # 本地模型，不进 Git
└── outputs/           # 本地日志和预测，不进 Git
```

当前实际仓库还没有 `src/`、`scripts/` 和训练代码。目录是后续实现目标，不代表这些文件已经存在。

## 6. 双端 Git 协作

### 代码手：Mac / Codex

代码手负责：

1. 编写和修改训练代码。
2. 更新配置和 README。
3. 在 Mac 上做语法、导入和小样本检查。
4. 在 `HANDOFF_LOG.md` 写交接摘要。
5. 提交并推送代码。

```bash
git add .
git commit -m "describe the change"
git push
```

### 训练手：Windows / RTX 4060

训练手负责：

1. 拉取最新代码。
2. 在 Windows 本地准备数据集。
3. 使用 CUDA 环境运行训练。
4. 保存训练日志和结果。
5. 向代码手反馈指标、报错和改进建议。

```powershell
git pull
```

训练手不需要把数据集、模型权重或完整训练输出推送到 Git。

## 7. 数据和文件管理

不进入 Git：

```text
.venv/
data/
checkpoints/
outputs/
*.pt
*.pth
*.ckpt
```

进入 Git：

```text
训练代码
配置文件
README
交接日志
requirements.txt
小型测试脚本
```

Windows 的完整数据建议放在：

```text
D:\Datasets\chest-xray\
```

Mac 只保留小样本或不保存完整数据集。训练结果可以把关键指标和必要的小文件反馈到日志，但不要直接提交大模型权重。

## 8. 当前明确的下一步

训练手按下面顺序执行：

1. `git pull` 获取本 README。
2. 确认 `nvidia-smi` 能识别 RTX 4060。
3. 创建 Windows `.venv`。
4. 安装 CUDA 版 PyTorch 和 `requirements.txt`。
5. 执行 CUDA 和库导入验证。
6. 申请/下载 CheXpert，并确认许可和访问流程。
7. 将 CheXpert 放到 Windows 本地 SSD。
8. 先做数据检查，不立即训练。
9. 把数据统计结果和环境验证结果反馈给代码手。
10. 等待代码手提交第一版 `inspect_chexpert.py` 和训练 baseline。

## 9. 交接规则

代码手每次上交改动，都要在 [HANDOFF_LOG.md](HANDOFF_LOG.md) 新增：

```text
## YYYY-MM-DD

- 身份：代码手
- 接收方：训练手
- 改动摘要：
- 验证结果：
- 训练手下一步：
- 注意事项：
```

训练手反馈时建议包含：

```text
## YYYY-MM-DD

- 身份：训练手
- 接收方：代码手
- 环境结果：
- 数据结果：
- 训练结果：
- 报错信息：
- 建议修改：
```

每次反馈都尽量写清楚实验编号、命令、指标和错误原文，避免只写“跑不了”或“效果不好”。
