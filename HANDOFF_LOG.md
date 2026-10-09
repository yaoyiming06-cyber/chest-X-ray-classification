# 交接日志

## 2026-10-08

- 身份：代码手
- 接收方：训练手
- 改动摘要：在 EVA-X `classification/competition/` 新增比赛专用 10 类标签及按类别记录的 CheXpert 不确定标签策略、标准清单 Dataset、目录/CSV 检查器、Study 级 logits 聚合、严格 Macro-AUC/Macro-F1 验证和内部概率预测命令；训练入口新增 `--dataset competition`，保持 CheXpert 读取器独立。
- 验证结果：标签映射及比赛组件单测通过；`train.py` 比赛参数解析、`predict.py --help`、Python 编译均通过。当前本机缺可选 Triton 优化包，不影响入口导入。
- 训练手下一步：比赛数据预计 2026-10-15 开放；先运行 `python competition/inspect_data.py <data_root>` 并核对标签表、图片与 Study/Subject 关系，再将真实文件转换为 `classification/competition/README.md` 所述内部清单。确认首轮轻量随机增强（小角度仿射及亮度/对比度）符合数据特征后再正式训练。
- 注意事项：组委会已确认 EVA-X 通用自监督预训练权重可用。U-MultiClass、U-SelfTrained、U-Ignore 尚需接入对应的训练损失/自训练阶段；当前遇到这些类别的 `-1` 会明确报错。真实图片布局、长表缺失类别语义和官方提交模板尚未确认；不要在确认前生成最终 submission.csv。

## 2026-09-20

- 身份：代码手
- 接收方：训练手
- 改动摘要：把 README 扩展为 Windows 训练端项目说明，明确 CheXpert -> NIH ChestX-ray14 -> MIMIC-CXR-JPG -> VinDr-CXR -> Shenzhen/Montgomery 的阶段路线，并补充环境、数据管理、baseline、实验记录和下一步执行清单。
- 验证结果：仅文档改动，无训练代码需要运行。
- 训练手下一步：拉取最新 README，确认 Windows RTX 4060、CUDA 和 PyTorch 环境，然后准备 CheXpert 数据集检查。
- 注意事项：当前仓库仍没有训练脚本；不要把 `.venv/`、数据集、模型权重和完整训练输出提交到 Git。

## 2026-09-20

- 身份：代码手
- 接收方：训练手
- 改动摘要：建立双端协作说明，新增 README 和交接日志。明确 Mac / Codex 端负责编码与文档，Windows / RTX 4060 端负责训练与结果反馈。
- 验证结果：仅文档改动，无训练代码需要运行。
- 训练手下一步：在 Windows 上拉取项目，创建 CUDA 版 PyTorch 环境，并确认 `nvidia-smi` 与 `torch.cuda.is_available()` 正常。
- 注意事项：`.venv/`、`data/`、`checkpoints/`、模型权重和训练输出不进 Git。

## 项目记忆：图像预处理与训练增强

- 当前 EVA-X-S/CheXpert 训练配置中，默认 `build_timm_transform=False` 时，训练集主要执行 `Resize -> CenterCrop -> ToTensor -> Normalize`，随机增强较少。
- 后续修改 `classification/utils/datasets.py`、训练 transform 或相关训练参数时，先提醒训练手：确认是否需要启用或补充适合胸片的轻量随机增强。
- 训练集可以使用轻微随机裁剪、水平翻转等增强；验证集和测试集只使用确定性的 Resize/CenterCrop/Normalize，不使用随机增强。
- 胸片增强必须保持医学含义，避免默认加入垂直翻转、大角度旋转、过强裁剪、强颜色扰动、Solarization 或强模糊。

## 项目记忆：第一版标签处理策略

- 本项目第一版采用 EVA-X CheXpert 风格的直接标签映射，不使用 mask：Dataset 返回 `image, target`，训练使用 `BCEWithLogitsLoss`。
- 对 CheXpert 或 `findings_fixed.json` 中已有 EVA 官方 5 类的映射保持一致：`Atelectasis`、`Edema` 的 `-1 -> 1`；`Cardiomegaly`、`Consolidation`、`Pleural Effusion` 的 `-1 -> 0`；`null -> 0`。
- 10 类任务新增的 `Enlarged Cardiomediastinum`、`Pneumothorax`、`Pneumonia`、`Lung Opacity`、`No Finding` 没有 EVA 官方 5 类基线映射，第一版暂按 `-1 -> 0`、`null -> 0` 处理，并在实验记录中标明这是项目扩展约定。
- 如果比赛训练集标签已经是完整的 `0/1`，直接使用原标签，不额外套用 CheXpert 的 `-1` 映射；比赛规则也优先于本地 CheXpert baseline。

## 交接模板

```text
## YYYY-MM-DD

- 身份：代码手
- 接收方：训练手
- 改动摘要：
- 验证结果：
- 训练手下一步：
- 注意事项：
```
