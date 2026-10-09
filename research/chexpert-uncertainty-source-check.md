# CheXpert 不确定标签：一手来源核查

核查日期：2026-10-08。范围仅含原论文、官方页及官方标签器源码，不推定本项目十标签的固定最优处理。

## 1. 五种方法的具体定义

以下定义见论文 [Uncertainty Approaches](https://arxiv.org/html/1901.07031v1#Sx4.SSx1)，官方页的 [Leveraging Uncertainty Labels](https://stanfordmlgroup.github.io/competitions/chexpert/) 与其一致。论文以 `u` 表示不确定，发布标签编码为 `-1`。

| 方法 | 训练与推理定义 |
| --- | --- |
| U-Ignore | 对某样本中标为 `u` 的观察项屏蔽二元交叉熵损失；其他观察项仍参与训练。不是删除整个样本。 |
| U-Zeros | 将全部 `u` 映射为 `0`。正文和官方页也拼作 U-Zeroes。 |
| U-Ones | 将全部 `u` 映射为 `1`。 |
| U-SelfTrained | 先用 U-Ignore 训练至收敛，用该模型的概率预测替换不确定标签，再以各观察项二元交叉熵的均值训练；原有 `0/1` 不替换。替换的是连续概率软标签，不是阈值化的硬标签。 |
| U-MultiClass | 每个观察项分别预测 `0/1/u` 三类，三类概率之和为 1，以各观察项多类交叉熵的均值训练。测试时仅对正、负两类做 softmax，输出 `p1/(p0+p1)`，不是直接输出含不确定类的三类 softmax 中的 `p1`。 |

## 2. 五个评估类的选择与建议边界

原论文 [Table 3](https://arxiv.org/html/1901.07031v1#Sx3.T3)、[Test Results 开头](https://arxiv.org/html/1901.07031v1#Sx6) 和[官方 baseline 描述](https://stanfordmlgroup.github.io/competitions/chexpert/)支持以下**原实验的最终选择**，可作为候选策略参考：

| 评估类 | 最终选择 | 表 3 验证 AUROC |
| --- | --- | --- |
| Atelectasis | U-Ones | 0.858 |
| Cardiomegaly | U-MultiClass | 0.854 |
| Consolidation | U-SelfTrained | 0.939 |
| Edema | U-Ones | 0.941 |
| Pleural Effusion | U-MultiClass | 0.936 |

- 比较基于 200 个 study、200 位患者；三位放射科医生的标签先二值化再多数投票。每种策略运行三次，每次按五类平均 AUC 选十个 checkpoint，共三十个 checkpoint 集成。因此表中排序对应这个验证集与实验流程。[验证集及选择过程](https://arxiv.org/html/1901.07031v1#Sx5)
- 显著性证据是 Atelectasis 的 U-Ones 优于 U-Zeros、Cardiomegaly 的 U-MultiClass 优于 U-Ignore；Consolidation、Edema、Pleural Effusion 的最好与最差方法未发现显著差异，不能写成五类都已证明唯一最优。[Results](https://arxiv.org/html/1901.07031v1#Sx5.SSx2.SSSx3)
- **原文内部差异**：分析段称 Consolidation 的 U-Zeros 最好，但表 3 中 U-SelfTrained 为 0.939、U-Zeros 为 0.932；最终模型段和官方页均选择 U-SelfTrained。引用最终选择应采用后者，同时保留该差异，不能只据分析段写 U-Zeros 为确定最优。[Analysis](https://arxiv.org/html/1901.07031v1#Sx5.SSx2.SSSx4)

## 3. blank 与 No Finding

- **blank = 未提及**，`0 = 明确阴性`，`-1 = 不确定`，`1 = 阳性`。blank 不等于报告明确否认病变，也不等于不确定。标签器未提及项生成 `np.nan`，随后写入 CSV；编码由常量文件确认。[论文 Mention Aggregation](https://arxiv.org/html/1901.07031v1#Sx2.SSx2.SSSx3)、[官方标签说明](https://stanfordmlgroup.github.io/competitions/chexpert/)、[聚合源码][aggregate]、[编码常量][constants]、[CSV 写出][write]
- **No Finding 的论文定义**：没有病理观察项被分类为阳性或不确定时，赋值 `1`；这是报告标签聚合规则，不是影像已被独立确认完全正常。Support Devices 不属于排除 No Finding 的病理项。[论文定义](https://arxiv.org/html/1901.07031v1#Sx2.SSx2.SSSx3)、[聚合源码][aggregate]
- **源码细节**：`aggregate()` 检查所有非 Support Devices 注释是否为阳性或不确定，随后跳过直接写入 No Finding 注释；只有条件成立才写 No Finding=`1`，否则该列为缺失。`phrases/mention/no_finding.txt` 还收录 emphysema、scoliosis 等异常词，匹配到的阳性/不确定注释也可阻止 No Finding。因此不能将它简化成“所选十标签全为 0”或“所选十标签全空白”的补集。[聚合源码][aggregate]、[词表][no-finding-phrases]、[提取源码][extract]
- **未查证的训练预处理**：本次三类来源明确了 blank 的标签语义，但未找到明确交代原模型如何把 blank 纳入训练损失的具体规则。U-Zeros 的定义只规定 `u -> 0`，不能把它当作 `blank -> 0` 的直接证据；原标签器也不是图像模型训练代码。

## 4. 十标签评估的证据限制

原论文训练输出覆盖十四个观察项，但不确定策略的验证 AUROC 比较与最终模型选择只针对上述五个 competition tasks；标签器的十四项报告抽取、否定和不确定检测 F1 属于 NLP 标签质量评估，不能替代十类图像分类策略比较。[Model](https://arxiv.org/html/1901.07031v1#Sx4)、[五类评估过程](https://arxiv.org/html/1901.07031v1#Sx5.SSx2.SSSx1)、[标签器评估](https://arxiv.org/html/1901.07031v1#Sx3)

因此这三类来源**没有证明全部十标签的固定最优不确定处理**，也没有证明未列入五类比较的标签应统一采用某种映射。若用于十标签任务，以上选择只能作为有出处的候选；其余标签、blank 预处理及最终策略需由该任务的标签定义和验证结果决定。这是本次证据边界，不是声称其他研究不存在。

## 5. 来源查证状态

- 成功：原论文 [arXiv 摘要页](https://arxiv.org/abs/1901.07031) 与 [HTML 全文](https://arxiv.org/html/1901.07031v1)，版本为 v1，提交于 2019-01-21；[官方页](https://stanfordmlgroup.github.io/competitions/chexpert/)；[官方标签器仓库](https://github.com/stanfordmlgroup/chexpert-labeler) README、常量、聚合、提取、词表及 CSV 写出代码。源码链接固定到核查时 master 的提交 `44ddeb363149aa657296237f18b5472a73c1756f`。
- 限制：PDF 已下载到 `/tmp`，但当前环境无 `pdftotext`，未通过 PDF 文本独立复核；论文结论来自可访问的 HTML 全文。未运行标签器或图像训练实验，未核验本项目十标签数据。

[aggregate]: https://github.com/stanfordmlgroup/chexpert-labeler/blob/44ddeb363149aa657296237f18b5472a73c1756f/stages/aggregate.py
[constants]: https://github.com/stanfordmlgroup/chexpert-labeler/blob/44ddeb363149aa657296237f18b5472a73c1756f/constants/constants.py
[write]: https://github.com/stanfordmlgroup/chexpert-labeler/blob/44ddeb363149aa657296237f18b5472a73c1756f/label.py
[no-finding-phrases]: https://github.com/stanfordmlgroup/chexpert-labeler/blob/44ddeb363149aa657296237f18b5472a73c1756f/phrases/mention/no_finding.txt
[extract]: https://github.com/stanfordmlgroup/chexpert-labeler/blob/44ddeb363149aa657296237f18b5472a73c1756f/stages/extract.py
