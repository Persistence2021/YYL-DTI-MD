
# 聚焦精神类疾病的药物靶标相互作用预测模型

**2020200730**                       
**姚玉龙**
---
## 项目架构

![](https://wy-static.wenxiaobai.com/chat-doc/094ef4c56a74b88f7f6083945edc6d29-image.png)

- **Data**: 项目使用的数据集和最终产生的结果；
- **drug_target_encoder**: 计算药物分子和蛋白靶标结构的表征方式；
- **hyper-dti-references**: 本次研究参考的文献代码；
  - 代码库链接：[HyperPCM: Robust task-conditioned modeling of drug-target interactions](https://github.com/ml-jku/hyper-dti)
- **scripts**: 完成DTI预测的机器学习代码；
- **md_target_info.csv**: UniProt完成检索的靶标信息，与 `source.csv` 一致，提到外部方便查阅。

---

## 数据部分（Data）

![](https://wy-static.wenxiaobai.com/chat-doc/713925de637116371da4db5d5ef3876b-image.png)

- **checkpoints**: 计算 MolBert 分子表征使用的模型权重文件（`last.ckpt`）；
  - CDDD 分子表征方法因 TensorFlow 模型加载权重文件出错（环境配置问题）被放弃。
- **Mental_disorder**: 完成精神类疾病筛选的数据集，具体文件如下：
  - `data.pickle`: HyperPCM 文献中的 Lenselink 数据集文件；
  - `source.csv`: 筛选整合的精神类疾病药物信息（共 1889 条）；
  - 字段示例：`Entry`, `Protein names`, `Organism`, `Sequence` 等。
- **pictures**: 研究中绘制的所有图片；
- **processed**: 未筛选数据集的实验表征文件；
- **raw**: 原始数据集（从 TDC 平台获取）；
- **results**: 代码运行后保存的所有结果。

---

## 精神类疾病（Mental_disorder）

![](https://wy-static.wenxiaobai.com/chat-doc/a608e2cc416232e747121ec521d01e38-image.png)

- **`_md` 后缀**: 表示数据集已完成筛选；
- **Lenselink 数据集**：因 Pandas 库版本冲突，改用 CSV 文件；
- **processed**: 项目使用的编码文件。

---

## 药物分子和靶标结构的编码代码（drug_target_encoder）

![](https://wy-static.wenxiaobai.com/chat-doc/2485e9f1b3aa254a83a2c6326028adf5-image.png)

### pro_molecule_descriptors
- **环境配置**: 需创建 MolBert 的 Conda 环境；
- **核心文件**：
  - `Drug_Embeddings.py`: 计算 ECFP 和 RDKit；
  - `MolBertFeaturizer.py`: 计算 MolBert；
  - 代码库链接：[BenevolentAI/MolBERT](https://github.com/BenevolentAI/MolBERT)。

### pro_target_embeddings
- **环境配置**: 需创建 Bio-embeddings 环境；
- **工具适配**：
  - SeqVec 仅支持 CPU 版本（GPU 版本报错未解决）；
  - UniRep 的 GPU 版本需调整 CUDA，实验中直接使用 CPU 版本；
  - 代码库链接：[sacdallago/bio-embeddings](https://github.com/sacdallago/bio-embeddings)。

---

## 代码参考（hyper-dti-references）
- 核心参考文件：`hyper-dti-references/hyper_dti/baselines/tabular_baselines.py`。

---

## 机器学习代码（scripts）

![](https://wy-static.wenxiaobai.com/chat-doc/82624a626cd983a29202a7712e710156-image.png)

### dataAcquisition
- **Filter**: 筛选 KIBA、Davis、BindingDB 和 Lenselink 数据集（`isin()` 方法）；
- **md_dataPreparation**: 完成药物分子和靶标信息的编码处理（类似 `train_test_split`）；
- **raw_dataPreparation**: 原始数据集的实验性编码处理。

### ML_DTI_Tasks
![](https://wy-static.wenxiaobai.com/chat-doc/3ab89406e3fc91fe94e41a4785310b18-image.png)
- **分类任务 & 回归任务**：
  - 模型初始化、参数设置；
  - 性能评估（分类：ROC/PR 曲线；回归：预测值-真实值分布）；
  - 单次训练（`run_single()`）与多次训练（`run_multiple()`）；
  - 分类任务支持超参数优化（因时间紧张未执行）。

### BindingDB_comparision
- 模型性能汇总展示（使用 Jupyter 编写）。

---

**注**：所有图片链接需确保网络访问权限，实验环境配置依赖的库版本需严格对齐。