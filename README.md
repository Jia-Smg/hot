# 类十字形热超构材料结构

1. `model_data_generation`：生成训练类十字形结构的样本数据（训练、测试数据），分析样本数据。
2. `database`：存储、分析生成的样本数据
3. `train_model`：包含训练类十字形结构的几何参数（输入）与热导率（输出）关系的深度学习程序（用人工神经网络表达几何参数与横纵热导率之间的关系）、训练好的神经网络模型（效果符合预期）。
    1. `train_data`：训练数据
    2. `train_model`：训练算法
        1. `k1k2.py`：隐藏层2层
        2. `k1k2_optm.py`：隐藏层3层
    3. `model-<date-time>`：训练合格的模型结果

4. `inverse_design`：在深度神经模型的基础上，通过遗传算法获取目标热导率的程序。
5. `construct_model`：
    - `construct_geometry_for_diodes.py`：将多个类十字形结构拼接为完整的热二极管组件的程序，主要工作是设置好热二极管comsol文件中的参数和几何结构，参数是每个类十字形结构的参数，设置几何结构是将这些类十字形结构放到对应的位置。

有效结果

`train_model/`

1. `model-2024_05_25-19_28_20`：效果符合预期的神经网络模型，通过类十字形的几何参数预测横纵热导率。
    1. 这个模型是以`cross-shape-diff-horizontal.mph`和
`cross-shape-diff-vertical.mph`两个模型仿真出来的样本数据训练的。
    2. 材料选择的是铜，横纵热导率的上限应为400。
    3. 【模型限制】当横/纵的一方热导率很大时（360+），另一方的热导率无法很小（90-）。这是由物理性质决定的。
2. `model-optm-2025_03_27-19_29_05`: k1k2_optm.py训练出来的模型（epoch 2500）
    - 包含训练过程数据
3. `model-optm-2025_03_27-20_12_08`: k1k2_optm.py训练出来的模型（epoch 3500）
    - 包含训练过程数据
- `construct_model/basic_model/cross-shape-parameterized.mph`：热二极管源文件，等待`construct_geometry_for_diodes.py`将每个格子的类十字形结构填充进去。
