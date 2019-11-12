[TOC]

#  SCZ research

## Person相关矩阵

- 计算Person相关矩阵，并取其上三角；
- 滑动窗口计算Person相关矩阵。

## PCA

使用PCA降维

## NMF

使用NMF降维

## AutoEncoder

使用Autoencoder降维。

# 实验结果

## 分类结果
|     模型      |      数据集       |     acc     | epoch |
| :-----------: | :---------------: | :---------: | :---: |
|     SVM       | Pos: 149 Neg: 103 | 0.70 ± 0.09 |   -   |
|     MLP       | Pos: 149 Neg: 103 |   0.7692    |  900  |
| NMF_32-MLP    | Pos: 149 Neg: 103 |   0.7692    |  500  |
| NMF_128-SVM   | Pos: 149 Neg: 103 | 0.68 ± 0.08 |   -   |
| NMF_128-MLP   | Pos: 149 Neg: 103 |   0.7308    |  800  |

## 分析讨论
- 线性NMF降低维度时损失大量信息，使降维后的特征矩阵在SVM和MLP分类器中的表现均出现下降。

- NMF降维可以加快MLP训练收敛速度。

- autoencoder获得编码其实较为稀疏
