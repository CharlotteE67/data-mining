# KDD 2020 Privileged Features Distillation at Taobao Recommendations

## Outline

### 简介

非学术知识导入，要介绍**优势特征**(Privileged Features) （后验数据）的概念给听众。（约 1- 2 min）（陈纪元）

### 核心科技

基于优势特征的概念，先指出现有技术（多任务学习）的缺陷（略讲，讲不清可以跳过），再顺势提出我们的核心模型——蒸馏模型，能够合理利用起优势特征的**优势特征蒸馏(Privileged Features Distillation)**，其中包括模型蒸馏（非线性化教师内部结构）与优势特征蒸馏（regular feature加入教师模型的输入）。其中穿插举例，如下图中的一些概念，Interacted Features在具体生活中是什么意思。（约3-5 min）（黎原朝）

![image-20210415222749550](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20210415222749550.png)

之后，介绍教师模型中的特征如何被“蒸馏”到学生模型中，具体的数学/代码实现是怎样的。再提出最终使用的统一蒸馏——优势特征蒸馏+模型蒸馏。（约3-5 min）（江宇辰）

### 实验

基于上面提出的理论知识，介绍实验结果（结合表格讲数据，吹模型多好多好）（约2-3 min）（商纪豪）

### 总结

重新描述该模型所想解决/优化的问题，梳理解决/优化问题的过程，再次与其他模型进行比较——重点解决了如何优雅地利用**优势数据（Privileged Features）这一后验数据**的问题。可以加上自己的看法与展望。结尾。 （约 1-2 min)  (陈纪元)







蒸馏核心点：带温度的softmax