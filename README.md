# PAKDD2020 阿里巴巴智能运维算法大赛(线上33.44)
[官方赛题介绍](https://tianchi.aliyun.com/competition/entrance/231775/information)
\-
一句话总结，尽可能提前的预测出未来会发生故障的磁盘，从而可更改硬盘使用策略，减少停机费用以及避免因磁盘损坏而引起的数据丢失。比赛中，数据粒度为磁盘每天的smart值。最终竞赛排名 ，线上f1score-33.44
## 一、 解决方案以及不足
主要方案为分析raw以及normalized的smart（Self-Monitoring Analysis and
Reporting Technology）值与磁盘故障的关系，通过emsmble tree
模型对无故障盘以及有故障盘进行不均衡的二分类（i.e. 1158:56031892, 0.0000207)
，并通过rank的方法，返回故障置信度高的盘，作为最后的故障盘输出。
### 1.1 数据预处理

根据题意以及为了增加些许故障盘样本，通过label
smoothing，将故障盘及其发生故障前30天的数据标为故障数据，其他清洗规则如下:  
*  剔除smart缺失值大于其本身30%的列;
*  剔除smart值的unique个数小于3的列;
*  剔除power on hours 等于0的列;
*  对同一块磁盘在同一天有多条数据去重.
### 1.2. 特征工程
特征工程具体分为如下四步: 

1.2.1.原始特征 - 基于correlation ratio
计算连续smart值与离散故障标签的相似度以及对smart值的业务理解，挑选初始的raw以及normalized
smart作为原始特征;

1.2.2. 构造continuous features -
对部分smart值，以磁盘自身数据为单位进行基于滑窗的特征构建（e.g.
窗口期内的最大值，最小值，方差，平均值，差分以及斜率值等);

1.2.3. 构造category features -
对部分smart值进行分段，分桶，每一段作为一个类别以及将model type也转为category
feature;

1.2.4. 特征选择 -
基于cv，平均特征在最佳cv分数中所有folds中的重要性，挑选topk个特征。

### 1.3. 模型训练

将赛题视为极度不均衡二分类的任务，评估指标选用了auc + 比赛定义的f1 score, 采用**lightgbm + focal loss或scale_pos_weight**并基于**下采样或结合上下采样**以及**隔月验证**通过**random search**+**time series split cv 或单月验证的方式**来对模型进行训练。 
* 下采样或结合上下采样 - **基于月份的下采样**（i.e.
  每个月随机sample一定数量的无故障磁盘）, **基于通电时间的下采样**（i.e.
  根据power on hours 将磁盘分段分桶，采样一定数量通电时间各不相同的无故障磁盘),**基于聚类的下采样**（i.e. 通过kmeans++聚类，对每个无故障盘簇进行采样）以及也可以在**下采样后配合SMOTE再做上采样**，所有采样只针对训练集，不包含验证集，单纯的下采样方法并未直接采到1:1的比例（数据量少极易过拟合，泛华能力差），竞赛中线上最好成绩使用的下采样比例为0.3.
* 隔月验证 -
  因将故障盘发生故障前30天的数据也标记为故障，所以如果以7月为验证集，为模拟真实业务场景，7月故障盘未知其往前推30天是否有故障也应为未知，所以6月的故障数据不全，故不使用，训练数据最多使用到5月31日。

### 1.4 方案不足

不足主要体现在如下方面：
* 并未挖掘出可用于前处理亦或是后处理的规则，只是单纯的依靠了模型的输出；
* 对learning to rank的使用不够深入；
* 模型迭代的策略以及对线上以及线下的错误分析；
* TODO - 根据top选手方案与自身对比，后续继续补充.
## 二、代码运行说明

通过`docker build -t 'test:$VERSION' .` 后，启动docker - `docker run -it
--rm --name test:$VERSION /bin/bash` `,
然后在docker环境中以如下pipeline逐步生成最终提交文件.
### 2.1. 数据预处理 

运行 `python3 data_preprocess.py`
清洗**2017年**、**2018年**磁盘数据并为其打上标签。

### 2.2. 特征工程

运行 `python3 feature_engineering.py`
生成得到以训练起始日期结尾的特征工程文件（e.g. fe_df_01_01.feather）。

### 2.3. 模型训练

运行 `python3 train.py` 进行参数选择，模型评估以及生成*.model用于最终预测.

### 2.4. 任务预测

运行 `python3 predict.py` 生成最终针对线上测试集的预测结果.


## 参考学习资料
[1] [smart参数中文说明（由刘登平同学提供）](./user_data/docs/SMART_explantation.jpeg)

[2] smart参数维基百科说明: <https://en.wikipedia.org/wiki/S.M.A.R.T.>

[3] 基于机器学习的磁盘故障预测的挑战及设计思想: 
<http://www.elecfans.com/d/739038.html>

[4] 基于knn的磁盘故障预测:
<https://github.com/yiguanxian/Disk-Failure-Prediction>

[6] Hard_Drive_Failure_Prediction_Using_Big_Data(BaiDu, 2015):
<https://ieeexplore.ieee.org/document/7371435>

[7] Predicting_Disk_Replacement_towards_Reliable_Data_Centers(IBM,
2016):
<https://www.kdd.org/kdd2016/subtopic/view/predicting-disk-replacement-towards-reliable-data-centers>

[8] Proactive_Prediction_of_Hard_Disk_Drive_Failure(Li, Suarez&Camacho, 2017): <https://github.com/yiguanxian/Disk-Failure-Prediction/>

[9] sample_data: <https://github.com/alibaba-edu/dcbrain/tree/master/diskdata>
