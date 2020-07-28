### feature_extract0

- 处理user_info数据的信息
- 使用one-hot处理

### feature_extract1

- 处理bill_detail数据
- 补充特征，扩充特征

### feature_extract2

- 处理browse_history的数据
- 统计特征
- 合并之前的feature_extract。得到45k*900的数据集

### feature_selection

- 对feature extract的数据，使用xgboost 做feature_selection
- 选出top 100维的数据

### horizontal_training

- 划分测试集，bank A，bank B的数据
- 本地模拟训练，得到各种情况下的auc