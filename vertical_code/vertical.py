import sklearn
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import xgboost as xgb


def performance_clf(model, X, y, name=None):
    y_predict = model.predict(X)
    if name:
        print(name, ':')
    print(f'accuracy score is: {accuracy_score(y,y_predict)}')
    print(f'precision score is: {precision_score(y,y_predict)}')
    print(f'recall score is: {recall_score(y,y_predict)}')
    print(f'auc: {roc_auc_score(y,y_predict)}')
    print('- - - - - - ')


def main(trains, num_rounds):
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'scale_pos_weight': 1 / 10,
        'gamma': 0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 8,  # 构建树的深度，越大越容易过拟合
        'lambda': 3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,  # 随机采样训练样本
        # 'colsample_bytree':0.7, # 生成树时进行的列采样
        'min_child_weight': 3,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.03,  # 如同学习率
        'seed': 1000,
        'nthread': 12,  # cpu 线程数
        'eval_metric': 'auc'
    }

    train, val = train_test_split(trains, test_size=0.1, random_state=22)  # 41697/13886
    plst = list(params.items())
    # 迭代次数

    y = train.y
    X = train.drop(['id', 'y'], axis=1)
    print(y.value_counts())

    val_y = val.y
    val_X = val.drop(['id', 'y'], axis=1)

    xgb_train = xgb.DMatrix(X, label=y)
    xgb_val = xgb.DMatrix(val_X, label=val_y)

    # return 训练和验证的错误率
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    model = xgb.train(plst, xgb_train, num_boost_round=num_rounds, evals=watchlist, early_stopping_rounds=50)

if __name__ == "__main__":
    bur = pd.read_csv("train/fs_bur.csv")
    hom = pd.read_csv("train/fs_hom.csv")

    num_rounds = 500
    main(bur, num_rounds)
    main(hom, num_rounds)

    merge_df = pd.merge(hom, bur, on=['id', 'y'])
    main(merge_df, 500)


