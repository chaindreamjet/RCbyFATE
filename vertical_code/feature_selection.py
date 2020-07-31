import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from hom.preprocessing import *

if __name__ == "__main__":
    # read bureau datasets
    app = pd.read_csv('./fate/fate_application.csv')
    bur = pd.read_csv('./fate/fate_bureau.csv')
    bureau_balance = pd.read_csv('./fate/fate_bureau_balance.csv')
    # read application datasets
    credit = pd.read_csv('./fate/credit_card.csv')
    install = pd.read_csv('./fate/installment_payment.py')
    pos = pd.read_csv('./fate/POS_CASH_balance.py')
    pre = pd.read_csv('./fate/previous_application.py')

    df_list = [app, bur, bureau_balance]
    bur = get_merge_df(df_list, ['SK_ID_CURR', 'TARGET'])

    df_list = [credit, install, pos, pre]
    hom = get_merge_df(df_list, ['SK_ID_CURR', 'TARGET'])

    fraud_df = bur[bur['TARGET'] == 1]
    no_fraud_df = bur[bur['TARGET'] == 0]
    no_fraud_df = no_fraud_df.sample(frac=1, random_state=42)
    no_fraud_df = no_fraud_df[:int(fraud_df.shape[0] * 1.5)]
    normal_distribution_df = pd.concat([fraud_df, no_fraud_df])
    normal_distribution_df.shape
    df = normal_distribution_df.sample(frac=1, random_state=22)

    cols = df.columns
    new = [x if x == 'SK_ID_CURR' or x == 'TARGET' else 'x_' + str(i) for i, x in enumerate(cols)]
    df.columns = new
    df.rename(columns={"SK_ID_CURR": "id", "TARGET": "y"}, inplace=True)
    b_train, b_test = train_test_split(df, test_size=0.1, random_state=22)

    train = b_train
    val = b_test

    y = train.y
    X = train.drop(['id', 'y'], axis=1)
    val_y = val.y
    val_X = val.drop(['id', 'y'], axis=1)
    xgb_train = xgb.DMatrix(X, label=y)
    xgb_val = xgb.DMatrix(val_X, label=val_y)

    ### feature selection
    '''
    feature selection
    '''
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'scale_pos_weight': 1 / 8,
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
        'nthread': 8,  # cpu 线程数
        'eval_metric': 'auc'
    }

    plst = list(params.items())
    num_rounds = 300  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    model = xgb.train(plst, xgb_train, num_boost_round=num_rounds, evals=watchlist, early_stopping_rounds=50)

    fscore = model.get_fscore()
    f_idx = sorted(fscore.items(), key=lambda x: x[1], reverse=True)
    sorted_idxs = []
    for i in f_idx:
        sorted_idxs.append(i[0])

    fs_output = df[["id", 'y'] + sorted_idxs[:30]]
    fs_output.to_csv("train/fs_bur.csv", index=False)
    hom.to_csv('train/fs_agg.csv',index=False)