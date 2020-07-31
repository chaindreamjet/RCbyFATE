from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from math import isnan
from hom.preprocessing import *
from hom.comp import missToComp, getComp
from hom.eda import *

if __name__ == "__main__":
    bureau = pd.read_csv("raw_data/bureau.csv")
    target = pd.read_csv("target.csv")
    raw = pd.merge(target, bureau, on='SK_ID_CURR')
    # raw = raw[:50000]

    '''
    创建新特征
    '''
    raw['DAYS_CREDIT_ENDDATE_MINUS_ENDDATE_FACT'] = raw['DAYS_CREDIT_ENDDATE']-raw['DAYS_ENDDATE_FACT']

    id_name = 'SK_ID_CURR'
    target_name = 'TARGET'

    '''
    填充缺失数据
    '''
    str_columns = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']
    cols = raw.columns
    nan_df = pd.DataFrame(raw[cols[~cols.isin([id_name, target_name])]].isnull().sum(), columns=['counts'])
    nan_cols = nan_df[nan_df['counts'] != 0].index.tolist()
    cols = raw.columns
    isTime = True

    temp_df = raw[['SK_ID_CURR', 'TARGET']]
    comp_df = temp_df.copy()
    cols = cols[~cols.isin(['SK_ID_CURR', 'TARGET', 'SK_ID_BUREAU'])]
    for col in tqdm(cols):
        if col in nan_cols:
            temp = raw.copy()
            pad = missToComp(temp, col, 'SK_ID_CURR', 'TARGET', dispersed=False, isTime=isTime)
            comp_df[col] = pad.getBestCompResult()
        else:
            comp_df[col] = raw[col].tolist()


    '''
    对于每一个SK_ID_CURR聚合SK_ID_BUREAU的数量
    '''
    count_df = raw.groupby(['SK_ID_CURR']).agg({'SK_ID_BUREAU':'count'})
    count_df.columns = ['BUREAU_COUNT']
    count_df = count_df.reset_index()

    '''
    对于所有的类别型变量进行类别count
    '''
    str_df = get_str_df(str_columns, raw, id_name='SK_ID_CURR')

    '''
    对所有的数值型变量进行统计特征提取
    '''
    num_columns = ['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT',
                   'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
                   'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE',
                   'AMT_ANNUITY', 'CREDIT_DAY_OVERDUE',
                   'DAYS_CREDIT_ENDDATE_MINUS_ENDDATE_FACT', 'CNT_CREDIT_PROLONG']
    groupby_rule = ['SK_ID_CURR']
    
    num_df = get_num_df(comp_df, "", num_columns, groupby_rule)

    df_list = [target, count_df, str_df, num_df]
    temp_merge_df = get_merge_df(df_list, 'SK_ID_CURR')
    cols = temp_merge_df.columns
    temp_merge_df.columns = [x if x == 'SK_ID_CURR' or x == 'TARGET' else 'x'+str(i) for i,x in enumerate(cols)]


    '''
    对CREDIT_ACTIVE进行聚合
    '''
    temp = raw.copy()
    secondary_rule = 'CREDIT_ACTIVE'
    temp['COUNT'] = temp['SK_ID_CURR'].tolist()
    count_df = temp.groupby(['SK_ID_CURR',secondary_rule]).agg({'COUNT':'count'})
    count_df = count_df.unstack(secondary_rule)
    count_df.columns = list(map(lambda x:secondary_rule+"_"+str(x),count_df.columns.tolist()))
    count_df = count_df.reset_index()
    ## 1.2: 统计str的数量
    str_columns = ['CREDIT_CURRENCY', 'CREDIT_TYPE']
    str_df = get_second_str_df(str_columns, temp, 'SK_ID_CURR', secondary_rule)
    ## 1.3: 统计num的统计特征
    num_df = get_second_num_df(temp,num_columns,secondary_rule,groupby_rule='SK_ID_CURR',id_name='SK_ID_CURR')
    df_list = [target,count_df,str_df,num_df]
    temp_merge_repay_state_df = get_merge_df(df_list,'SK_ID_CURR')
    cols = temp_merge_repay_state_df.columns
    temp_merge_repay_state_df.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]
    temp_merge_repay_state_df = getComp(temp_merge_repay_state_df, False, id_name, target_name, str_columns)
    # temp_merge_repay_state_df = getPCAResult(temp_merge_repay_state_df,target_name,id_name)

    '''
    对CREDIT_CURRENCY进行聚合
    '''
    temp = raw.copy()
    secondary_rule = 'CREDIT_CURRENCY'
    temp['COUNT'] = temp['SK_ID_CURR'].tolist()
    count_df = temp.groupby(['SK_ID_CURR',secondary_rule]).agg({'COUNT':'count'})
    count_df = count_df.unstack(secondary_rule)
    count_df.columns = list(map(lambda x:secondary_rule+"_"+str(x),count_df.columns.tolist()))
    count_df = count_df.reset_index()
    ## 1.2: 统计str的数量
    str_columns = ['CREDIT_ACTIVE','CREDIT_TYPE']
    str_df = get_second_str_df(str_columns,temp,'SK_ID_CURR',secondary_rule)
    ## 1.3: 统计num的统计特征
    num_df = get_second_num_df(temp,num_columns,secondary_rule,groupby_rule='SK_ID_CURR',id_name='SK_ID_CURR')
    df_list = [target,count_df,str_df,num_df]
    temp_merge_currency_df = get_merge_df(df_list,'SK_ID_CURR')
    cols = temp_merge_currency_df.columns
    temp_merge_currency_df.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]
    temp_merge_currency_df = getComp(temp_merge_currency_df,False, id_name,target_name,str_columns)
    # temp_merge_repay_state_df = getPCAResult(temp_merge_repay_state_df,target_name,id_name)

    '''
    记录最早/最晚一次CREDIT_CURRENCY，CREDIT_ACTIVE，CREDIT_TYPE,CNT_CREDIT_PROLON的状态
    '''
    df = comp_df[['SK_ID_CURR','TARGET']].drop_duplicates().sort_values('SK_ID_CURR')
    df = df.reset_index(drop=True)
    for item in str_columns:
        grouped = pd.DataFrame(comp_df.groupby('SK_ID_CURR')['DAYS_CREDIT',item].apply(record))
        df["EARLY_"+item+"_RECORD"] = [x[0][0] for x in grouped.values.tolist()]
        df["LATEST_"+item+"_RECORD"] = [x[0][1] for x in grouped.values.tolist()]
        
    '''
    最近一次违约的时间/最早一次违约时间
    '''
    grouped = pd.DataFrame(comp_df.groupby('SK_ID_CURR')['DAYS_CREDIT','CNT_CREDIT_PROLONG'].apply(default_time))
    time_record = grouped[0].values
    latest_default = [np.squeeze(item)[0] for item in time_record] 
    early_default = [np.squeeze(item)[1] for item in time_record] 
    latest_default = [0 if isnan(x)==True else x for x in latest_default ]
    early_default = [0 if isnan(x)==True else x for x in early_default ]

    df['EARLY_DEFAULT_TIME'] = early_default
    df['LATEST_DEFAULT_TIME'] = latest_default

    str_columns = ['EARLY_CREDIT_ACTIVE_RECORD', 'LATEST_CREDIT_ACTIVE_RECORD',
                'EARLY_CREDIT_CURRENCY_RECORD', 'LATEST_CREDIT_CURRENCY_RECORD',
                'EARLY_CREDIT_TYPE_RECORD', 'LATEST_CREDIT_TYPE_RECORD']
    str_columns = [x for x in str_columns if x in df.columns.tolist()]
    str_default_df = get_str_df(str_columns, df, id_name='SK_ID_CURR')

    str_default_df['EARLY_DEFAULT_TIME'] = early_default
    str_default_df['LATEST_DEFAULT_TIME'] = latest_default
    temp_str = pd.merge(target, str_default_df, on='SK_ID_CURR')

    df_list = [temp_merge_df, temp_merge_repay_state_df,
           temp_merge_currency_df, temp_str]

    merge_df = get_merge_df(df_list, ['SK_ID_CURR','TARGET'])
    cols = merge_df.columns
    merge_df.columns = [x if x == 'SK_ID_CURR' or x == 'TARGET' else 'x'+str(i) for i,x in enumerate(cols)]
    high_corr_cols = get_high_corr('TARGET', merge_df, id_name, 0.03)
    merge_df = merge_df[list(high_corr_cols.keys())]

    # merge_df.to_csv("fate/fate_bureau.csv", index=False)
