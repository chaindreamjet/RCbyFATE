import os
import sys
import time
import math
from tqdm import tqdm
import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

from math import isnan
from hom.preprocessing import *
from hom.comp import missToComp
from hom.eda import *


if __name__ == "__main__":
    raw = pd.read_csv("raw_data/previous_application.csv")
    target = pd.read_csv("target.csv")
    raw = pd.merge(target,raw,on='SK_ID_CURR')
    id_name = 'SK_ID_CURR'
    target_name = 'TARGET'
    # 创建新变量
    raw['diff1'] = raw['DAYS_LAST_DUE'] - raw['DAYS_TERMINATION']
    raw['diff2'] = raw['AMT_APPLICATION'] - raw['AMT_CREDIT']
    raw['diff3'] = raw['AMT_APPLICATION'] - raw['AMT_GOODS_PRICE']
    raw['diff4'] = raw['AMT_CREDIT'] - raw['AMT_GOODS_PRICE']
    
    cols = raw.columns
    str_columns = ['NAME_CONTRACT_TYPE','WEEKDAY_APPR_PROCESS_START','FLAG_LAST_APPL_PER_CONTRACT',
                'NFLAG_LAST_APPL_IN_DAY','NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS',
                'NAME_PAYMENT_TYPE','CODE_REJECT_REASON','NAME_TYPE_SUITE','NAME_CLIENT_TYPE',
                'NAME_GOODS_CATEGORY','NAME_PORTFOLIO','NAME_PRODUCT_TYPE','CHANNEL_TYPE',
                'NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION',
                'NFLAG_INSURED_ON_APPROVAL']
    num_columns = cols[3:][~cols[3:].isin(str_columns)]

    '''
    填充缺失值
    '''
    cols = raw.columns
    nan_df = pd.DataFrame(raw[cols[~cols.isin(['SK_ID_CURR','TARGET'])]].isnull().sum(),columns=['counts'])
    miss_columns = nan_df[nan_df['counts']!=0].index.tolist()
    unmiss_columns = nan_df[nan_df['counts']==0].index.tolist()

    comp_df = raw[['SK_ID_CURR','TARGET']]
    isTime = False
    for col in cols[~cols.isin(['SK_ID_CURR','TARGET'])]:
        if col in miss_columns and col not in str_columns:
            dispersed = False 
            temp =raw.copy()
            pad = missToComp(temp, col, 'SK_ID_CURR', 'TARGET', dispersed=dispersed,isTime=isTime)
            comp_df[col] = pad.getBestCompResult()
        else:
            comp_df[col] = raw[col].tolist()

    def get_type_count_df(secondary_rule, comp_df,str_columns):
        temp = comp_df.copy()
        count_df = temp.groupby([id_name,secondary_rule]).agg({'SK_ID_PREV':'count'})
        count_df = count_df.unstack(secondary_rule)
        count_df.columns = list(map(lambda x:secondary_rule+"_"+str(x),count_df.columns.tolist()))
        count_df = count_df.reset_index()
        # 2: 统计str的种类
        str_columns_temp = str_columns.remove(secondary_rule)
        str_df = get_second_str_df(str_columns,temp,id_name,secondary_rule)
        str_df = str_df.fillna(0)

        # 3: 对所有num_cols提取统计特征
        cols = temp.columns
        num_df = get_second_num_df(temp,num_columns,secondary_rule,groupby_rule=id_name,id_name=id_name)

        df_list = [target, str_df, count_df, num_df]
        temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
        cols = temp_merge_df.columns
        temp_merge_df.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]
        high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.05)
        cols = [id_name,target_name]
        cols.extend(list(high_corr.keys()))
        temp_merge_df = temp_merge_df[cols]
        comp_temp_name_df = getComp(temp_merge_df, id_name, target_name, str_columns)
        return comp_temp_name_df

    '''
    无分类统计
    '''
    # 1: count 计数SK_ID_PREV，unique_count计数SK_ID_PREV
    count_df = comp_df.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
    count_df.columns = ['previous_application_COUNT']
    count_df = count_df.reset_index()
    unique_count_df = comp_df.groupby(['SK_ID_CURR'])['SK_ID_PREV'].apply(unique_count)
    unique_count_df.columns = ['previous_application_UNIQUE_COUNT']
    unique_count_df = unique_count_df.reset_index()
    # 2: count计数
    str_df = get_str_df(str_columns,comp_df,'SK_ID_CURR')
    str_df = str_df.fillna(0)
    # 3: 统计统计特征
    num_df = get_num_df(comp_df,'',num_columns)

    df_list = [target, str_df, count_df, unique_count_df, num_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    cols = temp_merge_df.columns
    temp_merge_df.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_temp_df = getComp(temp_merge_df, id_name, target_name, str_columns)

    '''
    按照NAME_CONTRACT_TYPE统计
    '''
    type_df = target.copy()
    for col in tqdm(str_columns):
        df = get_type_count_df(col, comp_df,str_columns)
        type_df = pd.merge(df, type_df, on = [id_name, target_name])

    df_list = [temp_merge_df, type_df]
    temp_df = get_merge_df(df_list,['SK_ID_CURR','TARGET'])
    cols = temp_df.columns
    temp_df.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]