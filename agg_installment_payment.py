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
    raw = pd.read_csv('raw_data/installments_payments.csv')
    target = pd.read_csv("target.csv")
    raw = pd.merge(target,raw,on='SK_ID_CURR')
    raw = raw.sort_values(['SK_ID_PREV','DAYS_INSTALMENT'])
    id_name = 'SK_ID_CURR'
    target_name = 'TARGET'

    # 根据特征本质提取特征
    raw['diff1'] = raw['DAYS_INSTALMENT'] - raw['DAYS_ENTRY_PAYMENT']
    raw['diff2'] =raw['AMT_INSTALMENT']-raw['AMT_PAYMENT']

    '''
    填充缺失值
    '''
    cols = raw.columns
    nan_df = pd.DataFrame(raw[cols[~cols.isin(['SK_ID_CURR','TARGET'])]].isnull().sum(),columns=['counts'])
    miss_columns = nan_df[nan_df['counts']!=0].index.tolist()
    unmiss_columns = nan_df[nan_df['counts']==0].index.tolist()

    comp_df = raw[['SK_ID_CURR','TARGET']]
    isTime = True
    for col in cols[~cols.isin(['SK_ID_CURR','TARGET'])]:
        if col in miss_columns:
            temp = raw.copy()
            pad = missToComp(temp, col, 'SK_ID_CURR', 'TARGET', dispersed=False,isTime=isTime)
            comp_df[col] = pad.getBestCompResult()
        else:
            comp_df[col] = raw[col].tolist()

    '''
    不分类统计
    '''
    # 1: count 计数SK_ID_PREV，unique_count计数SK_ID_PREV
    temp = comp_df.copy()
    count_df = temp.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
    count_df.columns = ['installments_payments_COUNT']
    count_df = count_df.reset_index()
    unique_count_df = temp.groupby(['SK_ID_CURR'])['SK_ID_PREV'].apply(unique_count)
    unique_count_df.columns = ['installments_payments_UNIQUE_COUNT']
    unique_count_df = unique_count_df.reset_index()
    # 2: 统计统计特征
    cols = temp.columns
    cols = list(cols[~cols.isin(['SK_ID_CURR','TARGET','SK_ID_PREV'])])
    num_df = get_num_df(temp,'',cols)
    num_df = num_df.fillna(0)

    df_list = [target, count_df, unique_count_df, num_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_merge_df = getComp(temp_merge_df, id_name, target_name, [])


    '''
    按照NUM_INSTALMENT_VERSION==0提取
    '''
    temp = comp_df[comp_df['NUM_INSTALMENT_VERSION']==0]
    count_df = temp.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
    count_df = count_df.reset_index()

    # 对所有num_cols提取统计特征
    cols = temp.columns
    num_columns = list(cols[~cols.isin(['SK_ID_CURR','TARGET'])])
    num_df = get_num_df(temp,'',num_columns)

    df_list = [target, count_df, num_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_merge_version0_df = getComp(temp_merge_df, id_name, target_name, [])

    '''
    按照NUM_INSTALMENT_VERSION==1提取
    '''
    temp = comp_df[comp_df['NUM_INSTALMENT_VERSION']==1]
    count_df = temp.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
    count_df = count_df.reset_index()

    # 对所有num_cols提取统计特征
    cols = temp.columns
    num_columns = list(cols[~cols.isin(['SK_ID_CURR','TARGET'])])
    num_df = get_num_df(temp,'',num_columns)

    df_list = [target, count_df, num_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_merge_version1_df = getComp(temp_merge_df, id_name, target_name, [])

    '''
    按照NUM_INSTALMENT_NUMBER==3提取
    '''
    temp = comp_df[comp_df['NUM_INSTALMENT_NUMBER']==3]
    count_df = temp.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
    count_df = count_df.reset_index()

    # 对所有num_cols提取统计特征
    cols = temp.columns
    num_columns = list(cols[~cols.isin(['SK_ID_CURR','TARGET'])])
    num_df = get_num_df(temp,'',num_columns)

    df_list = [target, count_df, num_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_merge_number3_df = getComp(temp_merge_df, id_name, target_name, [])

    df_list = [comp_merge_df, comp_merge_version0_df, comp_merge_version1_df,
           comp_merge_version1_df, comp_merge_version1_df, 
          comp_merge_number1_df, comp_merge_number2_df, comp_merge_number3_df]
    temp_df = get_merge_df(df_list,['SK_ID_CURR','TARGET'])
    cols = temp_df.columns
    temp_df.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]