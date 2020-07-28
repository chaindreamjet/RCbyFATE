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

def unique_count(x):
    return len(list(set(x)))

def get_restriction_df(filter_df,func_name,id_name,col):
    out_df = filter_df.groupby(id_name).agg({col:func_name})
    if len(id_name)>1:
        out_df = out_df.unstack(id_name[1])
    out_df = out_df.reset_index()
    return out_df


if __name__ == "__main__":
    raw = pd.read_csv("raw_data/POS_CASH_balance.csv")
    target = pd.read_csv("target.csv")
    raw = pd.merge(target,raw,on='SK_ID_CURR')
    raw = raw.sort_values(['SK_ID_PREV','MONTHS_BALANCE'])
    id_name = 'SK_ID_CURR'
    target_name = 'TARGET'

    ## 根据特征本质提取特征
    raw['CNT_INSTALMENT_TO_CNT_INSTALMENT_FUTURE'] = raw['CNT_INSTALMENT'].values/raw['CNT_INSTALMENT_FUTURE'].values
    raw['CNT_INSTALMENT_FUTURE_MINUS_CNT_INSTALMENT_FUTURE'] = raw['CNT_INSTALMENT'].values-raw['CNT_INSTALMENT_FUTURE'].values

    '''
    缺失值填补
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
    无分类统计
    '''
    # 1: 根据SK_ID_CURR直接进行统计
    ## 1.1: 对NAME_CONTRACT_STATUS进行类别总数的统计
    str_columns = ['NAME_CONTRACT_STATUS', 'SK_DPD', 'SK_DPD_DEF']
    str_df = get_str_df(['NAME_CONTRACT_STATUS'],comp_df,'SK_ID_CURR')
    ## 1.2: count数量
    count_df = comp_df.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
    count_df.columns = ['POS_CASH_COUNT']
    count_df = count_df.reset_index()
    ## 1.3: 对SK_DPD，SK_DPD_DEF去零后求max，min，count
    filter_df = comp_df[comp_df['SK_DPD']!=0]
    SK_DPD_df = get_restriction_df(filter_df,['count','max','min'],[id_name],'SK_DPD')
    filter_df = comp_df[comp_df['SK_DPD_DEF']!=0]
    SK_DPD_DEF_df = get_restriction_df(filter_df,['count','max','min'],[id_name],'SK_DPD_DEF')
    ## 1.4: 提取统计特征
    cols = comp_df.columns
    num_columns = list(cols[~cols.isin(['SK_ID_CURR','TARGET','SK_ID_PREV','NAME_CONTRACT_STATUS'])])
    num_df = get_num_df(comp_df,'',num_columns)
    # 1.5: 对MONTHS_BALANCE记录总长度，记录最近一次和最远一次分别是什么时候
    month_balance_df = comp_df.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE':{'count','max','min'}})
    month_balance_df.columns = ['MONTHS_BALANCE_FIRST_RECORD_TIME',
    'MONTHS_BALANCE_LATEST_RECORD_TIME','MONTH_BALANCE_COUNTS']
    month_balance_df = month_balance_df.reset_index()

    df_list = [target, str_df, count_df, SK_DPD_df, SK_DPD_DEF_df, num_df, month_balance_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    cols = temp_merge_df.columns
    temp_merge_df.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_temp_df = getComp(temp_merge_df, id_name, target_name, str_columns)

    '''
    对NAME_CONTRACT_STATUS进行分类统计
    '''
    temp = comp_df.copy()
    secondary_rule = 'NAME_CONTRACT_STATUS'
    count_df = temp.groupby([id_name,secondary_rule]).agg({'SK_ID_PREV':'count'})
    count_df = count_df.unstack(secondary_rule)
    count_df.columns = list(map(lambda x:secondary_rule+"_"+str(x),count_df.columns.tolist()))
    count_df = count_df.reset_index()
    # 2: 统计str的种类
    # str_columns = ['SK_DPD', 'SK_DPD_DEF']
    # str_df = get_second_str_df(str_columns,temp,id_name,secondary_rule)
    # 3: 对SK_DPD，SK_DPD_DEF去零后求max，min，count
    filter_df = comp_df[comp_df['SK_DPD']!=0]
    SK_DPD_df = get_restriction_df(filter_df,['count','max','min'],[id_name,secondary_rule],'SK_DPD')
    filter_df = comp_df[comp_df['SK_DPD_DEF']!=0]
    SK_DPD_DEF_df = get_restriction_df(filter_df,['count','max','min'],[id_name,secondary_rule],'SK_DPD_DEF')
    # 4: 对所有num_cols提取统计特征
    cols = temp.columns
    num_df = get_second_num_df(temp,num_columns,secondary_rule,groupby_rule=id_name,id_name=id_name)

    df_list = [target, str_df, count_df, SK_DPD_df, SK_DPD_DEF_df, num_df, month_balance_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    cols = temp_merge_df.columns
    temp_merge_df.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_temp_name_df = getComp(temp_merge_df, id_name, target_name, str_columns)

    def getDiffTimeWindowResult(time_windows, comp_df):
        resriction = time_windows*3+1
        '''
        按照3个月来统计
        '''
        temp = comp_df[comp_df['MONTHS_BALANCE']>-4]
        # 1: 根据SK_ID_CURR直接进行统计
        ## 1.1: 对NAME_CONTRACT_STATUS进行类别总数的统计
        str_columns = ['NAME_CONTRACT_STATUS', 'SK_DPD', 'SK_DPD_DEF']
        str_df = get_str_df(['NAME_CONTRACT_STATUS'],temp,'SK_ID_CURR')
        ## 1.2: count数量
        count_df = temp.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
        count_df.columns = ['POS_CASH_COUNT']
        count_df = count_df.reset_index()
        ## 1.3: 对SK_DPD，SK_DPD_DEF去零后求max，min，count
        filter_df = temp[temp['SK_DPD']!=0]
        SK_DPD_df = get_restriction_df(filter_df,['count','max','min'],[id_name],'SK_DPD')
        filter_df = temp[temp['SK_DPD_DEF']!=0]
        SK_DPD_DEF_df = get_restriction_df(filter_df,['count','max','min'],[id_name],'SK_DPD_DEF')
        ## 1.4: 提取统计特征
        cols = temp.columns
        num_columns = list(cols[~cols.isin(['SK_ID_CURR','TARGET','SK_ID_PREV','NAME_CONTRACT_STATUS'])])
        num_df = get_num_df(temp,'',num_columns)

        df_list = [target, str_df, count_df, SK_DPD_df, SK_DPD_DEF_df, num_df]
        temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
        cols = temp_merge_df.columns
        temp_merge_df.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]
        high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
        cols = [id_name,target_name]
        cols.extend(list(high_corr.keys()))
        temp_merge_df = temp_merge_df[cols]
        comp_temp_month_df = getComp(temp_merge_df, id_name, target_name, str_columns)
        return comp_temp_month_df

    '''
    按照不同时间窗口来进行统计3,6,9,12,15,18,21,24
    '''
    comp_temp_month_df = getDiffTimeWindowResult(3, comp_df)
    for window in tqdm([6,9,12,15,18,21,24]):
        temp = comp_temp_month_df = getDiffTimeWindowResult(window, comp_df)
        comp_temp_month_df = pd.merge(comp_temp_month_df, temp, on=['SK_ID_CURR','TARGET'])

    df_list = [comp_temp_df, comp_temp_name_df, comp_temp_month_df]
    temp = get_merge_df(df_list,['SK_ID_CURR','TARGET'])
    cols = temp.columns
    temp.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]