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
    raw = pd.read_csv("raw_data/credit_card_balance.csv")
    target = pd.read_csv("target.csv")
    raw = pd.merge(target,raw,on='SK_ID_CURR')
    raw = raw.sort_values(['SK_ID_PREV','MONTHS_BALANCE'])
    raw = raw.replace(np.inf,np.nan)
    id_name = 'SK_ID_CURR'
    target_name = 'TARGET'

    # 5: 根据特征本质提取特征
    raw['AMT_CREDIT_LIMIT_ACTUAL_MINUS_AMT_BALANCE'] = raw['AMT_CREDIT_LIMIT_ACTUAL'] - raw['AMT_BALANCE']
    raw['AMT_PAYMENT_TOTAL_CURRENT_MINUS_AMT_PAYMENT_CURRENT'] =raw['AMT_PAYMENT_TOTAL_CURRENT']-raw['AMT_PAYMENT_CURRENT']
    raw['AMT_RECEIVABLE_PRINCIPAL_MINUS_AMT_RECIVABLE'] =raw['AMT_RECEIVABLE_PRINCIPAL']-raw['AMT_RECIVABLE']
    raw['AMT_RECIVABLE_MINUS_AMT_TOTAL_RECEIVABLE'] = raw['AMT_RECIVABLE']-raw['AMT_TOTAL_RECEIVABLE']
    raw['AMT_RECEIVABLE_PRINCIPAL_MINUS_AMT_TOTAL_RECEIVABLE'] = raw['AMT_RECEIVABLE_PRINCIPAL']-raw['AMT_TOTAL_RECEIVABLE']

    drawings = ['AMT_DRAWINGS_ATM_CURRENT','AMT_DRAWINGS_CURRENT',
                'AMT_DRAWINGS_OTHER_CURRENT','AMT_DRAWINGS_POS_CURRENT']
    cnt = ['CNT_DRAWINGS_ATM_CURRENT','CNT_DRAWINGS_CURRENT',
        'CNT_DRAWINGS_OTHER_CURRENT','CNT_DRAWINGS_POS_CURRENT']
    for i,x in enumerate(drawings):
        raw[x+"_TO_"+cnt[i]] = raw[x]/raw[cnt[i]]

    '''
    缺失值填补
    '''
    cols = raw.columns
    nan_df = pd.DataFrame(raw[cols[~cols.isin(['SK_ID_CURR','TARGET'])]].isnull().sum(),columns=['counts'])
    miss_columns = nan_df[nan_df['counts']!=0].index.tolist()
    unmiss_columns = nan_df[nan_df['counts']==0].index.tolist()

    comp_df = raw[['SK_ID_CURR','TARGET']]
    isTime = True
    for col in tqdm(cols[~cols.isin(['SK_ID_CURR','TARGET'])]):
        if col in miss_columns:
            temp = raw.copy()
            pad = missToComp(temp, col, 'SK_ID_CURR', 'TARGET', dispersed=False,isTime=isTime)
            comp_df[col] = pad.getBestCompResult()
        else:
            comp_df[col] = raw[col].tolist()

    '''
    不分类统计
    '''
    count_df = comp_df.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
    count_df.columns = ['BUREAU_COUNT']
    count_df = count_df.reset_index()
    # 2: 统计NAME_CONTRACT_STATUS的种类
    str_columns = ['NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF']
    str_df = get_str_df(['NAME_CONTRACT_STATUS'],comp_df,'SK_ID_CURR')
    # 3: 对MONTHS_BALANCE记录总长度，记录最近一次和最远一次分别是什么时候
    month_balance_df = comp_df.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE':{'count','max','min'}})
    month_balance_df.columns = ['MONTHS_BALANCE_FIRST_RECORD_TIME',
    'MONTHS_BALANCE_LATEST_RECORD_TIME','MONTH_BALANCE_COUNTS']
    month_balance_df = month_balance_df.reset_index()
    # 4: 对所有num_cols提取统计特征
    cols = comp_df.columns
    num_columns = list(cols[~cols.isin(['SK_ID_CURR','TARGET','SK_ID_PREV','NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF'])])
    num_df = get_num_df(comp_df,'',num_columns)
    num_df = num_df.fillna(0)
    # 5: 对SK_DPD，SK_DPD_DEF去零后求max，min，count
    temp_df = comp_df[['SK_ID_CURR']].drop_duplicates()
    filter_df = comp_df[comp_df['SK_DPD']!=0]
    sk_dpd_df = filter_df.groupby('SK_ID_CURR').agg({'SK_DPD':{'count','max','min'}})
    sk_dpd_df.columns = list(map(lambda x:"SK_DPD"+x,['_MIN','_MAX','_COUNT']))
    sk_dpd_df = sk_dpd_df.reset_index()
    filter_df = comp_df[comp_df['SK_DPD_DEF']!=0]
    SK_DPD_DEF_df = filter_df.groupby('SK_ID_CURR').agg({'SK_DPD_DEF':{'count','max','min'}})
    SK_DPD_DEF_df.columns = list(map(lambda x:"SK_DPD_DEF_df"+x,['_MIN','_MAX','_COUNT']))
    SK_DPD_DEF_df = SK_DPD_DEF_df.reset_index()
    # AMT_CREDIT_LIMIT_ACTUAL
    AMT_CREDIT_LIMIT_ACTUAL_df = comp_df.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].apply(unique_count)
    AMT_CREDIT_LIMIT_ACTUAL_df.columns = ['AMT_CREDIT_LIMIT_ACTUAL_UNIQUE_COUNT']
    AMT_CREDIT_LIMIT_ACTUAL_df = AMT_CREDIT_LIMIT_ACTUAL_df.reset_index()

    df_list = [target, count_df,str_df,month_balance_df,
            num_df, sk_dpd_df, SK_DPD_DEF_df, AMT_CREDIT_LIMIT_ACTUAL_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_merge_df = getComp(temp_merge_df, id_name, target_name, str_columns)

    '''
    按照NAME_CONTRACT_STATUS提取所有的特征
    '''
    temp = comp_df.copy()
    secondary_rule = 'NAME_CONTRACT_STATUS'
    temp['COUNT'] = temp[id_name].tolist()
    count_df = temp.groupby([id_name,secondary_rule]).agg({'COUNT':'count'})
    count_df = count_df.unstack(secondary_rule)
    count_df.columns = list(map(lambda x:secondary_rule+"_"+str(x),count_df.columns.tolist()))
    count_df = count_df.reset_index()
    # 2: 统计NAME_CONTRACT_STATUS的种类
    str_columns = ['SK_DPD','SK_DPD_DEF']
    str_df = get_second_str_df(str_columns,temp,id_name,secondary_rule)
    # 3: 对MONTHS_BALANCE记录总长度，记录最近一次和最远一次分别是什么时候
    month_balance_df = temp.groupby([id_name,secondary_rule]).agg({'MONTHS_BALANCE':{'count','max','min'}})
    unstack_df = month_balance_df.unstack([secondary_rule])
    columns = [id_name]
    unstack_columns = list(unstack_df.columns)
    columns.extend(unstack_columns)
    unstack_df = unstack_df.reset_index()
    rename_df = pd.DataFrame(unstack_df.values, columns=columns)
    rename_df = rename_df.dropna(axis=1, how='all')
    rename_df = rename_df.fillna(0)
    # 4: 对所有num_cols提取统计特征
    cols = temp.columns
    num_columns = list(cols[~cols.isin(['SK_ID_CURR','TARGET','SK_ID_PREV','NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF'])])
    num_df = get_second_num_df(temp,num_columns,secondary_rule,groupby_rule=id_name,id_name=id_name)
    # AMT_CREDIT_LIMIT_ACTUAL
    AMT_CREDIT_LIMIT_ACTUAL_df = temp.groupby([id_name,secondary_rule])['AMT_CREDIT_LIMIT_ACTUAL'].apply(unique_count)
    unstack_df = month_balance_df.unstack([secondary_rule])
    columns = [id_name]
    unstack_columns = list(unstack_df.columns)
    columns.extend(unstack_columns)
    unstack_df = unstack_df.reset_index()
    AMT_CREDIT_LIMIT_rename_df = pd.DataFrame(unstack_df.values, columns=columns)
    AMT_CREDIT_LIMIT_rename_df = AMT_CREDIT_LIMIT_rename_df.dropna(axis=1, how='all')
    AMT_CREDIT_LIMIT_rename_df = AMT_CREDIT_LIMIT_rename_df.fillna(0)

    df_list = [target, count_df,str_df,rename_df,
            num_df, AMT_CREDIT_LIMIT_rename_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_merge_name_df = getComp(temp_merge_df, id_name, target_name, str_columns)

    '''
    按照SK_DPD==0提取
    '''
    temp = comp_df[comp_df['SK_DPD']==0]
    count_df = temp.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
    count_df.columns = ['BUREAU_COUNT']
    count_df = count_df.reset_index()
    # 2: 统计NAME_CONTRACT_STATUS的种类
    str_columns = ['NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF']
    str_df = get_str_df(['NAME_CONTRACT_STATUS'],temp,'SK_ID_CURR')
    # 4: 对所有num_cols提取统计特征
    cols = temp.columns
    num_columns = list(cols[~cols.isin(['SK_ID_CURR','TARGET','SK_ID_PREV','NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF'])])
    num_df = get_num_df(temp,'',num_columns)
    num_df = num_df.fillna(0)
    # 5: 对SK_DPD，SK_DPD_DEF去零后求max，min，count
    temp_df = temp[['SK_ID_CURR']].drop_duplicates()
    filter_df = temp[comp_df['SK_DPD']!=0]
    sk_dpd_df = filter_df.groupby('SK_ID_CURR').agg({'SK_DPD':{'count','max','min'}})
    sk_dpd_df.columns = list(map(lambda x:"SK_DPD"+x,['_MIN','_MAX','_COUNT']))
    sk_dpd_df = sk_dpd_df.reset_index()
    filter_df = temp[temp['SK_DPD_DEF']!=0]
    SK_DPD_DEF_df = filter_df.groupby('SK_ID_CURR').agg({'SK_DPD_DEF':{'count','max','min'}})
    SK_DPD_DEF_df.columns = list(map(lambda x:"SK_DPD_DEF_df"+x,['_MIN','_MAX','_COUNT']))
    SK_DPD_DEF_df = SK_DPD_DEF_df.reset_index()
    # AMT_CREDIT_LIMIT_ACTUAL
    AMT_CREDIT_LIMIT_ACTUAL_df = temp.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].apply(unique_count)
    AMT_CREDIT_LIMIT_ACTUAL_df.columns = ['AMT_CREDIT_LIMIT_ACTUAL_UNIQUE_COUNT']
    AMT_CREDIT_LIMIT_ACTUAL_df = AMT_CREDIT_LIMIT_ACTUAL_df.reset_index()

    df_list = [target, count_df,str_df,month_balance_df,
            num_df, sk_dpd_df, SK_DPD_DEF_df, AMT_CREDIT_LIMIT_ACTUAL_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_merge_SK_DPD_df = getComp(temp_merge_df, id_name, target_name, str_columns)


    '''
    按照3个月提取所有的特征
    '''
    temp = comp_df[comp_df['MONTHS_BALANCE']>-4]
    count_df = temp.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
    count_df.columns = ['BUREAU_COUNT']
    count_df = count_df.reset_index()
    # 2: 统计NAME_CONTRACT_STATUS的种类
    str_columns = ['NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF']
    str_df = get_str_df(['NAME_CONTRACT_STATUS'],temp,'SK_ID_CURR')
    # 4: 对所有num_cols提取统计特征
    cols = temp.columns
    num_columns = list(cols[~cols.isin(['SK_ID_CURR','TARGET','SK_ID_PREV','NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF'])])
    num_df = get_num_df(temp,'',num_columns)
    num_df = num_df.fillna(0)
    # 5: 对SK_DPD，SK_DPD_DEF去零后求max，min，count
    temp_df = temp[['SK_ID_CURR']].drop_duplicates()
    filter_df = temp[comp_df['SK_DPD']!=0]
    sk_dpd_df = filter_df.groupby('SK_ID_CURR').agg({'SK_DPD':{'count','max','min'}})
    sk_dpd_df.columns = list(map(lambda x:"SK_DPD"+x,['_MIN','_MAX','_COUNT']))
    sk_dpd_df = sk_dpd_df.reset_index()
    filter_df = temp[temp['SK_DPD_DEF']!=0]
    SK_DPD_DEF_df = filter_df.groupby('SK_ID_CURR').agg({'SK_DPD_DEF':{'count','max','min'}})
    SK_DPD_DEF_df.columns = list(map(lambda x:"SK_DPD_DEF_df"+x,['_MIN','_MAX','_COUNT']))
    SK_DPD_DEF_df = SK_DPD_DEF_df.reset_index()
    # AMT_CREDIT_LIMIT_ACTUAL
    AMT_CREDIT_LIMIT_ACTUAL_df = temp.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].apply(unique_count)
    AMT_CREDIT_LIMIT_ACTUAL_df.columns = ['AMT_CREDIT_LIMIT_ACTUAL_UNIQUE_COUNT']
    AMT_CREDIT_LIMIT_ACTUAL_df = AMT_CREDIT_LIMIT_ACTUAL_df.reset_index()

    df_list = [target, count_df,str_df,month_balance_df,
            num_df, sk_dpd_df, SK_DPD_DEF_df, AMT_CREDIT_LIMIT_ACTUAL_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_merge_3month_df = getComp(temp_merge_df, id_name, target_name, str_columns)

    '''
    按照6个月提取所有的特征
    '''
    temp = comp_df[comp_df['MONTHS_BALANCE']>-7]
    count_df = temp.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
    count_df.columns = ['BUREAU_COUNT']
    count_df = count_df.reset_index()
    # 2: 统计NAME_CONTRACT_STATUS的种类
    str_columns = ['NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF']
    str_df = get_str_df(['NAME_CONTRACT_STATUS'],temp,'SK_ID_CURR')
    # 3: 对MONTHS_BALANCE记录总长度，记录最近一次和最远一次分别是什么时候
    month_balance_df = temp.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE':{'count','max','min'}})
    month_balance_df.columns = ['MONTHS_BALANCE_FIRST_RECORD_TIME',
    'MONTHS_BALANCE_LATEST_RECORD_TIME','MONTH_BALANCE_COUNTS']
    month_balance_df = month_balance_df.reset_index()
    # 4: 对所有num_cols提取统计特征
    cols = temp.columns
    num_columns = list(cols[~cols.isin(['SK_ID_CURR','TARGET','SK_ID_PREV','NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF'])])
    num_df = get_num_df(temp,'',num_columns)
    num_df = num_df.fillna(0)
    # 5: 对SK_DPD，SK_DPD_DEF去零后求max，min，count
    temp_df = temp[['SK_ID_CURR']].drop_duplicates()
    filter_df = temp[comp_df['SK_DPD']!=0]
    sk_dpd_df = filter_df.groupby('SK_ID_CURR').agg({'SK_DPD':{'count','max','min'}})
    sk_dpd_df.columns = list(map(lambda x:"SK_DPD"+x,['_MIN','_MAX','_COUNT']))
    sk_dpd_df = sk_dpd_df.reset_index()
    filter_df = temp[temp['SK_DPD_DEF']!=0]
    SK_DPD_DEF_df = filter_df.groupby('SK_ID_CURR').agg({'SK_DPD_DEF':{'count','max','min'}})
    SK_DPD_DEF_df.columns = list(map(lambda x:"SK_DPD_DEF_df"+x,['_MIN','_MAX','_COUNT']))
    SK_DPD_DEF_df = SK_DPD_DEF_df.reset_index()
    # AMT_CREDIT_LIMIT_ACTUAL
    AMT_CREDIT_LIMIT_ACTUAL_df = temp.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].apply(unique_count)
    AMT_CREDIT_LIMIT_ACTUAL_df.columns = ['AMT_CREDIT_LIMIT_ACTUAL_UNIQUE_COUNT']
    AMT_CREDIT_LIMIT_ACTUAL_df = AMT_CREDIT_LIMIT_ACTUAL_df.reset_index()

    df_list = [target, count_df,str_df,month_balance_df,
            num_df, sk_dpd_df, SK_DPD_DEF_df, AMT_CREDIT_LIMIT_ACTUAL_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_merge_6month_df = getComp(temp_merge_df, id_name, target_name, str_columns)

    '''
    按照12个月提取所有的特征
    '''
    temp = comp_df[comp_df['MONTHS_BALANCE']>-13]
    count_df = temp.groupby(['SK_ID_CURR']).agg({'SK_ID_PREV':'count'})
    count_df.columns = ['BUREAU_COUNT']
    count_df = count_df.reset_index()
    # 2: 统计NAME_CONTRACT_STATUS的种类
    str_columns = ['NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF']
    str_df = get_str_df(['NAME_CONTRACT_STATUS'],temp,'SK_ID_CURR')
    month_balance_df = month_balance_df.reset_index()
    # 4: 对所有num_cols提取统计特征
    cols = temp.columns
    num_columns = list(cols[~cols.isin(['SK_ID_CURR','TARGET','SK_ID_PREV','NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF'])])
    num_df = get_num_df(temp,'',num_columns)
    num_df = num_df.fillna(0)
    # 5: 对SK_DPD，SK_DPD_DEF去零后求max，min，count
    temp_df = temp[['SK_ID_CURR']].drop_duplicates()
    filter_df = temp[comp_df['SK_DPD']!=0]
    sk_dpd_df = filter_df.groupby('SK_ID_CURR').agg({'SK_DPD':{'count','max','min'}})
    sk_dpd_df.columns = list(map(lambda x:"SK_DPD"+x,['_MIN','_MAX','_COUNT']))
    sk_dpd_df = sk_dpd_df.reset_index()
    filter_df = temp[temp['SK_DPD_DEF']!=0]
    SK_DPD_DEF_df = filter_df.groupby('SK_ID_CURR').agg({'SK_DPD_DEF':{'count','max','min'}})
    SK_DPD_DEF_df.columns = list(map(lambda x:"SK_DPD_DEF_df"+x,['_MIN','_MAX','_COUNT']))
    SK_DPD_DEF_df = SK_DPD_DEF_df.reset_index()
    # AMT_CREDIT_LIMIT_ACTUAL
    AMT_CREDIT_LIMIT_ACTUAL_df = temp.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].apply(unique_count)
    AMT_CREDIT_LIMIT_ACTUAL_df.columns = ['AMT_CREDIT_LIMIT_ACTUAL_UNIQUE_COUNT']
    AMT_CREDIT_LIMIT_ACTUAL_df = AMT_CREDIT_LIMIT_ACTUAL_df.reset_index()

    df_list = [target, count_df,str_df,month_balance_df,
            num_df, sk_dpd_df, SK_DPD_DEF_df, AMT_CREDIT_LIMIT_ACTUAL_df]
    temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
    high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
    cols = [id_name,target_name]
    cols.extend(list(high_corr.keys()))
    temp_merge_df = temp_merge_df[cols]
    comp_merge_12month_df = getComp(temp_merge_df, id_name, target_name, str_columns)

    df_list = [comp_merge_df, comp_merge_name_df, comp_merge_3month_df, comp_merge_6month_df, comp_merge_12month_df]
    temp = get_merge_df(df_list,['SK_ID_CURR','TARGET'])
    cols = temp.columns
    temp.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]



