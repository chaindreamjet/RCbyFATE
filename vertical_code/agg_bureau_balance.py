
import warnings
warnings.filterwarnings("ignore")


from hom.preprocessing import *
from hom.comp import missToComp,getComp
from hom.eda import *

if __name__ == "__main__":
    raw = pd.read_csv('raw_data/bureau_balance.csv')
    bureau = pd.read_csv('raw_data/bureau.csv')
    bureau = bureau[['SK_ID_CURR','SK_ID_BUREAU']]
    raw = pd.merge(bureau,raw,on='SK_ID_BUREAU')
    target = pd.read_csv("target.csv")
    raw = pd.merge(target,raw,on='SK_ID_CURR')
    raw = raw.sort_values(['SK_ID_BUREAU','MONTHS_BALANCE'])
    id_name = 'SK_ID_CURR'
    target_name = 'TARGET'

    ## 没有缺失
    comp_df = raw.copy()

    # 1: 直接根据SK_ID_CURR进行统计
    ## 1.1: 无时间窗口统计
    ### 1.1.1 计数
    count_df = comp_df.groupby(['SK_ID_CURR']).agg({'SK_ID_BUREAU':'count'})
    count_df.columns = ['bureau_balance_COUNT']
    count_df = count_df.reset_index()
    unique_count_df = comp_df.groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].apply(unique_count)
    unique_count_df.columns = ['bureau_balance_UNIQUE_COUNT']
    unique_count_df = unique_count_df.reset_index()

    ### 1.1.2 统计STATUS的个数
    str_df = get_str_df(['STATUS'],comp_df,'SK_ID_CURR')

    ### 1.1.3 统计STATUS中为1,2,3,4,5的次数
    filter_df = comp_df[comp_df['STATUS'].isin(['1','2','3','4','5'])]
    count_default_df = filter_df.groupby(['SK_ID_CURR']).agg({'SK_ID_BUREAU':'count'})
    count_default_df.columns = ['bureau_balance_default_COUNT']
    count_default_df = count_default_df.reset_index()
    unique_default_count_df = filter_df.groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].apply(unique_count)
    unique_default_count_df.columns = ['bureau_balance_default_UNIQUE_COUNT']
    unique_default_count_df = unique_default_count_df.reset_index()
    ### 对MONTHS_BALANCE记录总长度，记录最近一次和最远一次分别是什么时候
    month_balance_default_df = filter_df.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE':{'count','max','min'}})
    month_balance_default_df.columns = ['MONTHS_BALANCE_FIRST_RECORD_TIME_DEFAULT',
    'MONTHS_BALANCE_LATEST_RECORD_TIME_DEFAUL','MONTH_BALANCE_COUNTS_DEFAUL']
    month_balance_default_df = month_balance_default_df.reset_index()

    ### 1.1.4 统计STATUS中为4,5的次数
    filter_df = comp_df[comp_df['STATUS'].isin(['4','5'])]
    count_sever_df = filter_df.groupby(['SK_ID_CURR']).agg({'SK_ID_BUREAU':'count'})
    count_sever_df.columns = ['bureau_balance_sever_COUNT']
    count_sever_df = count_sever_df.reset_index()
    unique_sever_count_df = filter_df.groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].apply(unique_count)
    unique_sever_count_df.columns = ['bureau_balance_sever_UNIQUE_COUNT']
    unique_sever_count_df = unique_sever_count_df.reset_index()
    ### 对MONTHS_BALANCE记录总长度，记录最近一次和最远一次分别是什么时候
    month_balance_sever_df = comp_df.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE':{'count','max','min'}})
    month_balance_sever_df.columns = ['MONTHS_BALANCE_FIRST_RECORD_TIME_SEVER',
    'MONTHS_BALANCE_LATEST_RECORD_TIME_SEVER','MONTH_BALANCE_COUNTS_SEVER']
    month_balance_sever_df = month_balance_sever_df.reset_index()

    ### 1.1.5: 对MONTHS_BALANCE记录总长度，记录最近一次和最远一次分别是什么时候
    month_balance_df = comp_df.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE':{'count','max','min'}})
    month_balance_df.columns = ['MONTHS_BALANCE_FIRST_RECORD_TIME',
    'MONTHS_BALANCE_LATEST_RECORD_TIME','MONTH_BALANCE_COUNTS']
    month_balance_df = month_balance_df.reset_index()

    def get_time_window_df(window, comp_df, id_name, target_name, target):
        # 1: 直接根据SK_ID_CURR进行统计
        ### 1.1.1 计数
        comp_df = comp_df[comp_df['MONTHS_BALANCE']>window]
        count_df = comp_df.groupby(['SK_ID_CURR']).agg({'SK_ID_BUREAU':'count'})
        count_df.columns = ['bureau_balance_COUNT']
        count_df = count_df.reset_index()
        unique_count_df = comp_df.groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].apply(unique_count)
        unique_count_df.columns = ['bureau_balance_UNIQUE_COUNT']
        unique_count_df = unique_count_df.reset_index()

        ### 1.1.2 统计STATUS的个数
        str_df = get_str_df(['STATUS'],comp_df,'SK_ID_CURR')

        ### 1.1.3 统计STATUS中为1,2,3,4,5的次数
        filter_df = comp_df[comp_df['STATUS'].isin(['1','2','3','4','5'])]
        count_default_df = filter_df.groupby(['SK_ID_CURR']).agg({'SK_ID_BUREAU':'count'})
        count_default_df.columns = ['bureau_balance_default_COUNT']
        count_default_df = count_default_df.reset_index()
        unique_default_count_df = filter_df.groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].apply(unique_count)
        unique_default_count_df.columns = ['bureau_balance_default_UNIQUE_COUNT']
        unique_default_count_df = unique_default_count_df.reset_index()
        ### 对MONTHS_BALANCE记录总长度，记录最近一次和最远一次分别是什么时候
        month_balance_default_df = filter_df.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE':{'count','max','min'}})
        month_balance_default_df.columns = ['MONTHS_BALANCE_FIRST_RECORD_TIME_DEFAULT',
        'MONTHS_BALANCE_LATEST_RECORD_TIME_DEFAUL','MONTH_BALANCE_COUNTS_DEFAUL']
        month_balance_default_df = month_balance_default_df.reset_index()

        ### 1.1.4 统计STATUS中为4,5的次数
        filter_df = comp_df[comp_df['STATUS'].isin(['4','5'])]
        count_sever_df = filter_df.groupby(['SK_ID_CURR']).agg({'SK_ID_BUREAU':'count'})
        count_sever_df.columns = ['bureau_balance_sever_COUNT']
        count_sever_df = count_sever_df.reset_index()
        unique_sever_count_df = filter_df.groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].apply(unique_count)
        unique_sever_count_df.columns = ['bureau_balance_sever_UNIQUE_COUNT']
        unique_sever_count_df = unique_sever_count_df.reset_index()
        ### 对MONTHS_BALANCE记录总长度，记录最近一次和最远一次分别是什么时候
        month_balance_sever_df = comp_df.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE':{'count','max','min'}})
        month_balance_sever_df.columns = ['MONTHS_BALANCE_FIRST_RECORD_TIME_SEVER',
        'MONTHS_BALANCE_LATEST_RECORD_TIME_SEVER','MONTH_BALANCE_COUNTS_SEVER']
        month_balance_sever_df = month_balance_sever_df.reset_index()

        ### 1.1.5: 对MONTHS_BALANCE记录总长度，记录最近一次和最远一次分别是什么时候
        month_balance_df = comp_df.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE':{'count'}})
        month_balance_df = month_balance_df.reset_index()
        
        df_list = [target, count_df, unique_count_df, count_default_df, unique_default_count_df, str_df, 
            count_sever_df, unique_sever_count_df, month_balance_sever_df, month_balance_df]
        temp_merge_df = get_merge_df(df_list,'SK_ID_CURR')
        cols = temp_merge_df.columns
        temp_merge_df.columns = [x if x=='SK_ID_CURR' or x=='TARGET' else 'x'+str(i) for i,x in enumerate(cols)]
        high_corr = get_high_corr(target_name, temp_merge_df, id_name, 0.03)
        cols = [id_name,target_name]
        cols.extend(list(high_corr.keys()))
        temp_merge_df = temp_merge_df[cols]
        comp_temp_name_df = getComp(temp_merge_df, False, id_name, target_name, ['STATUS'])
        return comp_temp_name_df

    temp_window3_df = get_time_window_df(-4, comp_df, id_name, target_name, target)
    temp_window6_df = get_time_window_df(-7, comp_df, id_name, target_name, target)
    temp_window9_df = get_time_window_df(-10, comp_df, id_name, target_name, target)
    temp_window12_df = get_time_window_df(-13, comp_df, id_name, target_name, target)
    temp_window15_df = get_time_window_df(-16, comp_df, id_name, target_name, target)
