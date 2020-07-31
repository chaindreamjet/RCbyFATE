import os
import sys
import time
import numpy as np
import pandas as pd


def unique_count(x):
    return len(list(set(x)))

def record(x):
    x.columns = ['0','1']
    date = x['0'].tolist()
    records = x['1'].tolist()
    return [records[date.index(min(date))],records[date.index(max(date))]]

def default_time(x):
    x.columns = ['0','1']
    date = x['0'].tolist()
    prolong = x['1'].tolist()
    if sum(prolong)==0:
        out = [np.nan,np.nan]
    else:
        out = [min([abs(date[i]) for i,x in enumerate(prolong) if x!=0]),max([abs(date[i]) for i,x in enumerate(prolong) if x!=0])]
    return out

def get_nan_info(df):
    nan_df = pd.DataFrame(df.isnull().sum(),columns=['counts'])
    nan_df = nan_df.drop(nan_df[nan_df['counts']==0].index)
    nan_df['percent'] = np.array(nan_df['counts'].values)/len(df)
    nan_df['percent'] = nan_df['percent'].apply(lambda x:format(x,".2%"))
    print("Input size is (%d,), the nan size is (%d,)"%(df.shape[1],nan_df.shape[0]))
    nan_df = nan_df.sort_values(['percent'])
    nan_df = nan_df.reset_index(drop=False)
    pd.set_option('display.max_rows', None)
    print(nan_df)
    return nan_df

def get_high_corr(target, df, id_name, threshold):
    columns = list(df.columns)
    if id_name in columns:
        columns.remove(id_name)
    columns.remove(target)
    high_corr = {}

    t0 = time.time()
    _output = sys.stdout
    k = len(columns)

    for index, name in enumerate(columns):
        corr_single = df[target].corr(df[name])
        if abs(corr_single) > threshold:
            high_corr[name] = corr_single
        t1 = time.time()
        _output.write(f'\rTotal number is:{k:.0f} complete number :{index+1:.0f} used time:{t1-t0:.4f}')
    return high_corr

def process_time(x):
    dic = {}
    datetime = time.ctime(x)
    dic['week'] = datetime.split()[0]
    ts = pd.to_datetime(datetime)
    dic['year'],dic['month'],dic['day'],dic['hour'] = ts.year, ts.month, ts.day, ts.hour
    dic['second'],dic['minute'] = ts.second, ts.minute
    return dic

def get_str_df(str_columns,raw_df,id_name):
    '''
    处理类别变量
    '''
    df = pd.DataFrame(columns=[id_name])
    df[id_name] = sorted(list(set(raw_df[id_name].values)))

    t0 = time.time()
    _output = sys.stdout
    k = len(str_columns)

    for index,item in enumerate(str_columns):
        count_df = pd.DataFrame(raw_df.groupby([id_name])[item].value_counts())
        unstack_df = count_df.unstack(item).reset_index()

        columns = [id_name]
        unstack_columns = list(unstack_df.columns.levels[1])
        if "" in unstack_columns:
            unstack_columns.remove("")
        columns.extend([item+"_"+str(x)+"_COUNT" for x in unstack_columns])
        rename_df = pd.DataFrame(unstack_df.values,columns=columns)
        rename_df = rename_df.dropna(axis=1,how='all')
        rename_df = rename_df.fillna(0)

        df = pd.merge(df,rename_df,on=id_name,how='left')

        t1 = time.time()
        _output.write(f'\rTotal number is:{k:.0f} complete number :{index+1:.0f} used time:{t1-t0:.4f}')
    return df


def get_num_df(raw, name, num_columns, groupby_rule='SK_ID_CURR',
               id_name='SK_ID_CURR',
               stat_func=['sum', 'mean', 'median', 'min', 'max', 'std']):
    t0 = time.time()
    _output = sys.stdout
    k = len(num_columns)

    num_df = raw[[id_name]].drop_duplicates().sort_values(id_name)
    for index, col in enumerate(num_columns):
        df0 = raw.groupby(groupby_rule).agg({col: stat_func})
        if name:
            col_name = col + "_" + name
        else:
            col_name = col
        df0.columns = [col_name + "_" + x.upper() for x in stat_func]
        df0 = df0.reset_index()
        num_df = pd.merge(num_df, df0, on=id_name )
        t1 = time.time()
        _output.write(f'\rTotal number is:{k:.0f} complete number :{index+1:.0f} used time:{t1-t0:.4f}')
    return num_df

def get_merge_df(df_list,id_name):
    merge_df = df_list[0]
    for index in range(1,len(df_list)):
        merge_df = pd.merge(merge_df,df_list[index],on=id_name,how='outer')
    return merge_df


def get_second_str_df(str_columns, raw_df, id_name, secondary_rule):
    '''
    处理类别变量
    '''
    df = pd.DataFrame(columns=[id_name])
    df[id_name] = sorted(list(set(raw_df[id_name].values)))

    t0 = time.time()
    _output = sys.stdout
    k = len(str_columns)

    for index, item in enumerate(str_columns):
        count_df = raw_df.groupby([id_name, secondary_rule]).agg({item: 'value_counts'})
        unstack_df = count_df.unstack([secondary_rule, item])

        columns = [id_name]
        unstack_columns = list(unstack_df.columns)
        columns.extend(unstack_columns)

        unstack_df = unstack_df.reset_index()
        rename_df = pd.DataFrame(unstack_df.values, columns=columns)
        rename_df = rename_df.dropna(axis=1, how='all')
        rename_df = rename_df.fillna(0)

        df = pd.merge(df, rename_df, on=id_name, how='left')

        t1 = time.time()
        _output.write(f'\rTotal number is:{k:.0f} complete number :{index+1:.0f} used time:{t1-t0:.4f}')
    return df


def get_second_num_df(raw, num_columns, secondary_rule, groupby_rule='SK_ID_CURR',
                      id_name='SK_ID_CURR',
                      stat_func=['sum', 'mean', 'median', 'min', 'max', 'std']):
    t0 = time.time()
    _output = sys.stdout
    k = len(num_columns)

    num_df = raw[[id_name]].drop_duplicates().sort_values(id_name)
    for index, col in enumerate(num_columns):
        df0 = raw.groupby([groupby_rule, secondary_rule]).agg({col: stat_func})
        df0 = df0.unstack(secondary_rule).reset_index()
        columns = df0.columns

        temp = pd.DataFrame(df0.values, columns=columns)

        num_df = pd.merge(num_df, temp, on=id_name)
        t1 = time.time()
        _output.write(f'\rTotal number is:{k:.0f} complete number :{index+1:.0f} used time:{t1-t0:.4f}')
    return num_df
