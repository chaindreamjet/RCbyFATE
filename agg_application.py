import os
import sys
import time
import math
import tqdm
import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

from math import isnan
from hom.preprocessing import *
from hom.comp import missToComp
from hom.eda import *



if __name__ == "__main__":
    '''
    read raw data
    '''
    raw = pd.read_csv("./raw_data/application_train.csv")
    '''
    eda
    '''
    # 1: draw binary distribution
    drawDistribution(raw, 'TARGET', 'SK_ID_CURR', 'Home Credit')

    '''
    Category columns: ordnial variables & unordnial variables
    1: NAME_EDUCATION_TYPE: Secondary / secondary special->0 Lower secondary->1 
    Incomplete higher->2 Higher education->3 Academic degree->4
    2: WEEKDAY_APPR_PROCESS_START: 'FRIDAY':5, 'MONDAY':1, 'SATURDAY':6, 'SUNDAY':7, 'THURSDAY':4, 'TUESDAY':2,
       'WEDNESDAY':3
    '''
    # NAME_EDUCATION_TYPE
    orderDic = {'Secondary / secondary special':0,'Lower secondary':1,'Incomplete higher':2,
        'Higher education':3,'Academic degree':4}
    NAME_EDUCATION_TYPE = [orderDic[x] for x in raw['NAME_EDUCATION_TYPE'].tolist()]
    # WEEKDAY_APPR_PROCESS_START
    orderDic = {'FRIDAY':5, 'MONDAY':1, 'SATURDAY':6, 'SUNDAY':7, 'THURSDAY':4, 'TUESDAY':2,
        'WEDNESDAY':3}
    WEEKDAY_APPR_PROCESS_START = [orderDic[x] for x in raw['WEEKDAY_APPR_PROCESS_START'].tolist()]

    # 3: str_columns->One-Hot Encoding
    str_columns = ['NAME_EDUCATION_TYPE','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR',
              'FLAG_OWN_REALTY','NAME_TYPE_SUITE','NAME_INCOME_TYPE',
              'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',
              'OCCUPATION_TYPE','ORGANIZATION_TYPE',
               'FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE',
               'EMERGENCYSTATE_MODE','WEEKDAY_APPR_PROCESS_START']
    str_df = get_str_df(str_columns,raw,'SK_ID_CURR')
    str_df = str_df.fillna(0)

    # 4: merge category columns
    str_columns = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR',
              'FLAG_OWN_REALTY','NAME_TYPE_SUITE','NAME_INCOME_TYPE',
              'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',
              'OCCUPATION_TYPE','ORGANIZATION_TYPE',
               'FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE',
               'EMERGENCYSTATE_MODE']
    non_str_raw = raw.drop(str_columns,axis=1)
    non_str_raw['NAME_EDUCATION_TYPE'] = NAME_EDUCATION_TYPE
    non_str_raw['WEEKDAY_APPR_PROCESS_START'] = WEEKDAY_APPR_PROCESS_START
    non_str_raw = pd.merge(non_str_raw,str_df,on='SK_ID_CURR')

    # 5: using special rule to complete part of numerical data
    OWN_CAR_AGE = raw['OWN_CAR_AGE'].tolist()
    FLAG_OWN_CAR = raw['FLAG_OWN_CAR'].tolist()
    OWN_CAR_AGE = [0 if isnan(x)==True and FLAG_OWN_CAR[i]=='N' else x for i,x in enumerate(OWN_CAR_AGE) ]
    raw['OWN_CAR_AGE'] = OWN_CAR_AGE

    # 6: miss to complete
    cols = non_str_raw.columns
    nan_df = pd.DataFrame(non_str_raw[cols[~cols.isin(['SK_ID_CURR','TARGET'])]].isnull().sum(),columns=['counts'])
    miss_columns = nan_df[nan_df['counts']!=0].index.tolist()
    dispersed_columns = str_df.columns

    comp_df = pd.DataFrame(columns = cols)
    for col in cols:
        if col in miss_columns:
            if col in dispersed_columns:
                dispersed = True
            else:
                dispersed = False
            isTime = False
            pad = missToComp(non_str_raw,col,id_name, target_name,dispersed=dispersed,isTime=isTime)
            comp_df[col] = pad.getBestCompResult()
        else:
            comp_df[col] = non_str_raw[col].tolist()   

    # save table
    comp_df.to_csv("fate/fate_application.csv",index=False)  
