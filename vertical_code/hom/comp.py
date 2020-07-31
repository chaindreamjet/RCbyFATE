from tqdm import tqdm
import numpy as np
import pandas as pd
from math import isnan
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor


'''
0: Check the loss ratio, if >80% -> drop it; else if 50%=<lr<80% -> binning(6); else -> 1
1: whether it is a time series Yes-> 2 No-> 3
2: time series, zero, group HM, group mean, group median ,ffill, bfill, interpolate
3: lr<20% -> 4;20%=<lr<50% ->5
4: we applied zero, HM, mean, median, KNN, KMeans 
5: we applied zero, HM, KNN, KMeans 
6: binning -> Equal frequency/ Equal distance/ Chimerge

基类filling->继承Binning(Filling)->继承MissToComp(Binning)
'''

class Filling():
    def __init__(self, df, col_name, id_name, target_name, dispersed, isTime):
        self.df = df
        self.col_name = col_name
        self.id_name = id_name
        self.target_name = target_name
        self.dispersed = dispersed
        self.isTime = isTime
        self.lr = self.df[self.col_name].isnull().sum()/self.df.shape[0]

    def zeroPad(self):
        self.df[self.col_name + "_ZERO"] = self.df[self.col_name].fillna(0)
        print("Zero padding")
        return self.df[self.col_name + "_ZERO"].tolist()

    def halfMinimunPad(self):
        self.df[self.col_name + "_HM"] = self.df[self.col_name].fillna(0.5*self.df[self.col_name].min())
        print('HM padding')
        return self.df[self.col_name + "_HM"].tolist()

    def meanPad(self):
        self.df[self.col_name + "_MEAN"] = self.df[self.col_name].fillna(self.df[self.col_name].mean())
        print("Mean padding")
        return self.df[self.col_name + "_MEAN"].tolist()

    def medianPad(self):
        self.df[self.col_name + "_MEDIAN"] = self.df[self.col_name].fillna(self.df[self.col_name].median())
        print('Median padding')
        return self.df[self.col_name + "_MEDIAN"].tolist()

    '''
    time series
    '''
    def groupHMPad(self):
        temp = self.df[[self.id_name, self.col_name,]]
        temp = temp.dropna()
        groupby = temp.groupby(self.id_name).agg({self.col_name:'min'})
        nan_values = self.df[self.col_name].tolist()
        pad_values = []
        for i, x in enumerate(nan_values):
            if isnan(x)==True:
                try:
                    pad_values.append(0.5*groupby[self.df.loc[i, self.id_name]])
                except:
                    pad_values.append(self.df[self.col_name].mean())
            else:
                pad_values.append(x)
        self.df[self.col_name+'_GOURP_HM'] = pad_values
        print("Group HM padding")
        return self.df[self.col_name+'_GOURP_HM'].tolist()
    
    ## 可调用mean, median
    def group(self, func):
        temp = self.df[[self.id_name, self.col_name,]]
        temp = temp.dropna()
        groupby = temp.groupby(self.id_name).agg({self.col_name:func})
        nan_values = self.df[self.col_name].tolist()
        pad_values = []
        for i,x in enumerate(nan_values):
            if isnan(x)==True:
                try:
                    pad_values.append(groupby[self.df.loc[i,self.id_name]])
                except:
                    pad_values.append(self.df[self.col_name].mean())
            else:
                pad_values.append(x)
        self.df[self.col_name+'_GROUP_'+func.upper()] = pad_values
        print("Group {} padding".format(func))
        return self.df[self.col_name+'_GROUP_'+func.upper()].tolist()

    # def groupMean(self):

    def forwardPad(self):
        self.df[self.col_name + '_FFILL'] = self.df[self.col_name].ffill()
        print("Forward padding")
        return self.df[self.col_name + "_FFILL"].tolist()

    def backPad(self):
        self.df[self.col_name + '_BFILL'] = self.df[self.col_name].bfill()
        print("Back padding")
        return self.df[self.col_name + "_BFILL"].tolist()

    def interpolatePad(self):
        self.df[self.col_name + '_INTERPOLATE'] = self.df[self.col_name].interpolate()
        print('Interpolated padding')
        return self.df[self.col_name + "_INTERPOLATE"].tolist()


    # def knn_missing_filled(self, x_train, y_train, x_test, k=6, dispersed=False):
    #     if dispersed:
    #         clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
    #     else:
    #         clf = KNeighborsRegressor(n_neighbors=k, weights='distance')
    #     clf.fit(x_train, y_train)
    #     return clf.predict(x_test)

    def knnPad(self):
        cols = self.df.columns
        nan_df = pd.DataFrame(self.df[cols[~cols.isin([self.id_name, self.target_name])]].isnull().sum(),
                              columns=['counts'])
        unmiss_cols = nan_df[nan_df['counts'] == 0].index.tolist()
        target_value = self.df[self.col_name].tolist()

        # check column type

        unmiss_cols = [x for x in unmiss_cols if type(self.df[x].tolist()[0]) != str]

        miss_index = [i for i, x in enumerate(target_value) if isnan(x) == True]
        comp_index = [i for i, x in enumerate(target_value) if isnan(x) == False]

        x = self.df[unmiss_cols]
        scaler = MinMaxScaler()
        scaler_x = scaler.fit_transform(x)
        x_train = scaler_x[comp_index[:min(10000, len(comp_index))]]
        x_test = scaler_x[miss_index]

        y = np.array(list(filter(lambda x:isnan(x) == False, target_value)))
        y = y.reshape(-1,1)
        y_train = scaler.fit_transform(y)
        y_temp = y_train
        y_train = y_train[:min(10000, len(comp_index))]
        y_train = y_train.reshape(-1, 1)
        # print(x_train.shape, y_train.shape)

        if self.dispersed:
            clf = KNeighborsClassifier(n_neighbors=6, weights='distance')
        else:
            clf = KNeighborsRegressor(n_neighbors=6, weights='distance')

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        self.df[self.col_name + "_KNN"] = 0
        self.df = self.df.reset_index(drop=True)
        self.df.loc[miss_index, self.col_name + "_KNN"] = y_pred
        self.df.loc[comp_index, self.col_name + "_KNN"] = y_temp
        print('KNN padding')
        return self.df[self.col_name + "_KNN"].tolist()

    def kmeansPad(self):
        cols = self.df.columns
        nan_df = pd.DataFrame(self.df[cols[~cols.isin([self.id_name, self.target_name])]].isnull().sum(),
                              columns=['counts'])
        unmiss_cols = nan_df[nan_df['counts'] == 0].index.tolist()
        target_value = self.df[[self.id_name, self.col_name]].reset_index(drop=True)
        unmiss_cols = [x for x in unmiss_cols if type(self.df[x].tolist()[0]) != str]
        
        x_train = self.df[unmiss_cols].values
        km = KMeans(n_clusters=6).fit(x_train)
        
        temp = pd.DataFrame(km.labels_, columns = ['labels'])
        target_value = target_value.reset_index(drop=True)
        temp = pd.concat([target_value, temp], axis=1)
        groupby = temp.dropna().groupby('labels').agg({self.col_name:'mean'})
 
        for index in groupby.index.tolist():
            temp.loc[temp['labels']==index, self.col_name] = temp[temp['labels']==index][self.col_name].fillna(np.squeeze(groupby.loc[index]))
        
        self.df[self.col_name + "_KMeans"] = temp[self.col_name].tolist()
        print("Kmeans padding")
        return self.df[self.col_name + "_KMeans"].tolist()

class Binning(Filling):
    def __init__(self,  df, col_name, id_name, target_name, dispersed, isTime):
        super().__init__(df, col_name, id_name, target_name, dispersed, isTime)
    
    def EqualDistanceBin(self,bins=6):
        edb_series = pd.cut(self.df[self.col_name], bins, labels=[x for x in range(1, bins+1)])
        edb_series = edb_series.cat.add_categories([0])
        edb_bin = list(edb_series.fillna(0))
        return edb_bin

    def EqualFrequencyBin(self,bins=6):
        k = len(list(set(pd.qcut(self.df[self.col_name], bins, duplicates='drop'))))
        efb_series = pd.qcut(self.df[self.col_name], bins, labels=[x for x in range(1, k)], duplicates='drop')
        efb_series = efb_series.cat.add_categories([0])
        efb_bin = list(efb_series.fillna(0))
        return efb_bin

    def ChiMerge(self, raw, variable, flag, confidenceVal=3.841, bin=10, sample=None):
        '''
        param df:DataFrame| 必须包含标签列
        param variable:str| 需要卡方分箱的变量名称（字符串）
        param flag:str    | 正负样本标识的名称（字符串）
        param confidenceVal:float| 置信度水平（默认是不进行抽样95%）
        param bin：int            | 最多箱的数目
        param sample: int          | 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
        note: 停止条件为大于置信水平且小于bin的数目
        return :DataFrame|采样结果
        '''    
        import pandas as pd
        import numpy as np
        
        
        #进行是否抽样操作
        if sample != None:
            raw = raw.sample(n=sample)
        else:
            raw
            
        #进行数据格式化录入
        total_num = raw.groupby([variable])[flag].count()  #统计需分箱变量每个值数目
        total_num = pd.DataFrame({'total_num': total_num})  #创建一个数据框保存之前的结果
        positive_class = raw.groupby([variable])[flag].sum()  #统计需分箱变量每个值正样本数
        positive_class = pd.DataFrame({'positive_class': positive_class})  #创建一个数据框保存之前的结果
        regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True,
                        how='inner')  # 组合total_num与positive_class
        regroup.reset_index(inplace=True)
        regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']  #统计需分箱变量每个值负样本数
        regroup = regroup.drop('total_num', axis=1)
        np_regroup = np.array(regroup)  #把数据框转化为numpy（提高运行效率）
        #print('已完成数据读入,正在计算数据初处理')

        #处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
        i = 0
        while (i <= np_regroup.shape[0] - 2):
            if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or ( np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
                np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
                np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
                np_regroup[i, 0] = np_regroup[i + 1, 0]
                np_regroup = np.delete(np_regroup, i + 1, 0)
                i = i - 1
            i = i + 1
    
        #对相邻两个区间进行卡方值计算
        chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
        for i in np.arange(np_regroup.shape[0] - 1):
            chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 \
            * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) / \
            ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * (
            np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
            chi_table = np.append(chi_table, chi)
        #print('已完成数据初处理，正在进行卡方分箱核心操作')

        #把卡方值最小的两个区间进行合并（卡方分箱核心）
        while (1) and len(chi_table)>0:
            # print(min(chi_table))
            if (len(chi_table) <= (bin - 1) and min(chi_table) >= confidenceVal):
                break
            chi_min_index = np.argwhere(chi_table == min(chi_table))[0]  # 找出卡方值最小的位置索引
            np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
            np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
            np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
            np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

            if (chi_min_index == np_regroup.shape[0] - 1):  # 最小值试最后两个区间的时候
                # 计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                            * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                        ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                # 删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index, axis=0)

            else:
                # 计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                        * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                        ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                # 计算合并后当前区间与后一个区间的卡方值并替换
                chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                        * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                    ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
                # 删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
        #print('已完成卡方分箱核心操作，正在保存结果')

        #把结果保存成一个数据框
        result_data = pd.DataFrame()  # 创建一个保存结果的数据框
        result_data['variable'] = [variable] * np_regroup.shape[0]  # 结果表第一列：变量名
        list_temp = []
        for i in np.arange(np_regroup.shape[0]):
            if i == 0:
                x = '0' + ',' + str(np_regroup[i, 0])
            elif i == np_regroup.shape[0] - 1:
                x = str(np_regroup[i - 1, 0]) + '+'
            else:
                x = str(np_regroup[i - 1, 0]) + ',' + str(np_regroup[i, 0])
            list_temp.append(x)
        result_data['interval'] = list_temp  #结果表第二列：区间
        result_data['flag_0'] = np_regroup[:, 2]  # 结果表第三列：负样本数目
        result_data['flag_1'] = np_regroup[:, 1]  # 结果表第四列：正样本数目

        return result_data

    def ChiMergeBin(self):
        # 随机抽样500000的数据进行分箱
        temp = shuffle(self.df)
        temp = temp[:min(temp.shape[0], 500000)]
        # print(temp)

        result_data = self.ChiMerge(temp, self.col_name, self.target_name, 10.645, 6, None)
        bins = [] #卡方的区间值
        bins.append(-float('inf'))
        for i in range(result_data["interval"].shape[0]-1):
            
            St = result_data["interval"][i].split(",")
            bins.append(float(St[1]))

        bins.append(float('inf'))
        chi_series = pd.cut(self.df[self.col_name], bins, labels=[x for x in range(1, len(bins))])
        chi_series = chi_series.cat.add_categories([0])
        chi_bin = list(chi_series.fillna(0))
        return chi_bin

    def IV_WOE(self,bin_name):
        d = self.df.groupby(bin_name,as_index=False).agg({self.target_name:['count','sum']})
        d.columns = ['CUTOFF','N','EVENTS']
        d['%_of_EVENTS'] = np.maximum(d['EVENTS'],0.5)/d['EVENTS'].sum()
        d['NON_EVENTS'] = d['N']-d['EVENTS']
        d['%_of_NON_EVENTS'] = np.maximum(d['NON_EVENTS'],0.5)/d['NON_EVENTS'].sum()
        d['WOE'] = np.log(d['%_of_EVENTS']/d['%_of_NON_EVENTS'])
        d['IV'] = d['WOE']*(d['%_of_EVENTS']-d['%_of_NON_EVENTS'])
        IV = d['IV'].sum()
        return IV

    def BestBin(self):
        self.df['edb_bin'] = self.EqualDistanceBin()
        self.df['efb_bin'] = self.EqualFrequencyBin()
        self.df['chi_bin'] = self.ChiMergeBin()
        IV_dic = {}
        for bin_name in ['edb_bin', 'efb_bin', 'chi_bin']:
            IV_dic[bin_name] = self.IV_WOE(bin_name)
        sorted_dic = sorted(IV_dic.items(), key=lambda x: x[1], reverse=True)
        best_bin_result = sorted_dic[0]
        print("The best binning result is {}, the iv value is {}".format(best_bin_result[0],best_bin_result[1]))
        return self.df[best_bin_result[0]].tolist()

class missToComp(Binning):
    def __init__(self,  df, col_name, id_name, target_name, dispersed, isTime):
        super().__init__(df, col_name, id_name, target_name, dispersed, isTime)
    
    def getBestCompResult(self):
        corr_df = self.df[[self.target_name]]
        corr_df = corr_df.copy()

        if self.lr==0:
            max_pad_way = self.col_name
            print("No loss!")
            return corr_df[max_pad_way].tolist()
        elif self.lr >0.8:
            print("Loss rate larger than 80%, drop it!")
            return None
        elif self.lr<=0.8 and self.lr>0.5:
            print("Loss rate is large, so binning!")
            return Binning.BestBin(self)
        else:
            if self.isTime:
                print("Time series padding, the loss ratio is {}".format(self.lr))
                # zero, group HM, group mean, group median ,ffill, bfill, interpolate
                corr_df[self.col_name+"_ZERO"] = Filling.zeroPad(self)
                corr_df[self.col_name+"_GROUP_HM"] = Filling.groupHMPad(self)
                corr_df[self.col_name+"_GROUP_MEAN"] = Filling.group(self,'mean')
                corr_df[self.col_name+"_GROUP_MEDIAN"] = Filling.group(self,'median')
                corr_df[self.col_name+"_ffill"] = Filling.forwardPad(self)
                corr_df[self.col_name+"_bfill"] = Filling.backPad(self)
                corr_df[self.col_name+"_interpolate"] = Filling.interpolatePad(self)
            else:
                print("Normal padding, the loss ratio is {}".format(self.lr))
                # zero, HM, mean, median, KNN, KMeans 
                corr_df[self.col_name+"_ZERO"] = Filling.zeroPad(self)
                corr_df[self.col_name+"_HM"] = Filling.halfMinimunPad(self)
                corr_df[self.col_name+"_KNN"] = Filling.knnPad(self)
                corr_df[self.col_name+"_KMeans"] = Filling.kmeansPad(self)
                if self.lr<=0.2:
                    corr_df[self.col_name+"_MEAN"] = Filling.meanPad(self)
                    corr_df[self.col_name+"_MEDIAN"] = Filling.medianPad(self)
            corr_relation = corr_df.corr()[self.target_name].drop(self.target_name)
            corr_index = list(corr_relation.index)
            corr_value = list(corr_relation.values)
            corr_value = list(map(lambda x:abs(x),corr_value))
            
            max_pad_way = corr_index[corr_value.index(max(corr_value))]
            print("The best padding way is {}, and correlation after padding is {}".format(max_pad_way,corr_relation[max_pad_way]))
            return corr_df[max_pad_way].tolist()
        # return corr_relation


def getComp(df, isTime, id_name, target_name, str_columns):
    cols = df.columns
    nan_df = pd.DataFrame(df[cols[~cols.isin([id_name, target_name])]].isnull().sum(), columns=['counts'])
    miss_columns = nan_df[nan_df['counts'] != 0].index.tolist()

    comp_df = pd.DataFrame(columns=cols)
    for col in tqdm(cols):
        if col in miss_columns:
            if col in str_columns:
                dispersed = True
            else:
                dispersed = False
            pad = missToComp(df, col, id_name, target_name, dispersed=dispersed, isTime=isTime)
            comp_df[col] = pad.getBestCompResult()
        else:
            comp_df[col] = df[col].tolist()
    return comp_df
# if __name__ == "__main__":
#     df = pd.read_csv('/Users/zhanghaha/3_project/homeCredit/raw_data/credit_card_balance.csv')
#     target = pd.read_csv('/Users/zhanghaha/3_project/homeCredit/target.csv')
#     df = pd.merge(df, target, on='SK_ID_CURR')
#     df = df.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'])
#     df = df[:1000]
#
#     pad = missToComp(df, 'AMT_DRAWINGS_ATM_CURRENT', 'SK_ID_CURR', 'TARGET', dispersed=False, isTime=False)
#     print(pad.getBestCompResult())



    
