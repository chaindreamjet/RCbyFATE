import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


'''
draw the binary category distribution 
'''
def drawDistribution(raw, target_name, id_name, table_name):
    values = raw.groupby(target_name)[id_name].count().reset_index()
    values['percentage'] = values[id_name]/values[id_name].sum()
    values = values[[target_name, 'percentage']]
    
    colors = ["dimgray", "lightgrey"]

    plt.Figure(figsize=(8,8))
    g = sns.barplot(x=target_name, y='percentage',data=values, palette=colors)
    for index,row in values.iterrows():
        #在柱状图上绘制该类别的数量 
            g.text(row.name,row.percentage,str(round(row.percentage*100,2))+"%",
                   color="black",
                   ha="center")
    titleName = table_name + f" Default Distributions \n (0: No Fraud || 1: Fraud)"
    plt.title(titleName, fontsize=12)

'''
draw the distribution of numerical data
'''
def get_compare_displot(raw, feature_name,target_name):
    fig, ax = plt.subplots(1, 2, figsize=(18,4))
    
    # log
    raw['LOG_'+feature_name] = np.log(raw[feature_name])
    feature_name = 'LOG_'+feature_name
    # split dataset into label 0 and label 1
    amount0_val = raw.loc[raw[target_name]==0,feature_name].values
    amount1_val = raw.loc[raw[target_name]==1,feature_name].values

    # get the lower bound and upper bound
    min_amount0,min_amount1  = min(amount0_val), min(amount1_val)
    max_amount0,max_amount1  = max(amount0_val), max(amount1_val)
    lower, upper = min(min_amount0,min_amount1), min(max_amount0,max_amount1)
    
    sns.distplot(amount0_val, ax=ax[0], color='dimgrey')
    ax[0].set_title('Label:0 Distribution of {}'.format(feature_name), fontsize=14)
    ax[0].set_xlim([lower, upper])

    sns.distplot(amount1_val, ax=ax[1], color='dimgrey')
    ax[1].set_title('Label:1 Distribution of {}'.format(feature_name), fontsize=14)
    ax[1].set_xlim([lower, upper])

    plt.show()

'''
draw the correlation 
'''

def get_corr_magnitude(df, target_name, id_name, table_name):
    corr = df.corr().loc[target_name].drop([target_name, id_name])
    bar_colors = ['dimgray' if x else 'lightgrey' for x in list(corr.values<0)]
    color_labels = {'dimgray':'Negative correlation','lightgrey':'Positive correlation'}

    corr = corr.apply(np.abs)
    
    fig, ax = plt.subplots(figsize=(8,12))
    plt.barh(range(len(corr)),corr.values, color=bar_colors)
    plt.xlabel('|Correlation|')
    plt.yticks(range(len(corr)), corr.index.tolist())
    plt.title("Magnitude of correlation with label Data source:{}".format(table_name))
    plt.ylim([-1,len(corr)])
    for col,lab in color_labels.items():
        plt.plot([],linestyle='', marker='s', c=col,label=lab)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines[-len(color_labels.keys()):], labels[-len(color_labels.keys()):], loc='upper right')
    plt.show()