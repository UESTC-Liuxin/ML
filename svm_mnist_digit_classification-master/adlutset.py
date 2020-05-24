# Data Manipulation
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import os
import math
import matplotlib.pyplot as plt

#获取数据集的目录
root=os.getcwd()
dataset_root=os.path.join(os.path.dirname(root),'DataSet','Adult')

class AdultSet(object):
    def __init__(self):
        pass

    def load(self):
        pass



def data_analyze(data):
    #step1 绘制正负样本在
    # 分别取出大于50K的和小于50K的数据
    pd_up = data[data.Income == '>50K']
    pd_low = data[data.Income == '<=50K']
    cols = 5
    rows = math.ceil(float(data.shape[1]) / cols)
    draw_columns = ['Age', 'Employment_Type', 'Education', 'School_Period',
               'Marital_Status', 'Employment_Area', 'Partnership', 'Ethnicity', 'Gender', 'Gain_Financial',
               'Losses_Financial',
               'Weekly_Working_Time', 'Birth_Country', 'Income']
    for i, column in enumerate(draw_columns):
        counts_up = pd_up[column].value_counts()
        counts_low = pd_low[column].value_counts()
        labels = counts_up.index
        h1=counts_up.values
        h2=counts_low.values
        width = len(labels)//4
        if width<10:
            width=10
        fig = plt.figure(column,figsize=(width,width))  # 指定了图的名称 和画布的大小

        plt.title(column)
        plt.bar(np.arange(len(counts_up.index)), h1,width=0.3,  alpha=0.8,
                         label='>50K')
        plt.bar(np.arange(len(counts_low.index)) + 0.5, h2, width=0.3,alpha=0.8,
                         label='<=50K')
        plt.xticks(np.arange(len(counts_up.index)), labels, fontsize=12,
                   rotation=90)  # 横坐标轴标签 rotation x轴标签旋转的角度
        plt.legend()
        plt.savefig(column)
        # plt.show()
    # plt.subplots_adjust(hspace=0.7, wspace=0.2)



# definition of columns
columns= ['Age','Employment_Type','Weighting_Factor', 'Education','School_Period',
        'Marital_Status','Employment_Area','Partnership','Ethnicity','Gender','Gain_Financial','Losses_Financial',
         'Weekly_Working_Time','Birth_Country','Income']

#非数字特征转换表
unnumric_feature_dict={
    'Marital_Status': {'Married-spouse-absent': 7, 'Married-civ-spouse': 6,
                       'Married-AF-spouse': 5, 'Divorced': 2, 'Never-married': 4, 'Separated': 3,'Widowed': 1},
     'Partnership': {'Wife': 1, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3,
                     'Other-relative': 4, 'Unmarried': 5},
     'Employment_Type': {'Private': 15, 'Self-emp-not-inc': 3, 'Self-emp-inc': 5, 'Federal-gov': 12,
                         'Local-gov': 8, 'State-gov': 10, 'Without-pay': 1, 'Never-worked': 0},
     'Employment_Area': {'Tech-support': 11, 'Craft-repair': 2, 'Other-service': 7, 'Sales': 10,
                         'Exec-managerial': 3, 'Prof-specialty': 12, 'Handlers-cleaners': 5,
                         'Machine-op-inspct': 6, 'Adm-clerical': 1,'Farming-fishing': 4,
                         'Transport-moving': 9, 'Priv-house-serv': 8, 'Protective-serv': 8,'Armed-Forces': 8},
     'Ethnicity': {'White': 5, 'Asian-Pac-Islander': 4, 'Amer-Indian-Eskimo': 3, 'Other': 2, 'Black': 1 },
     'Gender': {'Female': 1, 'Male': 0 }
}
origin_data = pd.read_csv(os.path.join(dataset_root,'adult.data'), names =columns)

#遍历整个数据集，把字符串中的空格删除，剔出值为?的
for i in columns:
    if not is_numeric_dtype(origin_data[i]):
        origin_data[i]=origin_data[i].str.replace(' ','')
        bool=origin_data[i].str.contains('\?')
        origin_data[i]=(origin_data[i])[~bool]
#删除空的
origin_data = origin_data[~(origin_data.isnull()).any(axis=1)]
#进行值赋值，用于创建全数值型dataset
dataset=origin_data.copy()
# dataset = dataset[dataset.astype(str) == ' ?']

dataset.replace(unnumric_feature_dict,inplace=True)
dataset.replace(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
                 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan',
                 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines',
                 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
                 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua',
                 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
                 'Peru', 'Hong', 'Holand-Netherlands'],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], inplace = True)

# print(dataset['Education'].value_counts().index)
# dataset.replace(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-U
# del dataset['education']
dataset.replace(['<=50K', '>50K'], [-1, 1], inplace = True)
# print(dataset)
# print(data_income.tail())

if __name__ == '__main__':
    data_analyze(origin_data)