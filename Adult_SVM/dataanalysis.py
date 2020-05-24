import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as preprocessing

# Importing the Dataset

names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]






class Adult_analyze(object):

    def __init__(self,path):
        self.dataset_train = pd.read_csv(path, sep=',', names=names)
        self.origin_train=self.dataset_train.copy()
        self.dataset_test = pd.read_csv('adult.test', sep=',', names=names, skiprows=[0])
        self.origin_test = self.dataset_test.copy()

    def conclusion(self):
        print("How many of each class do we have:" + str(np.unique(self.dataset_train['income'], return_counts=True)))
        print("How many of each class do we have:" + str(np.unique(self.dataset_test['income'], return_counts=True)))
        self.dataset_test['income'] = self.dataset_test['income'].str[:-1]
        dataset = pd.concat([self.dataset_train, self.dataset_test])
        adult_describe=dataset.describe()
        format = lambda x: '%.2f' % x
        adult_describe=adult_describe.applymap(format)
        fig=plt.figure(figsize=(10,5))
        colors = plt.cm.BuPu(np.linspace(0, 0.5, len(adult_describe.index)))
        colors = colors[::-1]
        plt.table(cellText=adult_describe.to_numpy(),colColours=colors,
                  rowLabels=list(adult_describe.index),colLabels=list(adult_describe.columns),
                  bbox=(0,0,1,1),loc='center')
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴
        plt.axis('off')  # 去掉坐标轴
        plt.savefig(os.path.join('Plots','describe'))
        # Setting all the categorical columns to type category
        for col in set(dataset.columns) - set(dataset.describe().columns):
            dataset[col] = dataset[col].astype('category')
        # Dropping the Missing Values
        dataset['native-country'] = dataset['native-country'].replace(' ?', np.nan)
        dataset['workclass'] = dataset['workclass'].replace(' ?', np.nan)
        dataset['occupation'] = dataset['occupation'].replace(' ?', np.nan)

        dataset.dropna(how='any', inplace=True)
        dataset.drop(labels=['education', 'fnlwgt', 'hours-per-week'], axis=1, inplace=True)
        y = np.asarray(dataset['income'])

        print("Amount of classes:" + str(np.unique(y, return_counts=True)))

    def distributed_map(self):
        fig = plt.figure(figsize=(20, 15))
        cols = 5
        rows = math.ceil(float(self.origin_train.shape[1]) / cols)
        for i, column in enumerate(self.origin_train.columns):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title(column)
            if self.origin_train.dtypes[column] == np.object:
                self.origin_train[column].value_counts().plot(kind="bar", axes=ax)
            else:
                self.origin_train[column].hist(axes=ax)
                plt.xticks(rotation="vertical")
        plt.subplots_adjust(hspace=0.7, wspace=0.2)
        plt.savefig(os.path.join('Plots','distributed_map'))

    def correlations(self):
        #计算数据的相关性
        fig = plt.figure()
        result = self.origin_train
        encoders = {}
        for column in result.columns:
            if result.dtypes[column] == np.object:
                encoders[column] = preprocessing.LabelEncoder()
                result[column] = encoders[column].fit_transform(result[column])
        # Calculate the correlation and plot it
        sns.heatmap(result.corr(), square=True)
        plt.savefig(os.path.join('Plots','correlations'))









# print(dataset)
# print(data_income.tail())

if __name__ == '__main__':
    path='adult.data'
    analyze=Adult_analyze(path)
    analyze.conclusion()
    # analyze.distributed_map()
    analyze.correlations()