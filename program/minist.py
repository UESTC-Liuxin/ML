import os
import numpy as np

class Mnist(object):

    def __init__(self,root=None,train=True,batch_size=1):
        self.dataname = "Mnist"
        self.dims = 28 * 28
        self.shape = [28, 28, 1]
        self.image_size = 28
        self.root=root
        self.train=train
        self.batch_size=batch_size
        self.data, self.targets = self.load_mnist()

    def load_mnist(self):
        # data_dir = os.path.join("DataSet", "MNIST\\raw")
        if self.train:
            fdata = open(os.path.join(self.root, 'train-images-idx3-ubyte'))
            flabels = open(os.path.join(self.root, 'train-labels-idx1-ubyte'))
        else:
            fdata = open(os.path.join(self.root, 't10k-images-idx3-ubyte'))
            flabels = open(os.path.join(self.root, 't10k-labels-idx1-ubyte'))
        # 利用np.fromfile语句将这个ubyte文件读取进来
        # 需要注意的是用np.uint8的格式
        # 还有读取进来的是一个一维向量
        # <type 'tuple'>: (47040016,)，这就是loaded变量的读完之后的数据类型
        loaded = np.fromfile(file=fdata, dtype=np.uint8)
        data = loaded[16:].reshape((-1, 28, 28, 1)).astype(np.float)

        loaded = np.fromfile(file=flabels, dtype=np.uint8)
        targets = loaded[8:].reshape((data.shape[0])).astype(np.float)

        targets = np.asarray(targets)
        return data / 255., targets

    def __getitem__(self, item):
        return (self.data[item],self.targets[item])

    def __len__(self):
        return self.targets.shape[0]



# class DataLoader(object):
#     def __init__(self,dataset,shuffle=True,batch_size=1):
#         self.dataset=dataset
#         self.batch_size=batch_size
#         self.data_lenth=len(self.dataset)
#         self.index=0-self.batch_size
#         if self.shuffle:
#             # 目的是为了打乱数据集
#             self.sample_list=range(self.data_lenth)
#             np.random.shuffle(self.sample_list)
#     def __iter__(self):
#         return
#     def __next__(self):
#         if self.index<self.data_lenth:
#             self.index+=self.batch_size
#             return self.sample_list[self.index:self.index+self.batch_size]
#         else:
#             self.index=0-self.batch_size
#             raise StopIteration



if __name__ == '__main__':
    dataset=Mnist(root=os.path.join('MNIST','raw'))
