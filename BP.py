import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer ,LabelEncoder#标签二值化
from sklearn.model_selection import train_test_split   #切割数据,交叉验证法
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve,auc,precision_recall_curve,f1_score,confusion_matrix,accuracy_score
from sklearn.model_selection import KFold,StratifiedKFold
from minist import Mnist

LR=0.1
BatchSize=1





def data_analyze(data,target):
    # 数据分布
    fig1 = plt.figure()
    plt.hist(target, bins=10, rwidth=0.8)
    plt.title('dataset histogram')
    plt.xlabel('class_id')
    plt.ylabel('class_num')
    # 图片抽样查看
    fig2 = plt.figure()
    images = np.squeeze(data[:20],axis=3)
    print(images.shape)
    for i in np.arange(1, 21):
        plt.subplot(5, 4, i)
        plt.imshow(images[i - 1])
    fig2.suptitle('Images')
    plt.show()


def sigmoid(x):
    return 1.0/(1+np.exp(-x))
    # return np.exp(x) / (1 + np.exp(x))

def dsigmoid(x):
    return x*(1-x)

class Net(object):
    def __init__(self,layers):
        #权重的初始化,范围-1到1：+1的一列是偏置值
        self.W1 = np.random.random((layers[0] + 1, layers[1]))*2 - 1
        self.W2= np.random.random((layers[1] + 1, layers[2])) * 2 - 1
        self.W3 = np.random.random((layers[2] + 1, layers[3])) * 2 - 1

    def Linear_transform(self,input,W):
        #添加偏置值：最后一列全是1
        temp = np.ones([input.shape[0],input.shape[1]+1])
        temp[:,0:-1] = input
        input = temp
        # print(input.shape,W.shape)
        return sigmoid(np.dot(input, W)),input

    def train(self,data,targets,lr=0.01,batch_size=1,epochs=10000):
        for n in range(epochs+1):
            Loss=[]
            Acc=[]
            sample=list(range(len(data)))
            np.random.shuffle(sample)
            for  i in range(0,len(sample),batch_size):
                X=data[i:i+batch_size]
                Y=targets[i:i+batch_size]
                # L1：输入层传递给隐藏层的值；输入层64个节点，隐藏层100个节点
                # L2：隐藏层传递到输出层的值；输出层10个节点
                L1,input1= self.Linear_transform(X,self.W1)
                L2,input2= self.Linear_transform(L1,self.W2)
                L3, input3 = self.Linear_transform(L2, self.W3)

                #计算损失
                Y_pred=self.softmax(L3)
                # print(y_pred)
                loss=np.mean(self.cross_entropy_func(Y_pred,Y))
                Loss.append(loss)
                predictions = np.argmax(Y_pred, 1)
                # np.equal()：相同返回true，不同返回false
                accuracy = np.mean(np.equal(predictions, np.argmax(Y,axis=1)))
                Acc.append(accuracy)

                Z3_delta= (L3-Y) * dsigmoid(L3)
                # print(Z3_delta.shape)
                Z2_delta_b = Z3_delta.dot(self.W3.T) * dsigmoid(input3)
                # print(Z2_delta_b.shape)
                #由于我们在每一层的输入之前进行了维度扩充，添加了一个1来计算b，这个b对于上一层的反向传播时没有意义的
                #需要去掉，才能维持维度统一
                Z2_delta=Z2_delta_b[:,:Z2_delta_b.shape[1]-1]
                Z1_delta_b = Z2_delta.dot(self.W2.T) * dsigmoid(input2)
                Z1_delta = Z1_delta_b[:, :Z1_delta_b.shape[1] - 1]

                # 计算改变后的新权重
                self.W3 += lr *input3.T.dot(Z3_delta)/batch_size
                self.W2 += lr * input2.T.dot(Z2_delta)/batch_size
                self.W1 += lr * input1.T.dot(Z1_delta)/batch_size


            #每个epoach输出一次准确率
            #获取预测结果：返回与十个标签值逼近的距离，数值最大的选为本次的预测值
            Out = self.predict(X_val)
            #将最大的数值所对应的标签返回
            #计算损失
            Y_pred = self.softmax(Out)
            predictions=np.argmax(Y_pred,1)
            #np.equal()：相同返回true，不同返回false
            accuracy = accuracy_score(y_val, predictions)
            loss = np.mean(self.cross_entropy_func(Y_pred, labels_bin_val))
            print('epoach：',n,'train_acc：%.4f'%np.mean(Acc) ,'train_loss:%.4f:'%np.mean(Loss),
                  'test_acc：%.4f'%accuracy ,'test_loss:%.4f:'%np.mean(loss))

    def softmax(self,x):
        """ softmax function """
        # assert(len(x.shape) > 1, "dimension must be larger than 1")
        # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行
        out = x-np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
        out = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)
        return out

    def cross_entropy_func(self,y_pred,y_gt):
        return np.sum(y_pred*y_gt,axis=1)


    def predict(self,x):
        # 转为二维数据：由一维一行转为二维一行
        x = np.atleast_2d(x)

        # L1：输入层传递给隐藏层的值；输入层64个节点，隐藏层100个节点
        # L2：隐藏层传递到输出层的值；输出层10个节点
        L1, input1 = self.Linear_transform(x, self.W1)
        L2, input2 = self.Linear_transform(L1, self.W2)
        L3, input3 = self.Linear_transform(L2, self.W3)
        return L3

if __name__ =='__main__':
    train_dataset=Mnist(root=os.path.join('DataSet','MNIST','raw'))
    X = train_dataset.data
    Y = train_dataset.targets
    #进行简单的数据统计
    data_analyze(X,Y)
    #图像展开为全连接,reshape公用内存
    X_flatten=X.reshape(-1,28*28)
    #标准化
    scaler=preprocessing.StandardScaler().fit(X_flatten)
    X_scaled = scaler.transform(X_flatten)
    nn = Net([28*28,392,100,10])
    #sklearn切分数据
    X_train,X_val,y_train,y_val = train_test_split(X_scaled,Y)
    #标签二值化：将原始标签(十进制)转为新标签(二进制)
    labels_bin_train = LabelBinarizer().fit_transform(y_train)
    labels_bin_val = LabelBinarizer().fit_transform(y_val)

    # print('train......................')
    # nn.train(X_train,labels_bin_train,batch_size=BatchSize,epochs=20)
    # print('test.......................')
    test_dataset=Mnist(root=os.path.join('DataSet','MNIST','raw'),train=False)
    X = test_dataset.data
    Y = test_dataset.targets
    data_analyze(X,Y)
    #图像展开为全连接,reshape公用内存
    X_flatten=X.reshape(-1,28*28)
    #标准化
    scaler=preprocessing.StandardScaler().fit(X_flatten)
    X_scaled = scaler.transform(X_flatten)
    out=nn.predict(X_scaled)
    #求出最大预测值的下标
    out = nn.softmax(out)
    y_pred=np.argmax(out, axis=1)
    # print(y_pred.shape)
    # print(y_pred[:100])
    # print(Y[:100])
    print('measure.......................')
    f1=f1_score(Y, y_pred, average='weighted')
    accuracy = accuracy_score(Y, y_pred)
    confu_mat=confusion_matrix(Y, y_pred)
    print('F1-Measure:%0.4f'%f1,'acc:%.4f'%accuracy)

