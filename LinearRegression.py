import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()
iris=datasets.load_iris()

width=np.array([x[3] for x in iris.data])
length=np.array([x[0] for x in iris.data])
x0=tf.placeholder(shape=[None,1],dtype=tf.float32)#x0和y0都是一维列向量，此处不能定义为1*1矩阵
y0=tf.placeholder(shape=[None,1],dtype=tf.float32)
A=tf.Variable(tf.random_normal(shape=[1,1]))#正态随机数初始化A和b
b=tf.Variable(tf.random_normal(shape=[1,1]))

y=tf.add(tf.matmul(x0,A),b)#模型是y=Ax+b
loss=tf.reduce_mean(tf.square(y-y0))#误差用残差平方和计量

init=tf.global_variables_initializer()
sess.run(init)


rate=0.05
GDO=tf.train.GradientDescentOptimizer(rate)
TrainStep=GDO.minimize(loss)

losslist=[]
BatchSize=25
for i in range(100):
	num=np.random.choice(len(width),size=BatchSize)#在0~len(width)之间随机采样BatchSize个数据
x0_rand=np.transpose([width[num]])#标量经过转置变为一维列向量
y0_rand=np.transpose([length[num]])
sess.run(TrainStep,feed_dict={x0:x0_rand,y0:y0_rand})#投喂训练
TempLoss=sess.run(loss,feed_dict={x0:x0_rand,y0:y0_rand})
losslist.append(TempLoss)
if((i+1)%25==0):
	print('Step #'+str(i+1)+'   '+str(TempLoss))
[slope]=sess.run(A)
[intercept]=sess.run(b)
bestfit=[]
for i in width:
	bestfit.append(slope*i+intercept)
plt.plot(width,length,'o',label="OriginalData")
plt.plot(width,bestfit,'r-',label="BestLine")
plt.show()
plt.plot(losslist,'k-')
plt.show()