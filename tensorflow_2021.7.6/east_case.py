# coding = utf-8

"""
Created on 2021 7 6

@author: 陈雨
"""

# 我们将要使用TensorFlow搭建一个神经网络
# 计算出正确的W和b,Weightsx_data+biases(x_data0.1+0.3)

import tensorflow as  tf
import numpy as np

#创建数据
#np.random.rand生成特定形状下[0,1)下的均匀分布随机数
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

tf.compat.v1.disable_eager_execution()


'''创建tensorflow的结构开始'''


#生成一维的-1到1的数
Weights = tf.Variable(tf.random.uniform([1],-1.0,1.0))

#设置偏向
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data+biases

#误差
loss = tf.reduce_mean(tf.square(y-y_data))

#0.5是学习效率GradientDescentOptimizer优化器减少误差(梯度下滑)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)

#用该优化器减少误差
train = optimizer.minimize(loss)

#将其结构赋予生命的意思
init = tf.compat.v1.initialize_all_variables()

#激活以上的代码
sess = tf.compat.v1.Session()

#训练上面的神经网络
sess.run(init)
for step in range(10000):
	sess.run(train)
	print(step,sess.run(Weights),sess.run(biases))
