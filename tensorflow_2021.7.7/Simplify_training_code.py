 #  -*-coding:utf8 -*-

"""
Created on 2021 7 7

@author: 陈雨
"""

# tensorflow2.0——keras相关接口简化训练代码

import tensorflow as tf

def preporocess(x,y):
    x = tf.cast(x,dtype=tf.float32) / 255
    # 铺平
    x = tf.reshape(x,(-1,28 *28))                   
    x = tf.squeeze(x,axis=0)
    print('里面x.shape:',x.shape)
    y = tf.cast(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y

def main():
    #   加载手写数字数据
    mnist = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    #   处理数据
    #   训练数据
    # 将x,y分成一一对应的元组
    db = tf.data.Dataset.from_tensor_slices((train_x, train_y)) 

    db = db.map(preporocess)  # 执行预处理函数
    db = db.shuffle(60000).batch(2000)  # 打乱加分组

    #   测试数据
    db_test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    db_test = db_test.map(preporocess)
    db_test = db_test.shuffle(10000).batch(10000)

    #   设置超参
    iter_num = 2000  # 迭代次数
    lr = 0.01  # 学习率

    #   定义模型器和优化器
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    
    #   优化器
    # optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)              #   定义优化器
    model.compile(optimizer= optimizer,loss=tf.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])       #   定义模型配置
    model.fit(db,epochs=10,validation_data=db,validation_steps=2)          #  运行模型，参数validation_data是指在哪个测试集上进行测试
    model.evaluate(db_test)                                                     #   最后打印测试数据相关准确率数据

if __name__ == '__main__':
    main()