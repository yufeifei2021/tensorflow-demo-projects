#  -*-coding:utf8 -*-

"""
Created on 2021 7 7

@author: 陈雨
"""

import tensorflow as tf

#查看tensorflow版本
print(tf.__version__)

print('GPU', tf.test.is_gpu_available())

a = tf.constant(2.0)
b = tf.constant(4.0)
print(a + b)


#TensorFlow1.x-GPU代码测试
# import tensorflow as tf
# tf.test.is_gpu_available()
