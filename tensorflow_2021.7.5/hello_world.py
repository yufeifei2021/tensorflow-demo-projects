#  -*-coding:utf8 -*-

"""
Created on 2021 7 5

@author: 陈雨
"""


import tensorflow as tf

# 关于tf新旧版本一些模块的变更问题
tf.compat.v1.disable_eager_execution()

hello = tf.constant('hello,tensorf')

sess = tf.compat.v1.Session()

print(sess.run(hello))
