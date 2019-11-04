# _*_ coding: utf-8 _*_
# @Author   : daluzi
# @time     : 2019/11/4 20:11
# @File     : graphs.py
# @Software : PyCharm

import tensorflow as tf

# 创建一个常量op
m1 = tf.constant([[3,3]])
# 创建一个常量op
m2 = tf.constant([[2],[3]])
# 创建一个矩阵乘法op，把m1和m2传入
product = tf.matmul(m1, m2)
print(product)

# 定义一个会话，启动默认图
sess = tf.Session(tf.Graph())
# 调用sess的run方法来执行矩阵乘法op
# run(product)触发了图中的3个op
result = sess.run(product)
print(result)
sess.close()

# with tf.compat.v1.Session() as sess:
# 	result = sess.run(product)
# 	print(result)
