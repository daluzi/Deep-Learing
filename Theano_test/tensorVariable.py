# _*_ coding: utf-8 _*_
# @Author   : daluzi
# @time     : 2019/10/27 18:50
# @File     : tensorVariable.py
# @Software : PyCharm

import theano
from theano import tensor as T

#初始化张量
x = T.scalar(name = 'input', dtype = 'float32')
w = T.scalar(name='weight', dtype='float32')
b = T.scalar(name='bias', dtype='float32')
z = w * x + b
#编译程序
net_input = theano.function(inputs=[w,x,b], outputs=z)
#执行程序
print('net_input: %2f'% net_input(2.0, 3.0, 0.5))
