# _*_ coding: utf-8 _*_
# @Author   : daluzi
# @time     : 2019/10/28 16:08
# @File     : grad.py
# @Software : PyCharm

import theano
x = theano.tensor.fscalar('x')
y = 1 / (1 + theano.tensor.exp(-x))
dx = theano.grad(y , x)
f = theano.function([x],dx)
print(f(3))