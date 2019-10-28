# _*_ coding: utf-8 _*_
# @Author   : daluzi
# @time     : 2019/10/28 16:02
# @File     : theanoFunction.py
# @Software : PyCharm

import theano
x, y = theano.tensor.fscalars('x', 'y')
z1 = x + y
z2 = x * y

f = theano.function([x, y ], [z1, z2])
print(f(2,3))

