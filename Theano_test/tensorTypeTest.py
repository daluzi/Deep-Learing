# _*_ coding: utf-8 _*_
# @Author   : daluzi
# @time     : 2019/10/28 15:45
# @File     : tensorTypeTest.py
# @Software : PyCharm

import theano
import numpy as np
import theano.tensor as T

r = T.row()
print(r.broadcastable)#True,False

mtr = T.matrix()
print(mtr.broadcastable)#False,False

f_row = theano.function([r, mtr], [r + mtr])
R = np.arange(1, 3).reshape(1,2)
print(R)
M = np.arange(1,7).reshape(3,2)
print(M)

print(f_row(R , M))