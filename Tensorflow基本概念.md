# Tensorflow基本概念





* 使用图（graphs）来表示计算任务。

* 在被称之为会话（Session）的上下文（context）中执行图。

* 使用tensor表示数据。

* 通过变量（Variable）维护状态。

* 使用feed和fetch可以为任意的操作赋值或者从其中获取数据。

  

  Tensorflow是一个编程系统，使用图（graphs）来表示计算任务，图（graphs）中的节点称之为op（operation），一个op获得0个或多个Tensor，执行计算，产生0个或多个Tensor。Tensor看作是一个n维的数组或列表。图必须在会话（Session）里被启动。



特点：

	* 真正的可移植性
	* 多语言支持
	* 高度的灵活性与效率







Fetch：即可同时运行多个op:  `sess.run([mul,add])`

Feed：feed的数据以字典的形式传入：

```python
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2=[2.]}))
```

