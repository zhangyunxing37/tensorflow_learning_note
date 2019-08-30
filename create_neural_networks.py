# coding: utf-8
import tensorflow as tf 

import numpy as np
# """
# 利用TensorFlow来定义自己的神经网络框架主要步骤如下：
# 步骤1：准备好自己的训练数据---数据决定了整个框架的构建
# 步骤2：定义好添加神经层的函数--定义前向传播的的架构：输入，权重矩阵，隐藏层，预测层
# 步骤3：定义损失函数--选择 optimizer 使 loss 达到最小--反向传播
# 步骤4：生成会话，训练steps轮
# """

# 1.准备数据
x_data = np.random.normal(0, 1,(3000, 2))
noise = np.random.normal(0, 0.5, (3000,1))
y_data = np.array(list(map(lambda x : x[0] ** 2+x[1], x_data.tolist())))[:,np.newaxis] - 0.5 + noise


# 2.设置输入输出的节点数
# None表示全数据输入---对应的是batch_size
xs = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, 1])


# 3.设计权重矩阵的网络结构
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases=tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 输入与权重矩阵相乘
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

lay_1 = add_layer(xs, 2, 15, tf.nn.relu)
prediction = add_layer(lay_1, 15, 1, activation_function=None)



# 4.定义loss损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))



# 5.优化器--梯度下降算法选择
# 0.01为学习率
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)



# 6.对所有变量进行初始化，因为前面的所有框架都搭好了，正式将结构导入运算流里
init = tf.global_variables_initializer()
sess=tf.Session()
# 以上都没有进行运算，只有运算流框架启动后就开始运算
sess.run(init)


# 迭代1000次训练---自己写循环
for i in range(5000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
