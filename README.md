# neural-network
BP,CNN starter
minst.py是载入训练与测试数据程序，主程序只调用其载入数据

四个压缩包是MINST数据库，手写数字识别。训练集60000个，测试集10000个

FCnetwork_version1.0：全连接神经网络，用的误差反向传播方法以及批处理增加计算速度结构与version 2.0类似。

FCnetwork_version2.0：比上个版本改用了network层的模块化编程，可以像组装积木一样组装神经网络。

结构是三层输入层784个节点对应784个像素点，隐含层60个节点，采用Relu激活函数（version1.0用Sigmoid），
仿真显示在这种情况下Relu训练出来的效果优于Sigmoid。输出层10个节点对应十个数字，激活函数为softmax损失函数为交叉熵函数。
