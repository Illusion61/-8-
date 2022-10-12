### cora数据集内容
- x:存储训练集中有标签节点特征的向量
- allx:存储训练集所有节点特征的向量
- tx:存储测试集节点特征的向量
- y:存储训练集节点分类
- ally:存储训练集节点分类
- ty:存储测试集节点分类
- graph:用邻接表的形式存储节点之间边的信息

### 问题回答
> 1.将数据中每个相对独立的个体抽象为节点(具有自己的特征).节点之间的关系抽象为边,边可以带有权值来描述节点之间关联的强度或者其他信息,边可以是有向边或者无向边(两条有向边)<br>
> 2.用邻接表或者邻接矩阵存储图,在节点比较多边稀疏的时候用邻接表,节点少边多的时候用邻接矩阵<br>
> 3.图神经网络的输入是每一个节点的特征和图的结构(节点之间的关系).<br>图神经网络的输出可以是图的分类(比如化学分子是否有可能存在,病毒是否会突变),可以是另外一张图,也可以是输出每个节点的分类<br>
> 4.图神经网络的作用:处理图结构数据的信息(包含节点之间的关系)并输出结果

### 论文问题回答
> 1.过平滑有点类似梯度消失,表现为图中重要的区分信息的消失以及信息的趋同.由于一些节点拥有相同的邻居节点,所以它们的输入比较相似,得出的结果也会趋于类似,而上一层的结果会作为下一层的输入,导致得出的结果越来越相似,GCN的深度越深过平滑现象越明显<br>
> 2.DropEdge的工作机制有点类似Dropout,不过DropEdge是随机按照概率p在一次传播的时候忽略一些边.<br>DropEdge可以帮助减少相邻节点的影响,防止深度过深的时候过平滑<br>同时DropEdge类似于亚种在生物进化中的角色:物种为了生存往往会倾向于适应现在的特有环境(神经网络为了提高准确度往往会倾向于适应现有的部分数据),而加入dropedge之后就好像环境时刻在变化(每次随机从数据中去掉一些边),会增加GCN对更广泛范围的数据的适应能力,减轻过拟合<br>
> 3.随着层数的增加,大多数数据集Original和DropEdge在验证集上的精度都是逐渐减少,但是dropedge的精度大部分情况下比Original的要高

### 实验数据记录表格
|GCN层数|2|4|8|16|32|64|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Original|0.804|0.739|0.478|0.165|0.194|0.186|
|DropEdge|0.808|0.701|0.147|0.139|0.280|0.219|

Original2:0.8040000200271606
Original4:0.7390000224113464
Original8:0.4780000150203705
Original16:0.16500000655651093
Original32:0.1940000057220459
Original64:0.1860000044107437
DropOut2:0.8080000281333923
DropOut4:0.7010000348091125
DropOut8:0.147
DropOut16:0.13900001347064972
DropOut32:0.2800000011920929
DropOut64:0.21900001168251038
