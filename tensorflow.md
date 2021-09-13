# TensorFlow

TensorFlow 是一个开源的、基于 Python 的机器学习框架，它由 Google 开发，并在图形分类、音频处理、推荐系统和**自然语言处理**等场景下有着丰富的应用，是目前最热门的机器学习框架。

除了 Python，TensorFlow 也提供了 C/C++、Java、Go、R 等其它编程语言的接口。

这套 TensorFlow 教程对一些常见的深度学习网络进行了介绍，并给出了完整的实现代码，不仅适合初学者入门，也适合程序员进阶。

## 1.TensorFlow是什么

任何曾经试图在 [Python](http://c.biancheng.net/python/) 中只利用 NumPy 编写神经网络代码的人都知道那是多么麻烦。编写一个简单的一层前馈网络的代码尚且需要 40 多行代码，当增加层数时，编写代码将会更加困难，执行时间也会更长。

[TensorFlow](http://c.biancheng.net/tensorflow/) 使这一切变得更加简单快捷，从而缩短了想法到部署之间的实现时间。在本教程中，你将学习如何利用 TensorFlow 的功能来实现深度神经网络。

TensorFlow 是由 Google Brain 团队为深度神经网络（DNN）开发的功能强大的开源软件库，于 2015 年 11 月首次发布，在 Apache 2.x 协议许可下可用。截至今天，短短的两年内，其 [GitHub 库](https://github.com/tensorflow/tensorflow)大约 845 个贡献者共提交超过 17000 次，这本身就是衡量 TensorFlow 流行度和性能的一个指标。

![TensorFlow的领先地位示意图](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G1160HH.gif)
图 1 TensorFlow的领先地位示意图

开源深度学习库 TensorFlow 允许将深度神经网络的计算部署到任意数量的 CPU 或 GPU 的服务器、PC 或移动设备上，且只利用一个 TensorFlow API。你可能会问，还有很多其他的深度学习库，如 Torch、Theano、Caffe 和 MxNet，那 TensorFlow 与其他深度学习库的区别在哪里呢？**包括 TensorFlow 在内的大多数深度学习库能够自动求导、开源、支持多种 CPU/GPU、拥有预训练模型，并支持常用的NN架构，如递归神经网络（RNN）、卷积神经网络（CNN）和深度置信网络（DBN）。**

TensorFlow 则还有更多的特点，如下：

- 支持所有流行语言，如 Python、[C++](http://c.biancheng.net/cplus/)、[Java](http://c.biancheng.net/java/)、R和Go。
- 可以在多种平台上工作，甚至是移动平台和分布式平台。
- 它受到所有云服务（AWS、Google和Azure）的支持。
- Keras——高级神经网络 API，已经与 TensorFlow 整合。
- 与 Torch/Theano 比较，TensorFlow 拥有更好的计算图表可视化。
- 允许模型部署到工业生产中，并且容易使用。
- 有非常好的社区支持。
- TensorFlow 不仅仅是一个软件库，它是一套包括 TensorFlow，TensorBoard 和 TensorServing 的软件。

> [谷歌 research 博客](https://research.googleblog.com/2016/11/celebrating-tensorflows-first-year-html)列出了全球一些使用 TensorFlow 开发的有趣项目：
>
> - **Google 翻译**运用了 TensorFlow 和 TPU（Tensor Processing Units）。
> - **Project Magenta** 能够使用强化学习模型生成音乐，运用了 TensorFlow。
> - 澳大利亚海洋生物学家使用了 TensorFlow 来**发现和理解濒临灭绝的海牛**。
> - 一位日本农民运用 TensorFlow 开发了一个应用程序，**使用大小和形状等物理特性对黄瓜进行分类**。

## 2.TensorFlow安装和下载（超详细）

本节将介绍在不同的操作系统（Linux、Mac和Windows）上如何全新安装 [TensorFlow](http://c.biancheng.net/tensorflow/) 1.3。

首先了解安装 TensorFlow 的必要要求，TensorFlow 可以在 Ubuntu 和 macOS 上基于 native pip、Anaconda、virtualenv 和 [Docker](http://c.biancheng.net/docker/) 进行安装，对于 Windows 操作系统，可以使用 native pip 或 Anaconda。

**Anaconda** 适用于这三种操作系统，安装简单，在同一个系统上维护不同的项目环境也很方便，因此本教程将基于 Anaconda 安装 TensorFlow。

> 有关 Anaconda 及其环境管理的更多详细信息，请参考https://conda.io/docs/user-guide/index.html。

本教程中的代码已经在以下平台上进行了测试：

- Windows 10，Anaconda 3，[Python](http://c.biancheng.net/python/) 3.5，TensorFlow GPU，CUDA toolkit 8.0，cuDNN v5.1，NVDIA GTX 1070
- Windows 10/Ubuntu 14.04/Ubuntu 16.04/macOS Sierra，Anaconda 3，Python 3.5，TensorFlow（CPU）

### 2.1TensorFlow安装准备工作

TensorFlow 安装的前提是系统安装了 Python 2.5 或更高版本，教程中的例子是以 Python 3.5（Anaconda 3 版）为基础设计的。为了安装 TensorFlow，首先确保你已经安装了 Anaconda。可以从网址（https://www.continuum.io/downloads）中下载并安装适用于 Windows/macOS 或 Linux 的 Anaconda。

安装完成后，可以在窗口中使用以下命令进行安装验证：

```python
conda --version
```

> 安装了 Anaconda，之后决定安装 TensorFlow CPU 版本或 GPU 版本。几乎所有计算机都支持 TensorFlow CPU 版本，而 GPU 版本则要求计算机有一个 CUDA compute capability 3.0 及以上的 NVDIA GPU 显卡（对于台式机而言最低配置为 NVDIA GTX 650）。 
>
> CPU 与 GPU 的对比：**中央处理器（CPU）**由对顺序**串行**处理优化的内核（4～8个）组成。**图形处理器（GPU）**具有大规模**并行**架构，由数千个更小且更有效的核芯（大致以千计）组成，能够同时处理多个任务。
>
> 对于 TensorFlow GPU 版本，需要先安装 CUDA toolkit 7.0 及以上版本、NVDIA【R】驱动程序和 cuDNN v3 或以上版本。Windows 系统还另外需要一些 DLL 文件，读者可以下载所需的 DLL 文件或安装 Visual Studio [C++](http://c.biancheng.net/cplus/)。
>
> 还有一件事要记住，cuDNN 文件需安装在不同的目录中，并需要确保目录在系统路径中。当然也可以将 CUDA 库中的相关文件复制到相应的文件夹中。

### 2.2TensorFlow安装具体做法

1. 在命令行中使用以下命令创建 conda 环境（如果**使用 Windows，最好在命令行中以管理员身份执行**）：

   ```python
   conda create -n tensorflow python=3.5
   ```

2. 激活 conda 环境：


   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G13539632.gif)

3. 该命令应提示：


   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G13610318.gif)

4. 根据要在 conda 环境中安装的 TensorFlow 版本，输入以下命令：


   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G13630V1.gif)

5. 在命令行中输入 python，并输入以下代码：（安装验证的代码）

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G13AYK.gif)
   
6. 输出如下：

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G13G4H1.gif)

7. 在命令行中禁用 conda 环境（Windows 调用 deactivate 命令，MAC/Ubuntu 调用 source deactivate 命令）。

**TensorFlow安装过程解读分析**

Google 使用 wheel 标准分发 TensorFlow，它是 .whl 后缀的 ZIP 格式文件。Python 3.6 是 Anaconda 3 默认的 Python 版本，且没有已安装的 wheel。在编写本教程时，Python 3.6 支持的 wheel 仅针对 Linux/Ubuntu，因此，在创建 TensorFlow 环境时，这里指定 Python 3.5。接着新建 conda 环境，命名为 tensorflow，并安装 pip，python，wheel 及其他软件包。

> whl格式本质上是一个**压缩包**，里面包含了py文件，以及经过编译的pyd文件。 使得可以在不具备编译环境的情况下，选择合适自己的python环境进行安装。
>
> pyd 是**python插件库中的一种文件**，而且可以保护python文件的源码不被暴露。

conda 环境创建后，**调用 source activate/activate 命令激活环境**。在激活的环境中，使用 pip install 命令安装所需的 TensorFlow（从相应的 TensorFlow-API URL下载）。尽管有利用 conda forge 安装 TensorFlow CPU 的 Anaconda 命令，但 **TensorFlow 推荐使用 pip install**。在 conda 环境中安装 TensorFlow 后，就可以禁用了。现在可以执行第一个 TensorFlow 程序了。

程序运行时，可能会看到一些警告（W）消息和提示（I）消息，最后是输出代码：

```python
Welcome to the exciting world of Deep Neural Networks!
```

恭喜你已成功安装并执行了第一个 TensorFlow 代码，在下一节中更深入地通读代码。

**拓展阅读**

另外，你也可以安装 **Jupyter notebook**：

1. 安装 python：

   ```python
   conda install -c anaconda ipython
   ```

2. 安装 nb_conda_kernels：

   ```python
   conda install -channel=conda-forge nb_conda_kernels
   ```

3. 启动 Jupyter notebook：

   ```python
   jupyter notebook
   ```

TIP：这将会打开一个新的浏览器窗口。

如果已安装了 TensorFlow，则可以调用 `pip install--upgrade tensorflow` 进行升级。

另外，你可以通过以下网址找到关于 TensorFlow 安装的更多资料：

- http://www.tensorflow.org/install/
- [https:www.tensorflow.org/install/install_sources](http://https:www.tensorflow.org/install/install_sources)
- http://llvm.org/
- https://bazel.build/

## 3.第一个TensorFlow程序（hello world）详解

在任何计算机语言中学习的第一个程序是都是 Hello world，本教程中也将遵守这个惯例，从程序 Hello world 开始。

下面一起看一下这段简单的代码：

1. 导入tensorflow，这将导入 TensorFlow 库，并允许使用其精彩的功能：

   ```python
   import tensorflow as if//导入
   ```

2. 由于要打印的信息是一个常量字符串，因此使用 tf.constant：

   ```python
   message = tf.constant('Welcome to the exciting world of Deep Neural Networks!')
   // 常量
   tf.constant(value, dtype=None, shape=None, name='Const',verify_shape=False)
   value:是一个必须的值，可以是一个数值，也可以是一个列表；可以是一维的，也可以是多维的。
   ```

3. 为了**执行计算图**，利用 with 语句定义 Session，并使用 run 来运行：

   ```python
   with tf.Session() as sess:
     print(sess.run(message).decode())
   ```

   > 使用with tf.Session()  创建上下文（Context）来执行，当上下文退出时自动释放。
   >
   > decode 解码

4. 输出中包含一系列警告消息（W），具体取决于所使用的计算机和操作系统，并声明如果针对所使用的计算机进行编译，代码运行速度可能会更快：


   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G202103Y.gif)

5. 如果使用 TensorFlow GPU 版本，则还会获得一系列介绍设备的提示消息（I）：

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G2022Y58.gif)
   
6. 最后是在会话中打印的信息：

   ```python
   Welcome to the exciting world of Deep Neural Networks!
   ```

### 3.1TensorFlow 程序解读分析

前面的代码分为以下三个主要部分：

- **第一部分 import 模块包含代码将使用的所有库**，在目前的代码中只使用 TensorFlow，其中语句 import tensorflow as tf 则允许 [Python](http://c.biancheng.net/python/) 访问 TensorFlow 所有的类、方法和符号。
- **第二个模块包含图形定义部分...创建想要的计算图**。在本例中计算图只有一个节点，tensor 常量消息由字符串“Welcome to the exciting world of Deep Neural Networks”构成。
- **第三个模块是通过会话执行计算图**，这部分使用 with 关键字创建了会话，最后在会话中执行以上计算图。

> 计算图和反向传播都是**深度学习训练神经网络的重要核心概念**。
>
> **计算图**被定义为有向图，其中节点对应于数学运算。 计算图是表达和评估数学表达式的一种方式。
>
> 例如，这里有一个简单的数学公式 
>
> ```shell
> p = x + y
> ```
>
> 我们可以绘制上述方程的计算图如下。
> ![img](http://www.yiibai.com/uploads/images/201806/1406/738090622_96682.png)
>
> 上面的计算图具有一个加法节点(具有“+”符号的节点)，其具有两个输入变量`x`和`y`以及一个输出`q`。
>
> 让我们再举一个例子，稍微复杂些。如下等式。
>
> ```shell
> g = ( x + y ) ∗ z
> ```
>
> 以上等式由以下计算图表示。
>
> ![img](http://www.yiibai.com/uploads/images/201806/1406/829090623_54957.png)

现在来解读输出。收到的警告消息提醒 TensorFlow 代码可以以更快的速度运行，这能够通过从 source 安装 TensorFlow 来实现。收到的提示消息给出计算设备的信息。这两个消息都是无害的，如果不想看到它们，可以通过以下两行代码实现：

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
```

以上代码用于忽略级别 <u>2 及以下</u>的消息（级别 1 是提示，级别 2 是警告，级别 3 是错误）。

该程序打印计算图执行的结果，**计算图的执行则使用 sess.run() 语句，sess.run 求取 message 中所定义的 tensor 值；计算图执行结果输入到 print 函数，并使用 decode 方法改进，print 函数向 stdout 输出结果**：

```python
b'Welcome to the exciting world of Deep Neural Networks!'
```

这里的输出结果是一个字节字符串。要删除字符串引号和“b”（表示字节，byte）只保留单引号内的内容，可以使用 decode() 方法。

> decode 解码

## 4.TensorFlow程序结构（深度剖析）

[TensorFlow](http://c.biancheng.net/tensorflow/) 与其他编程语言**非常不同**。

首先通过将程序分为两个独立的部分，构建任何拟创建神经网络的蓝图，包括计算图的定义及其执行。起初对于传统程序员来说很麻烦，但是**图定义和执行的分开设计**让 TensorFlow 能够多平台工作以及并行执行，TensorFlow 也因此更加强大。

> 计算图：是包含节点和边的网络。本节定义所有要使用的数据，也就是张量（tensor）对象（常量、变量和占位符），同时定义要执行的所有计算，即运算操作对象（Operation Object，简称 OP）。

**每个节点**可以有零个或多个输入，但**只有一个输出**。网络中的节点表示对象（张量和运算操作），边表示运算操作之间流动的张量。计算图定义神经网络的蓝图，但其中的张量还没有相关的数值。

> 为了构建计算图，需要定义所有要执行的常量、变量和运算操作。常量、变量和占位符将在下一节中介绍，数学运算操作将在矩阵运算章节中详细讨论。

简单的例子描述程序结构——通过定义并执行计算图来实现两个向量相加。

计算图的执行：使用会话对象来实现计算图的执行。会话对象封装了评估张量和操作对象的环境。这里真正实现了运算操作并将信息从网络的一层传递到另外一层。不同张量对象的值仅在会话对象中被初始化、访问和保存。在此之前张量对象只被抽象定义，在会话中才被赋予实际的意义。

### 4.1具体做法

通过以下步骤定义一个计算图：

1. 在此以两个向量相加为例给出计算图。假设有两个向量 v_1 和 v_2 将作为输入提供给 Add 操作。建立的计算图如下：

   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G32219396.gif)

2. 定义该图的相应代码如下所示：

    

   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G3223B02.gif)
   
3. 然后在会话中执行这个图：

    

   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G32255O9.gif)
   
4. 以上两行相当于下面的代码。**上面的代码的优点是不必显式写出关闭会话的命令**：

    

   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G323355Q.gif)
   
5. 运行结果是显示两个向量的和：

   ```python
   {3 3 8 7}
   ```

请记住，**每个会话都需要使用 close() 来明确关闭，而 with 格式可以在运行结束时隐式关闭会话**。

### 4.2解读分析

计算图的构建非常简单。添加变量和操作，并按照逐层建立神经网络的顺序传递它们（让张量流动）。TensorFlow 还允许**使用 with tf.device() 命令来使用具有不同计算图形对象的特定设备（CPU/GPU）**。在例子中，计算图由三个节点组成， v_1 和 v_2 表示这两个向量，Add 是要对它们执行的操作。

接下来，为了使这个图生效，首先需要使用 tf.Session() 定义一个会话对象 sess。然后使用 Session 类中定义的 run 方法运行它，如下所示：

```python
run(fetches,feed_dict=None,options=None,run_metadata)
	//获取
    // feed_dict就是用来赋值的，格式为字典型
    //选择器
    //运行元数据
```

运算结果的值在 fetches 中提取；在示例中，提取的张量为 v_add。run 方法将导致在每次执行该计算图的时候，都将对与 v_add 相关的张量和操作进行赋值。如果抽取的不是 v_add 而是 v_1，那么最后给出的是向量 v_1 的运行结果：

```python
{1,2,3,4}
```

此外，一次可以提取一个或多个张量或操作对象，例如，如果结果抽取的是 [v_1...v_add]，那么输出如下：

```python
{array([1,2,3,4]),array([2,1,5,3]),array([3,3,8,7])}
```

在同一段代码中，可以有多个会话对象。

### 4.3拓展阅读

你一定会问为什么必须编写这么多行的代码来完成一个简单的向量加，或者显示一条简单的消息。其实你可以利用下面这一行代码非常方便地完成这个工作：

```python
print(tf.Session().run(tf.add(tf.constant([1,2,3,4]),tf.constant([2,1,5,3]))))
							//tf.constant表示创建常量
```

编写这种类型的代码不仅影响计算图的表达，而且当在 for 循环中重复执行相同的操作（OP）时，可能**会导致占用大量内存**。养成显式定义所有张量和操作对象的习惯，不仅可使代码更具可读性，还可以帮助你以更清晰的方式可视化计算图。

> 注意，使用 TensorBoard 可视化图形是 TensorFlow 最有用的功能之一，特别是在构建复杂的神经网络时。我们构建的计算图可以在图形对象的帮助菜单下进行查看。

如果你正在使用 Jupyter Notebook 或者 [Python](http://c.biancheng.net/python/) shell 进行编程，使用 tf.InteractiveSession 将比 tf.Session 更方便。InteractiveSession 使自己成为默认会话，因此你可以使用 eval() 直接调用运行张量对象而不用显式调用会话。下面给出一个例子：



![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G32615351.gif)

## 5.TensorFlow常量、变量和占位符详解

最基本的 [TensorFlow](http://c.biancheng.net/tensorflow/) 提供了一个库来定义和执行对张量的各种数学运算。张量，可理解为一个 n 维矩阵，所有类型的数据，包括标量、矢量和矩阵等都是特殊类型的张量。


![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G4044Q25.gif)


TensorFlow 支持以下三种类型的张量：

1. **常量**：常量是其值不能改变的张量。
2. **变量**：当一个量在会话中的值需要更新时，使用变量来表示。例如，在神经网络中，权重需要在训练期间更新，可以通过将权重声明为变量来实现。变量在使用前需要被显示初始化。另外需要注意的是，常量存储在计算图的定义中，每次加载图时都会加载相关变量。换句话说，它们是占用内存的。另一方面，变量又是分开存储的。它们可以存储在参数服务器上。
3. **占位符**：用于将值输入 TensorFlow 图中。它们可以和 feed_dict 一起使用来输入数据。在训练神经网络时，它们通常用于提供新的训练样本。**在会话中运行计算图时，可以为占位符赋值**。这样在构建一个计算图时不需要真正地输入数据。需要注意的是，**占位符不包含任何数据，因此不需要初始化它们**。

### 5.1TensorFlow 常量

声明一个标量常量：

```python
t_1 = tf.constant(4)
```

一个形如 [1，3] 的常量向量可以用如下代码声明：

```python
t_2 = tf.constant([4,3,2])
```

要创建一个所有元素为零的张量，可以使用 **tf.zeros() 函数**。这个语句可以创建一个形如 [M，N] 的零元素矩阵，数据类型（dtype）可以是 int32、float32 等：

```python
tf.zeros([M,N],tf.dtype)
```

例如：

```python
zero_t = tf.zeros([2,3],tf.int32)
\# Results in an 2x3 array of zeros:[[0 0 0],[0 0 0]]
```

还可以创建与现有 Numpy 数组或张量常量具有相同形状的张量常量，如下所示：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G40552202.gif)

创建一个所有元素都设为 1 的张量。下面的语句即创建一个形如 [M，N]、元素均为 1 的矩阵：

```python
tf.ones([M,N],tf,dtype)
```

例如：

```python
ones_t = tf.ones([2,3],tf.int32)
\# Results in an 2x3 array of ones:[[1 1 1],[1 1 1]]
```


更进一步，还有以下语句：

- 在一定范围内生成一个**从初值到终值等差排布的序列**：

  ```python
  tf.linspace(start,stop,num)
  ```

  > tf.linspace(start, end, num)：
  >
  > 这个函数主要的参数就这三个，start代表起始的值，end表示结束的值，num表示在这个区间里生成数字的个数，生成的数组是等间隔生成的。
  >
  > **start和end这两个数字必须是浮点数，不能是整数**，如果是整数会出错的，请注意！

  

  > np.linspace(start, end, num)：
  >
  > 主要的参数也是这三个，我们平时用的时候绝大多数时候就是用这三个参数。start代表起始的值，end表示结束的值，num表示在这个区间里生成数字的个数，生成的数组是等间隔生成的。
  >
  > **start和end这两个数字可以是整数或者浮点数**

  相应的值为 (stop-start)/(num-1)。例如：

  ```python
  range_t = tf.linspace(2.0,5.0,5)
  \#We get:[2. 2.75 3.5 4.25 5.]
  ```

  

- 从**开始（默认值=0）**生成一个数字序列，增量为 delta（默认值=1），**直到终值（但不包括终值）**：

  ```python
  tf.range(start,limit,delta)
  ```

  > tf.range函数：创建数字序列，该数字开始于开始并且将增量扩展到不包括极限的序列。

  下面给出实例：

  ```python
  range_t = tf.range(10)
  \#Result:[0 1 2 3 4 5 6 7 8 9]
  ```


TensorFlow 允许创建具有不同分布的随机张量：

> mean：均值
>
> stddev：标准差
>
> seed() 方法：
>
> ​			1.改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数
>
> ​			2.seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。



> **tf.random_normal()函数**用于从“服从指定**正态分布**的序列”中随机取出指定个数的值。**正态分布随机数组**
>
> tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
>
>
>     shape: 输出张量的形状，必选
>     mean: 正态分布的均值，默认为0
>     stddev: 正态分布的标准差，默认为1.0
>     dtype: 输出的类型，默认为tf.float32
>     seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
>     name: 操作的名称



> ##### **tf.truncated_normal(shape, mean, stddev)****截尾正态分布随机数组**
>
> **截断的产生正态分布**的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。
>
> - shape，生成张量的维度
> - mean，均值
> - stddev，标准差



> **tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)**
>
> **从均匀分布中输出随机值**		**伽马分布随机数组**
>
> 生成的值在该[minval,maxval) 范围内跟踪同步分布。下限minval 包含在范围内，而社区maxval 被排除在外。
>
> 对于浮点数,默认范围是 [0, 1).对于极限可能, maxval 必须明确地指定.
>
> 在紧急情况下，随机偶尔会有轻微的变化，最大有效值 - minval 是 2 的合理功率。maxval - 分钟值的值，更短，对于输出（2**32 或 2**64）的范围。
>
> - 形状：一维转动张量或巨蟒阵列。输出张量的形状。
> - minval：dtype 类型的 0-D 张量或 Python 值；生成的随机值范围的下限；默认为 0。
> - maxval：dtype 类型的 0-D 张量或 Python 值。要生成的随机值范围的区域。如果 dtype 是浮点，则默认为 1 。
> - dtype：输出的类型：float16、float32、float64、int32、orint64。
> - 种子：一个 Python 瞬间。用于为发布创造一个随机种子。查看tf.set_random_seed 行为。
> - 名称：操作的名称（任选）。



1. 使用以下语句创建一个具有一定**均值（默认值=0.0）**和**标准差（默认值=1.0）**、形状为 [M，N] 的**正态分布随机数组**：


   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G40PaX.gif)

2. 创建一个具有一定**均值（默认值=0.0）**和**标准差（默认值=1.0）**、形状为 [M，N] 的**截尾正态分布随机数组**：

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G40T54H.gif)
   
3. 要在种子的 [minval（default=0），maxval] 范围内创建形状为 [M，N] 的给定**伽马分布随机数组**，请执行如下语句：

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G4091O95.gif)
   
4. 要将给定的张量随机裁剪为指定的大小，使用以下语句：

   ```python
   tf.random_crop(t_random,[2,5],seed=12)
   ```

   > tf.random_crop()：随机地将张量变化为给定的大小。 

   这里，t_random 是一个已经定义好的张量。这将导致随机从张量 t_random 中裁剪出一个大小为 [2，5] 的张量。

   很多时候需要以随机的顺序来呈现训练样本，可以使用 **tf.random_shuffle() 来沿着它的第一维随机排列张量**。如果 t_random 是想要重新排序的张量，使用下面的代码：

   ```python
   tf.random_shuffle(t_random)
   ```

   > tf.random_shuffle()：随机地将张量沿其第一维度打乱。

5. 随机生成的张量受初始种子值的影响。要在多次运行或会话中获得相同的随机数，应该将种子设置为一个常数值。当使用大量的随机张量时，可以使用 **tf.set_random_seed() 来为所有随机产生的张量设置种子**。以下命令将所有会话的随机张量的种子设置为 54：

   ```python
   tf.set_random_seed(54)
   ```

   TIP：种子只能有整数值。

### 5.2TensorFlow 变量

它们通过使用变量类来创建。变量的定义还包括应该初始化的常量/随机值。下面的代码中创建了两个不同的张量变量 t_a 和 t_b。两者将被初始化为形状为 [50，50] 的随机均匀分布，最小值=0，最大值=10：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G41112550.gif)

> 注意：变量通常在神经网络中表示权重和偏置。

下面的代码中定义了两个变量的权重和偏置。权重变量使用正态分布随机初始化，均值为 0，标准差为 2，权重大小为 100×100。偏置由 100 个元素组成，每个元素初始化为 0。在这里也使用了可选参数名以给计算图中定义的变量命名：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G41139128.gif)

在前面的例子中，都是利用一些常量来初始化变量，也可以指定一个变量来初始化另一个变量。下面的语句将利用前面定义的权重来初始化 weight2：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G4115R07.gif)

变量的定义将指定变量如何被初始化，但是必须显式初始化所有的声明变量。在计算图的定义中通过声明初始化操作对象来实现：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G41221H7.gif)

每个变量也可以在运行图中单独使用 tf.Variable.initializer 来初始化：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G41242601.gif)

保存变量：使用 Saver 类来保存变量，定义一个 Saver 操作对象：

```python
saver = tf.train.Saver()
```

### 5.3TensorFlow 占位符

介绍完常量和变量之后，我们来讲解最重要的元素——占位符，它们用于将数据提供给计算图。可以使用以下方法定义一个占位符：

```python
tf.placeholder(dtype,shape=None,name=None)
```

dtype 定占位符的数据类型，并且必须在声明占位符时指定。在这里，为 x 定义一个占位符并计算 y=2*x，使用 feed_dict 输入一个随机的 4×5 矩阵：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G414255I.gif)

### 5.4解读分析

需要注意的是，所有常量、变量和占位符将在代码的计算图部分中定义。如果在定义部分使用 print 语句，只会得到有关张量类型的信息，而不是它的值。

为了得到相关的值，需要创建会话图并对需要提取的张量显式使用运行命令，如下所示：

```python
print(sess.run(t_1))
\#Will print the value of t_1 defined in step 1
```

### 5.5拓展阅读

很多时候需要大规模的常量张量对象；在这种情况下，为了优化内存，最好将它们声明为一个可训练标志设置为 False 的变量：

t_large = tf.Varible(large_array,trainable = False)

TensorFlow 被设计成与 Numpy 配合运行，因此所有的 TensorFlow 数据类型都是基于 Numpy 的。使用 tf.convert_to_tensor() 可以将给定的值转换为张量类型，并将其与 TensorFlow 函数和运算符一起使用。该函数接受 Numpy 数组、[Python](http://c.biancheng.net/python/) 列表和 Python 标量，并允许与张量对象互操作。

下表列出了 TensorFlow 支持的常见的数据类型：


![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G4153M25.gif)


请注意，与 Python/Numpy 序列不同，TensorFlow 序列不可迭代。试试下面的代码：

```python
for i in tf.range(10)
```

你会得到一个错误提示：

```python
\#typeError("'Tensor'object id not iterable.")
```

## 6.TensorFlow矩阵基本操作及其实现

**矩阵运算**，例如执行乘法、加法和减法，是任何神经网络中信号传播的重要操作。通常在计算中需要随机矩阵、零矩阵、一矩阵或者单位矩阵。

本节将告诉你如何获得不同类型的矩阵，以及如何对它们进行不同的矩阵处理操作。

### 6.1具体做法

开始一个**交互式会话**，以便得到计算结果：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G50A5147.gif)

一些其他有用的矩阵操作，如按元素相乘、乘以一个标量、按元素相除、按元素余数相除等，可以执行如下语句：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G50G2506.gif)

tf.div 返回的张量的类型与第一个参数类型一致。

### 6.2解读分析

所有加法、减、除、乘（按元素相乘）、取余等矩阵的算术运算都要求两个张量矩阵是相同的数据类型，否则就会产生错误。可以使用 tf.cast() 将张量从一种数据类型转换为另一种数据类型。

### 6.3拓展阅读

如果在**整数张量之间进行除法**，最好使用 **tf.truediv(a，b)**，因为它首先将整数张量转换为**浮点类**，然后再执行按位相除。

## 7.TensorFlow TensorBoard可视化数据流图

[TensorFlow](http://c.biancheng.net/tensorflow/) 使用 TensorBoard 来提供计算图形的图形图像。这使得理解、调试和优化复杂的神经网络程序变得很方便。TensorBoard 也可以提供有关网络执行的量化指标。它读取 TensorFlow 事件文件，其中包含运行 TensorFlow 会话期间生成的摘要数据。

**具体做法**

使用 TensorBoard 的第一步是确定想要的 OP 摘要。以**深度神经网络(DNN)**为例，通常需要知道损失项（目标函数）如何随时间变化。在自适应学习率的优化中，学习率本身会随时间变化。可以**在 tf.summary.scalar OP 的帮助下得到需要的术语摘要**。假设损失变量定义了误差项，我们想知道它是如何随时间变化的：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G52351363.gif)

还可以使用 tf.summary.histogram 可视化梯度、权重或特定层的输出分布：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G52411I1.gif)

摘要将在会话操作中生成。可以在计算图中定义 tf.merge_all_summaries OP 来通过一步操作得到摘要，而不需要单独执行每个摘要操作。

生成的摘要需要用事件文件写入：


![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G5245RD.gif)


这会将所有摘要和图形写入 summary_dir 目录中。现在，为了可视化摘要，需要从命令行中调用 TensorBoard：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G52524M7.gif)

接下来，打开浏览器并输入地址 http://localhost:6006/（或运行 TensorBoard 命令后收到的链接）。

你会看到类似于图 1 中的图，顶部有很多标签。Graphs（图表）选项卡能将运算图可视化：


![运算图可视化](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G5254D27.gif)
图 1 运算图可视化

## 8.TensorFlow低版本代码自动升级为1.0版本

[TensorFlow](http://c.biancheng.net/tensorflow/) 1.x 不提供向后兼容性。这意味着在 TensorFlow 0.x 上运行的代码可能无法在 TensorFlow 1.0 上运行。因此，如果代码是用 TensorFlow 0.x 框架编写的，你需要升级它们（旧的 GitHub 存储库或你自己的代码）。

这一节将指出 TensorFlow 0.x 和 TensorFlow 1.0 之间的主要区别，并展示**如何使用脚本 tf_upgrade.py 自动升级 TensorFlow 1.0 的代码。**

**具体做法**

1. 从网址 https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/compatibility 下载 tf_upgrade.py。

2. 如果要将一个文件从 TensorFlow 0.x 转换为 TensorFlow 1.0，请在命令行使用以下命令：

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G5435E52.gif)
   
3. 例如，如果有一个名为 test.py 的 TensorFlow 程序文件，可使用下述命令：

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G54531309.gif)
   

   **这将创建一个名为 test_1.0.py 的新文件。**

4. 如果要**迁移目录中的所有文件**，请在命令行中使用以下命令：

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G54450237.gif)
   
5. 在大多数情况下，该目录还包含数据集文件；可以使用以下命令确保非Python文件也会被复制到新目录（上例中的 my-dir_1p0）中：

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G54616230.gif)
   
6. 在所有这些情况下，都会生成一个 report.txt 文件。该文件包含转换的细节和过程中的任何错误。

7. **对于无法更新的部分代码，需要阅读 report.txt 文件并手动升级脚本。**

**拓展阅读**

tf_upgrade.py 有一些局限性：

- 它不能改变 tf.reverse() 的参数，因此必须手动修复。

- 对于参数列表重新排序的方法，如 tf.split() 和 tf.reverse_split()，它会尝试引入关键字参数，但实际上并不能重新排列参数。

- **有些结构必须手动替换**，例如：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G54FX47.gif)

  替换为：

  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G54I2427.gif)

## 9.TensorFlow XLA加速线性代数编译器

加速线性代数器（Accelerated linear algebra，XLA）是线性代数领域的专用编译器。根据 https://www.tensorflow.org/performance/xla/，它仍处于实验阶段，用于优化 [TensorFlow](http://c.biancheng.net/tensorflow/) 计算。

XLA 可以提高服务器和移动平台的执行速度、内存使用率和可移植性。它提供了双向 JIT（Just In Time）编译或 AoT（Ahead of Time）编译。使用 XLA，你可以生成平台相关的二进制文件（针对大量平台，如 x64、ARM等），可以针对内存和速度进行优化。

### 9.1准备工作***

目前，XLA 并不包含在 TensorFlow 的二进制版本中。用时需要从源代码构建它。

从源代码构建 TensorFlow，需要 TensorFlow 版的 LLVM 和 Bazel。**TensorFlow.org 仅支持从 macOS 和 Ubuntu 的源代码构建。**从源代码构建 TensorFlow 所需的步骤如下（参见https://www.tensorflow.org/install/install_sources）：

1. **确定要安装哪个版本的 TensorFlow——仅支持 CPU 的 TensorFlow 或支持 GPU 的 TensorFlow。**

2. 复制 TensorFlow 存储库：


   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G600504P.gif)

3. 安装以下依赖：

   - Bazel
   - TensorFlow 的 [Python](http://c.biancheng.net/python/) 依赖项
   - 对GPU版本，需要NVIDIA软件包以支持TensorFlow

4. 配置安装。在这一步中，需要选择不同的选项，如 XLA、Cuda 支持、Verbs 等：

   ./configure

5. 使用 bazel-build。

6. 对于仅使用 CPU 的版本：

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G616455F.gif)
   
7. 如果有兼容的 GPU 设备，并且需要 GPU 支持，请使用：

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G61G2R7.gif)
   
8. 成功运行后，将获得一个脚本：build_pip_package。按如下所示运行这个脚本来**构建 whl 文件**：


   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G61J35Z.gif)

9. 安装 pip 包：

   
   ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G61P2233.gif)


现在你已经准备好了。

### 9.2具体做法

TensorFlow 生成 TensorFlow 图表。在 XLA 的帮助下，可以在任何新类型的设备上运行 TensorFlow 图表。

- JIT 编译：在会话级别中打开JIT编译：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G61U15b.gif)
  
- 这是手动打开 JIT 编译：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G61914Y4.gif)
  
- 还可以通过将操作指定在特定的 XLA 设备（XLA_CPU 或 XLA_GPU）上，通过 XLA 来运行计算：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G61944563.gif)
  

  AoT编译：独立使用 tfcompile 将 TensorFlow 图转换为不同设备（手机）的可执行代码。

  TensorFlow.org 中关于 tfcompile 的论述：tfcompile 采用一个由 TensorFlow 的 feed 和 fetch 概念所标识的子图，并生成一个实现该子图的函数。feed 是函数的输入参数，fetch 是函数的输出参数。所有的输入必须完全由 feed 指定；生成的剪枝子图不能包含占位符或变量节点。通常将所有占位符和变量指定值，这可确保生成的子图不再包含这些节点。生成的函数打包为一个 cc_library，带有导出函数签名的头文件和一个包含实现的对象文件。用户编写代码以适当地调用生成的函数。

## 10.TensorFlow指定CPU和GPU设备操作详解

[TensorFlow](http://c.biancheng.net/tensorflow/) 支持 CPU 和 GPU，支持分布式计算。可以在一或多个计算机系统的多个设备上使用 TensorFlow。

> TensorFlow 
>
> **支持的 CPU 设备命名为“/device：CPU：0”（或“/cpu：0”）**；
>
> **第 i 个 GPU 设备命名为“/device：GPU：I”（或“/gpu：I”）**。

如前所述，**GPU 比 CPU 要快得多，因为它们有许多小的内核**。然而，在所有类型的计算中都使用 GPU 也并不一定都有速度上的优势。有时，比起使用 GPU 并行计算在速度上的优势收益，使用 GPU 的其他代价相对更为昂贵。

为了解决这个问题，TensorFlow 可以选择将计算放在一个特定的设备上。**默认情况下，如果 CPU 和 GPU 都存在，TensorFlow 会优先考虑 GPU。**

TensorFlow **将设备表示为字符串**。本节展示如何在 TensorFlow 中指定某一设备用于矩阵乘法的计算。

### 10.1具体做法

- 要验证 TensorFlow 是否确实在使用指定的设备（CPU 或 GPU），可以创建会话，并将 log_device_placement 标志设置为 True，即：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G63Z4413.gif)
  
- 如果你不确定设备，并希望 TensorFlow 选择现有和受支持的设备，则可以将 allow_soft_placement 标志设置为 True：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G6394C51.gif)
  
- 手动选择 CPU 进行操作：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G6400A25.gif)
  

  得到以下输出：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G640253C.gif)
  

  可以看到，在这种情况下，所有的设备都是 '/cpu：0'。

- 手动选择一个 GPU 来操作：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G64051E7.gif)

  输出现在更改为以下内容：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G64103202.gif)

  每个操作之后的'/cpu：0'现在被替换为'/gpu：0'。

- 手动选择多个GPU：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10G6415A23.gif)

  在这种情况下，如果系统有 3 个 GPU 设备，那么第一组乘法将由'/：gpu：1'执行，第二组乘以'/gpu：2'执行。

### 10.2解读分析

**函数 tf.device() 选择设备（CPU 或 GPU）**。

with 块确保设备被选择并用于其操作。

with 块中定义的所有变量、常量和操作将使用在 tf.device() 中选择的设备。

会话配置**使用 tf.ConfigProto 进行控制**。通过设置 allow_soft_placement 和 log_device_placement 标志，告诉 TensorFlow 在指定的设备不可用时自动选择可用的设备，并在执行会话时给出日志消息作为描述设备分配的输出。

## 11.浅谈深度学习之TensorFlow

**DNN（深度神经网络算法）**现在是AI社区的流行词。最近，DNN 在许多数据科学竞赛/Kaggle 竞赛中获得了多次冠军。

> 自从 1962 年 Rosenblat 提出感知机（Perceptron）以来，DNN 的概念就已经出现了，而自 Rumelhart、Hinton 和 Williams 在 1986 年发现了梯度下降算法后，DNN 的概念就变得可行了。直到最近 DNN 才成为全世界 AI/ML 爱好者和工程师的最爱。

主要原因在于现代计算能力的可用性，如 GPU 和 [TensorFlow](http://c.biancheng.net/tensorflow/) 等工具，可以通过几行代码轻松访问 GPU 并构建复杂的神经网络。

作为一名机器学习爱好者，你必须熟悉神经网络和深度学习的概念，但为了完整起见，我们将在这里介绍基础知识，并探讨 TensorFlow 的哪些特性使其成为深度学习的热门选择。

神经网络是一个生物启发式的计算和学习模型。像生物神经元一样，它们从其他细胞（神经元或环境）获得加权输入。这个加权输入经过一个处理单元并产生可以是二进制或连续（概率，预测）的输出。

**人工神经网络（ANN）**是这些神经元的网络，可以随机分布或排列成一个分层结构。这些神经元通过与它们相关的一组权重和偏置来学习。

下图对生物神经网络和人工神经网络的相似性给出了形象的对比：

![生物神经网络和人工神经网络的相似性](http://c.biancheng.net/uploads/allimg/190107/2-1Z10GGGB45.gif)
图 1 生物神经网络和人工神经网络的相似性



根据 Hinton 等人的定义，深度学习（https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf）是由多个处理层（隐藏层）组成的计算模型。层数的增加会导致学习时间的增加。由于数据量庞大，学习时间进一步增加，现今的 CNN 或生成对抗网络（GAN）的规范也是如此。

**为了实际实现 DNN，需要高计算能力**。NVDIA 公司 GPU 的问世使其变得可行，随后 Google 的 TensorFlow 使得实现复杂的 DNN 结构成为可能，而不需要深入复杂的数学细节，[大数据](http://c.biancheng.net/big_data/)集的可用性为 DNN 提供了必要的数据来源。

> TensorFlow 成为最受欢迎的深度学习库，原因如下：
>
> 1. TensorFlow 是一个强大的库，用于执行大规模的数值计算，如矩阵乘法或自动微分。这两个计算是实现和训练 DNN 所必需的。
> 2. TensorFlow 在后端使用 C/[C++](http://c.biancheng.net/cplus/)，这使得计算速度更快。
> 3. TensorFlow 有一个高级机器学习 API（tf.contrib.learn），可以更容易地配置、训练和评估大量的机器学习模型。
> 4. 可以在 TensorFlow 上使用高级深度学习库 Keras。**Keras** 非常便于用户使用，并且可以轻松快速地进行原型设计。**它支持各种 DNN，如RNN、CNN，甚至是两者的组合**。

**任何深度学习网络都由四个重要部分组成：**

​	**数据集、**

​	**定义模型（网络结构）、**

​	**训练/学习**

​	**预测/评估。**

可以在 TensorFlow 中实现所有这些。

### 11.1数据集

DNN 依赖于大量的数据。可以收集或生成数据，也可以使用可用的标准数据集。TensorFlow 支持三种主要的读取数据的方法，可以在不同的数据集中使用；本教程中用来训练建立模型的一些数据集介绍如下：

- **MNIST：这是最大的手写数字（0～9）数据库**。它由 60000 个示例的训练集和 10000 个示例的测试集组成。该数据集存放在 Yann LeCun 的主页（http://yann.lecun.com/exdb/mnist/）中。这个数据集已经包含在tensorflow.examples.tutorials.mnist 的 TensorFlow 库中。
- **CIFAR10：这个数据集包含了 10 个类别的 60000 幅 32×32 彩色图像，每个类别有 6000 幅图像。**其中训练集包含 50000 幅图像，测试数据集包含 10000 幅图像。数据集的 10 个类别分别是：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。该数据由多伦多大学计算机科学系维护（https://www.cs.toronto.edu/kriz/cifar.html）。
- **WORDNET：这是一个英文的词汇数据库。**它包含名词、动词、副词和形容词，被归为一组认知同义词（Synset），即代表相同概念的词语，例如 shut 和 close，car 和 automobile 被分组为无序集合。它包含 155287 个单词，组织在 117659 个同义词集合中，总共 206941 个单词对。该数据集由普林斯顿大学维护（https://wordnet.princeton.edu/）。
- **ImageNET：这是一个根据 WORDNET 层次组织的图像数据集（目前只有名词）。**每个有意义的概念（synset）由多个单词或单词短语来描述。每个子空间平均由 1000 幅图像表示。目前共有 21841 个同义词，共有 14197122 幅图像。自 2010 年以来，每年举办一次 ImageNet 大规模视觉识别挑战赛（ILSVRC），将图像分类到 1000 个对象类别中。这项工作是由美国普林斯顿大学、斯坦福大学、A9 和谷歌赞助（http://www.image-net.org/）。
- **YouTube-8M：这是一个由数百万 YouTube 视频组成的大型标签视频数据集。**它有大约 700 万个 YouTube 视频网址，分为 4716 个小类，并分为 24 个大类。它还提供预处理支持和框架功能。数据集由 Google Research（https://research.google.com/youtube8m/）维护。

### 11.2读取数据

在 TensorFlow 中可以通过三种方式读取数据：

1. 通过feed_dict传递数据；
2. 从文件中读取数据；
3. 使用预加载的数据；

在本教程中都使用这三种方式来读取数据。

接下来，你将依次学习每种数据读取方式。

#### 11.2.1通过feed_dict传递数据

在这种情况下，运行每个步骤时都会使用 run() 或 eval() 函数调用中的 feed_dict 参数来提供数据。这是在占位符的帮助下完成的，这个方法允许传递 Numpy 数组数据。可以使用 TensorFlow 的以下代码：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10GH344M4.gif)

这里，x 和 y 是占位符；使用它们，在 feed_dict 的帮助下传递包含 X 值的数组和包含 Y 值的数组。

#### 11.2.2从文件中读取

当数据集非常大时，使用此方法可以确保不是所有数据都立即占用内存（例如 60 GB的 YouTube-8m 数据集）。从文件读取的过程可以通过以下步骤完成：

- 使用字符串张量 ["file0"，"file1"] 或者 [("file%d"i)for in in range(2)] 的方式创建文件命名列表，或者使用 `files=tf.train.match_filenames_once('*.JPG')` 函数创建。

- 文件名队列：创建一个队列来保存文件名，此时需要使用 tf.train.string_input_producer 函数：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10GH5053N.gif)
  

  这个函数还提供了一个选项来排列和设置批次的最大数量。整个文件名列表被添加到每个批次的队列中。如果选择了 shuffle=True，则在每个批次中都要重新排列文件名。

- Reader用于从文件名队列中读取文件。根据输入文件格式选择相应的阅读器。read方法是标识文件和记录（调试时有用）以及标量字符串值的关键字。例如，文件格式为.csv 时：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10GH53R13.gif)
  
- Decoder：使用一个或多个解码器和转换操作来将值字符串解码为构成训练样本的张量：

  
  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10GH605P6.gif)

#### 11.2.3预加载的数据

当数据集很小时可以使用，可以在内存中完全加载。因此，可以将数据存储在常量或变量中。在使用变量时，需要将可训练标志设置为 False，以便训练时数据不会改变。预加载数据为 TensorFlow 常量时：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10GH6395a.gif)

> **一般来说，数据被分为三部分：训练数据、验证数据和测试数据。**

### 11.3定义模型

建立描述网络结构的**计算图**。它涉及指定信息从一组神经元到另一组神经元的超参数、变量和占位符序列以及损失/错误函数。你将在本章后面的章节中了解更多有关计算图的内容。

### 11.4训练/学习

在 DNN 中的学习通常基于梯度下降算法（后续章节将详细讨论），其目的是要找到训练变量（权重/偏置），将损失/错误函数最小化。这是通过初始化变量并使用 run() 来实现的：

![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10GHT4644.gif)

### 11.5评估模型

一旦网络被训练，通过 predict() 函数使用验证数据和测试数据来评估网络。这可以评价模型是否适合相应数据集，可以避免过拟合或欠拟合的问题。一旦模型取得让人满意的精度，就可以部署在生产环境中了。

**拓展阅读**

在 TensorFlow 1.3 中，增加了一个名为 TensorFlow Estimator 的新功能。 TensorFlow Estimator 使创建神经网络模型的任务变得更加容易，它是一个封装了训练、评估、预测和服务过程的更高层次的API。它提供了使用预先制作的估算器的选项，或者可以编写自己的定制估算器。通过预先制定的估算器，不再需要担心构建计算或创建会话，它会处理所有这些。

目前 TensorFlow Estimator 有 6 个预先制定的估算器。使用 TensorFlow 预制的 Estimator 的另一个优点是，它本身也可以在 TensorBoard 上创建可视化的摘要。

## 12.TensorFlow常用Python扩展包

[TensorFlow](http://c.biancheng.net/tensorflow/) 能够实现大部分神经网络的功能。但是，这还是不够的。对于预处理任务、序列化甚至绘图任务，还需要更多的 [Python](http://c.biancheng.net/python/) 包。

下面列出了一些常用的 Python 包：

- **Numpy：这是用 Python 进行科学计算的基础包。**它支持n维数组和矩阵的计算，还拥有大量的高级数学函数。这是 TensorFlow 所需的必要软件包，因此，使用 `pip install tensorflow` 时，如果尚未安装 Numpy，它将被自动安装。

- **Matplolib：这是 Python 2D 绘图库。**使用它可以只用几行代码创建各类图，包括直方、条形图、错误图、散点图和功率谱等。它可以使用 pip 进行安装：


  ![img](http://c.biancheng.net/uploads/allimg/190107/2-1Z10GK0252G.gif)

- **OS：这包括在基本的 Python 安装中。**它提供了一种使用操作系统相关功能（如读取、写入及更改文件和目录）的简单便携方式。

- **Pandas：这提供了各种[数据结构](http://c.biancheng.net/data_structure/)和数据分析工具。**使用 Pandas，您可以在内存数据结构和不同格式之间读取和写入数据。可以读取 .csv 和文本文件。可以使用 `pip install` 或 `conda install` 进行安装。

- **Seaborn：这是一个建立在 Matplotlib 上的专门的统计数据可视化工具。**

- **H5fs**：H5fs 是能够在 HDFS（分层数据格式文件系统）上运行的 Linux 文件系统（也包括其他带有 FUSE 实现的操作系统，如 macOS X）。

- **PythonMagick：这是 ImageMagick 库的 Python 绑定。**它是一个显示、转换和编辑光栅图像及矢量图像文件的库。它支持超过 200 个图像文件格式。它可以使用 ImageMagick 提供的源代码来安装。某些 .whl 格式也可用 pip install([http://www.lfd.uci.edu/%7Egohlke/pythonlibs/#pythonmagick](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pythonmagick)) 来安装。

- **TFlearn：TFlearn 是一个建立在 TensorFlow 之上的模块化和透明的深度学习库。**它为 TensorFlow 提供更高级别的 API，以促进和加速实验。它目前支持最近的大多数深度学习模型，如卷积、LSTM、BatchNorm、BiRNN、PReLU、残差网络和生成网络。它只适用于TensorFlow 1.0 或更高版本。请使用 `pip install tflearn` 安装。

- **Keras：Keras 也是神经网络的高级 API，它使用 TensorFlow 作为其后端。**它可以运行在 Theano 和 CNTK 之上。添加图层只需要一行代码，非常用户友好，可以使用 `pip install keras` 来安装。

## 13.回归算法有哪些，常用回归算法（3种）详解

回归是数学建模、分类和预测中最古老但功能非常强大的工具之一。回归在工程、物理学、生物学、金融、社会科学等各个领域都有应用，是数据科学家常用的基本工具。

回归通常是机器学习中使用的第一个算法。通过学习因变量和自变量之间的关系实现对数据的预测。例如，对房价估计时，需要确定房屋面积（自变量）与其价格（因变量）之间的关系，可以利用这一关系来预测给定面积的房屋的价格。可以有多个影响因变量的自变量。

因此，**回归有两个重要组成部分：自变量和因变量之间的关系，以及不同自变量对因变量影响的强度**。

以下是几种常用的回归方法：

**1.线性回归**



使用最广泛的建模技术之一。已存在 200 多年，已经从几乎所有可能的角度进行了研究。线性回归假定输入变量（X）和单个输出变量（Y）之间呈线性关系。它旨在找到预测值 Y 的线性方程：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q00925925.gif)

![image-20210618083845519](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210618083845519.png)

因此，这里尽量最小化损失函数：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q01001955.gif)


其中，需要对所有训练样本的误差求和。根据输入变量 X 的数量和类型，可划分出多种线性回归类型：简单线性回归（一个输入变量，一个输出变量），多元线性回归（多个输入变量，一个输出变量），多变量线性回归（多个输入变量，多个输出变量）。



**2.逻辑回归**



用来**确定一个事件的概率**。通常来说，事件可被表示为类别因变量。事件的概率用 logit 函数（Sigmoid 函数）表示：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q01110928.gif)

现在的目标是估计权重 W=(w1,w2,...,wn) 和偏置项 b。在逻辑回归中，使用最大似然估计量或随机梯度下降来估计系数。损失函数通常被定义为交叉熵项：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q0134H60.gif)


逻辑回归用于分类问题，例如，对于给定的医疗数据，可以使用逻辑回归判断一个人是否患有癌症。如果输出类别变量具有两个或更多个层级，则可以使用多项式逻辑回归。另一种用于两个或更多输出变量的常见技术是 OneVsAll。对于多类型逻辑回归，交叉熵损失函数被修改为：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q01429402.gif)

**3.正则化**



当有大量的输入特征时，需要正则化来确保预测模型不会太复杂。**正则化可以帮助防止数据过拟合。**它也可以用来获得一个凸损失函数。有两种类型的正则化——L1 和 L2 正则化，其描述如下：

- 当数据高度共线时，L1 正则化也可以工作。在 L1 正则化中，与所有系数的绝对值的和相关的附加惩罚项被添加到损失函数中。L1 正则化的正则化惩罚项如下：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q016222O.gif)


- L2 正则化提供了稀疏的解决方案。当输入特征的数量非常大时，非常有用。在这种情况下，惩罚项是所有系数的平方之和：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q01640945.gif)

其中，λ是正则化参数。



## 14.TensorFlow损失函数（定义和使用）详解

正如前面所讨论的，在回归中定义了损失函数或目标函数，其目的是找到使损失最小化的系数。本节将介绍如何在 [TensorFlow](http://c.biancheng.net/tensorflow/)中定义损失函数，并根据问题选择合适的损失函数。

声明一个损失函数需要将系数定义为变量，将数据集定义为占位符。可以有一个常学习率或变化的学习率和正则化常数。

在下面的代码中，设 m 是样本数量，n 是特征数量，P 是类别数量。这里应该在代码之前定义这些全局参数：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q03U9208.gif)

在**标准线性回归**的情况下，只有一个输入变量和一个输出变量：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q03919E9.gif)

在**多元线性回归**的情况下，输入变量不止一个，而输出变量仍为一个。现在可以定义占位符X的大小为 [m，n]，其中 m 是样本数量，n 是特征数量，代码如下：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q03946120.gif)

在**逻辑回归**的情况下，损失函数定义为交叉熵。输出 Y 的维数等于训练数据集中类别的数量，其中 P 为类别数量：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q04022437.gif)

如果想把 L1 正则化加到损失上，那么代码如下：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q04043X8.gif)

对于 L2 正则化，代码如下：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q04101G1.gif)

由此，你应该学会了如何实现不同类型的损失函数。那么根据手头的回归任务，你可以选择相应的损失函数或设计自己的损失函数。在损失项中也可以结合 L1 和 L2 正则化。

**拓展阅读**

为确保收敛，损失函数应为凸的。一个光滑的、可微分的凸损失函数可以提供更好的收敛性。随着学习的进行，损失函数的值应该下降，并最终变得稳定。



## 15.TensorFlow优化器种类及其用法详解

高中数学学过，函数在一阶导数为零的地方达到其最大值和最小值。梯度下降算法基于相同的原理，即调整系数（权重和偏置）使损失函数的梯度下降。

在回归中，使用梯度下降来优化损失函数并获得系数。本节将介绍如何使用 [TensorFlow](http://c.biancheng.net/tensorflow/) 的梯度下降优化器及其变体。

按照损失函数的负梯度成比例地对系数（W 和 b）进行更新。根据训练样本的大小，有三种梯度下降的变体：

1. Vanilla 梯度下降：在 Vanilla 梯度下降（也称作批梯度下降）中，在每个循环中计算整个训练集的损失函数的梯度。该方法可能很慢并且难以处理非常大的数据集。该方法能保证收敛到凸损失函数的全局最小值，但对于非凸损失函数可能会稳定在局部极小值处。
2. 随机梯度下降：在随机梯度下降中，一次提供一个训练样本用于更新权重和偏置，从而使损失函数的梯度减小，然后再转向下一个训练样本。整个过程重复了若干个循环。由于每次更新一次，所以它比 Vanilla 快，但由于频繁更新，所以损失函数值的方差会比较大。
3. 小批量梯度下降：该方法结合了前两者的优点，利用一批训练样本来更新参数。

### 15.1TensorFlow优化器的使用

首先确定想用的优化器。TensorFlow 为你提供了各种各样的优化器：

- 这里从最流行、最简单的梯度下降优化器开始：

  
  ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q10F3K7.gif)

  GradientDescentOptimizer

   

  中的 learning_rate 参数可以是一个常数或张量。它的值介于 0 和 1 之间。

  必须为优化器给定要优化的函数。使用它的方法实现最小化。该方法计算梯度并将梯度应用于系数的学习。该函数在 TensorFlow 文档中的定义如下：

  
  ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q10I3940.gif)

  综上所述，这里定义计算图：

  
  ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q10JC48.gif)

  馈送给 feed_dict 的 X 和 Y 数据可以是 X 和 Y 个点（随机梯度）、整个训练集（Vanilla）或成批次的。

- 梯度下降中的另一个变化是增加了动量项。为此，使用优化器

   

  tf.train.MomentumOptimizer()

  。它可以把 learning_rate 和 momentum 作为初始化参数：

  
  ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q10TBF.gif)
  
- 可以使用

   tf.train.AdadeltaOptimizer()

   

  来实现一个自适应的、单调递减的学习率，它使用两个初始化参数 learning_rate 和衰减因子 rho：

  
  ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q10Z5217.gif)
  
- TensorFlow 也支持 Hinton 的

   

  RMSprop

  ，其工作方式类似于 Adadelta 的 tf.train.RMSpropOptimizer()：

  
  ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q10934X6.gif)

  Adadelta 和 RMSprop 之间的细微不同可参考 http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf 和 https://arxiv.org/pdf/1212.5701.pdf。

- 另一种 TensorFlow 支持的常用优化器是

   Adam 优化器

  。该方法利用梯度的一阶和二阶矩对不同的系数计算不同的自适应学习率：

  
  ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q11052929.gif)
  
- 除此之外，TensorFlow 还提供了以下优化器：

  
  ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q11110H5.gif)


通常建议你从较大学习率开始，并在学习过程中将其降低。这有助于对训练进行微调。可以使用 TensorFlow 中的 tf.train.exponential_decay 方法来实现这一点。

根据 TensorFlow 文档，在训练模型时，通常建议在训练过程中降低学习率。该函数利用指数衰减函数初始化学习率。需要一个 global_step 值来计算衰减的学习率。可以传递一个在每个训练步骤中递增的 TensorFlow 变量。函数返回衰减的学习率。

变量：

- learning_rate：标量float32或float64张量或者[Python](http://c.biancheng.net/python/)数字。初始学习率。
- global_step：标量int32或int64张量或者Python数字。用于衰减计算的全局步数，非负。
- decay_steps：标量int32或int64张量或者Python数字。正数，参考之前所述的衰减计算。
- decay_rate：标量float32或float64张量或者Python数字。衰减率。
- staircase：布尔值。若为真则以离散的间隔衰减学习率。
- name：字符串。可选的操作名。默认为ExponentialDecay。


返回：

- 与learning_rate类型相同的标量张量。衰减的学习率。

实现指数衰减学习率的代码如下：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q11215P5.gif)



## 16.TensorFlow csv文件读取数据（代码实现）详解

大多数人了解 Pandas 及其在处理[大数据](http://c.biancheng.net/big_data/)文件方面的实用性。[TensorFlow](http://c.biancheng.net/tensorflow/) 提供了读取这种文件的方法。

前面章节中，介绍了如何在 TensorFlow 中读取文件，本节将重点介绍如何从 CSV 文件中读取数据并在训练之前对数据进行预处理。

> 本节将采用哈里森和鲁宾菲尔德于 1978 年收集的波士顿房价数据集（http://lib.stat.cmu.edu/datasets/boston），该数据集包括 506 个样本场景，每个房屋含 14 个特征：
>
> 1. CRIM：城镇人均犯罪率
> 2. ZN：占地 25000 平方英尺（1 英尺=0.3048 米）以上的住宅用地比例
> 3. INDUS：每个城镇的非零售商业用地比例
> 4. CHAS：查尔斯河（Charles River）变量（若土地位于河流边界，则为 1；否则为 0）
> 5. NOX：一氧化氮浓度（每千万）
> 6. RM：每个寓所的平均房间数量
> 7. AGE：1940 年以前建成的自住单元比例
> 8. DIS：到 5 个波士顿就业中心的加权距离
> 9. RAD：径向高速公路可达性指数
> 10. TAX：每万美元的全价值物业税税率
> 11. PTRATIO：镇小学老师比例
> 12. B：1000(Bk-0.63)2，其中 Bk 是城镇黑人的比例
> 13. LSTAT：低地位人口的百分比
> 14. MEDV：1000 美元自有住房的中位值

### 16.1TensorFlow读取csv文件过程

1. 导入所需的模块并声明全局变量：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q139553F.gif)
   
2. 定义一个将文件名作为参数的函数，并返回大小等于 BATCH_SIZE 的张量：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q140303A.gif)
   
3. 定义 f_queue 和 reader 为文件名：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q1404YK.gif)
   
4. 这里指定要使用的数据以防数据丢失。对 .csv 解码并选择需要的特征。例如，选择 RM、PTRATIO 和 LSTAT 特征：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q14111N2.gif)
   
5. 定义参数来生成批并使用 tf.train.shuffle_batch() 来随机重新排列张量。该函数返回张量 feature_batch 和 label_batch：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q14135N6.gif)
   
6. 这里定义了另一个函数在会话中生成批：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q14152502.gif)
   
7. 使用这两个函数得到批中的数据。这里，仅打印数据；在学习训练时，将在这里执行优化步骤：


   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q1420R52.gif)

### 16.2TensorFlow csv数据预处理

用前面章节提到的 TensorFlow 控制操作和张量来对数据进行预处理。例如，对于波士顿房价的情况，大约有 16 个数据行的 MEDV 是 50.0。在大多数情况下，这些数据点包含缺失或删减的值，因此建议不要考虑用这些数据训练。可以使用下面的代码在训练数据集中删除它们：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q14245532.gif)

这里定义了一个张量布尔条件，若 MEDV 等于 50.0 则为真。如果条件为真则可使用 TensorFlow tf.where() 操作赋为零值。



## 17.TensorFlow实现简单线性回归

本节将针对波士顿房价数据集的房间数量（RM）采用简单线性回归，目标是预测在最后一列（MEDV）给出的房价。

> 波士顿房价数据集可从http://lib.stat.cmu.edu/datasets/boston处获取。

本小节直接从 [TensorFlow](http://c.biancheng.net/tensorflow/) contrib 数据集加载数据。使用随机梯度下降优化器优化单个训练样本的系数。

### 17.1实现简单线性回归的具体做法

1. 导入需要的所有软件包：

   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q3060W40.gif)
   
2. 在神经网络中，所有的输入都线性增加。为了使训练有效，输入应该被归一化，所以这里定义一个函数来归一化输入数据：

   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q30A4163.gif)
   
3. 现在使用 TensorFlow contrib 数据集加载波士顿房价数据集，并将其分解为 X_train 和 Y_train。可以对数据进行归一化处理：

   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q30K3605.gif)
   
4. 为训练数据声明 TensorFlow 占位符：

   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q30RDb.gif)
   
5. 创建 TensorFlow 的权重和偏置变量且初始值为零：

   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q30U3a2.gif)
   
6. 定义用于预测的线性回归模型：

   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q30929450.gif)
   
7. 定义损失函数：

   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q3100G11.gif)
   
8. 选择梯度下降优化器：

   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q31042c0.gif)
   
9. 声明初始化操作符：

   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q3113E57.gif)
   
10. 现在，开始计算图，训练 100 次：

    ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q3122a20.gif)
    
11. 查看结果：

    ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q3130YO.gif)

### 17.2解读分析

从下图中可以看到，简单线性回归器试图拟合给定数据集的线性线：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q3135R47.gif)


在下图中可以看到，随着模型不断学习数据，损失函数不断下降：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q31414530.gif)


下图是简单线性回归器的 TensorBoard 图：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q31501X2.gif)


该图有两个名称范围节点 Variable 和 Variable_1，它们分别是表示偏置和权重的高级节点。以梯度命名的节点也是一个高级节点，展开节点，可以看到它需要 7 个输入并使用 GradientDescentOptimizer 计算梯度，对权重和偏置进行更新：



![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q315425O.gif)


### 17.3总结

本节进行了简单的线性回归，但是如何定义模型的性能呢？

有多种方法可以做到这一点。统计上来说，可以计算 R2 或将数据分为训练集和交叉验证集，并检查验证集的准确性（损失项）。



## 18.TensorFlow实现多元线性回归（超详细）

在 [TensorFlow](http://c.biancheng.net/tensorflow/) 实现简单线性回归的基础上，可通过在权重和占位符的声明中稍作修改来对相同的数据进行多元线性回归。

在多元线性回归的情况下，由于每个特征具有不同的值范围，归一化变得至关重要。这里是波士顿房价数据集的多重线性回归的代码，使用 13 个输入特征。

> 波士顿房价数据集可从http://lib.stat.cmu.edu/datasets/boston处获取。

### 18.1多元线性回归的具体实现

1. 导入需要的所有软件包：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q40030K4.gif)
   
2. 因为各特征的数据范围不同，需要归一化特征数据。为此定义一个归一化函数。另外，这里添加一个额外的固定输入值将权重和偏置结合起来。为此定义函数 append_bias_reshape()。该技巧有时可有效简化编程：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q40120B2.gif)
   
3. 现在使用 TensorFlow contrib 数据集加载波士顿房价数据集，并将其划分为 X_train 和 Y_train。注意到 X_train 包含所需要的特征。可以选择在这里对数据进行归一化处理，也可以添加偏置并对网络数据重构：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q402123Y.gif)
   
4. 为训练数据声明 TensorFlow 占位符。观测占位符 X 的形状变化：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q40313250.gif)
   
5. 为权重和偏置创建 TensorFlow 变量。通过随机数初始化权重：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q403405U.gif)
   
6. 定义要用于预测的线性回归模型。现在需要矩阵乘法来完成这个任务：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q40413T5.gif)
   
7. 为了更好地求微分，定义损失函数：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q4043QQ.gif)
   
8. 选择正确的优化器：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q40520296.gif)
   
9. 定义初始化操作符：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q40544254.gif)
   
10. 开始计算图：

    
    ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q4062B32.gif)
    
11. 绘制损失函数：

    
    ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q40A3K5.gif)
    

在这里，我们发现损失随着训练过程的进行而减少：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q40H0H6.gif)


本节使用了 13 个特征来训练模型。简单线性回归和多元线性回归的主要不同在于权重，且系数的数量始终等于输入特征的数量。下图为所构建的多元线性回归模型的 TensorBoard 图：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q40J62W.gif)


现在可以使用从模型中学到的系数来预测房价：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q40S3917.gif)



## 19.TensorFlow逻辑回归处理MNIST数据集

本节基于回归学习对 MNIST 数据集进行处理，但将添加一些 TensorBoard 总结以便更好地理解 MNIST 数据集。

> MNIST由https://www.tensorflow.org/get_started/mnist/beginners提供。

大部分人已经对 MNIST 数据集很熟悉了，它是机器学习的基础，包含手写数字的图像及其标签来说明它是哪个数字。

对于逻辑回归，对输出 y 使用独热（one-hot）编码。因此，有 10 位表示输出，每位的值为 1 或 0，独热意味着对于每个图片的标签 y，10 位中仅有一位的值为 1，其余的为 0。

因此，对于手写数字 8 的图像，其编码值为 [0000000010]：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q5435V22.gif)

### 19.1具体做法

1. 导入所需的模块：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q5441CW.gif)
   
2. 可以从模块 input_data 给出的

    

   TensorFlow

    

   示例中获取 MNIST 的输入数据。该 one_hot 标志设置为真，以使用标签的 one_hot 编码。这产生了两个张量，大小为 [55000，784] 的 mnist.train.images 和大小为 [55000，10] 的 mnist.train.labels。mnist.train.images 的每项都是一个范围介于 0 到 1 的像素强度：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q5445SO.gif)
   
3. 在 TensorFlow 图中为训练数据集的输入 x 和标签 y 创建占位符：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q5451O25.gif)
   
4. 创建学习变量、权重和偏置：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q54533194.gif)
   
5. 创建逻辑回归模型。TensorFlow OP 给出了 name_scope（"wx_b"）：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q545551C.gif)
   
6. 训练时添加 summary 操作来收集数据。使用直方图以便看到权重和偏置随时间相对于彼此值的变化关系。可以通过 TensorBoard Histogtam 选项卡看到：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q5461H35.gif)
   
7. 定义交叉熵（cross-entropy）和损失（loss）函数，并添加 name scope 和 summary 以实现更好的可视化。使用 scalar summary 来获得随时间变化的损失函数。scalar summary 在 Events 选项卡下可见：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q54640C5.gif)
   
8. 采用 TensorFlow GradientDescentOptimizer，学习率为 0.01。为了更好地可视化，定义一个 name_scope：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q54G15I.gif)
   
9. 为变量进行初始化：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q54HcA.gif)
   
10. 组合所有的 summary 操作：

    
    ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q54K0P7.gif)
    
11. 现在，可以定义会话并将所有的 summary 存储在定义的文件夹中：

    
    ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q54PR14.gif)
    
12. 经过 30 个周期，准确率达到了 86.5%；经过 50 个周期，准确率达到了 89.36%；经过 100 个周期，准确率提高到了 90.91 %。

### 19.2解读分析

这里使用张量 tensorboard--logdir=garphs 运行 TensorBoard。在浏览器中，导航到网址 localhost：6006 查看 TensorBoard。该模型图如下：

 

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q54SI48.gif)


在 Histogram 选项卡下，可以看到权重（weights）和偏置（biases）的直方图：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q54941221.gif)


权重和偏置的分布如下：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q55001Y7.gif)


可以看到，随着时间的推移，偏置和权重都发生了变化。在该示例中，根据 TensorBoard 中的分布可知偏置变化的范围更大。在 Events 选项卡下，可以看到 scalar summary，即本示例中的交叉熵。下图显示交叉熵损失随时间不断减少：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q5502c63.gif)



## 20.浅谈感知机与神经网络（无师自通）

最近十年以来，神经网络一直处于机器学习研究和应用的前沿。深度神经网络（DNN）、迁移学习以及计算高效的图形处理器（GPU）的普及使得图像识别、语音识别甚至文本生成领域取得了重大进展。

神经网络受人类大脑的启发，也被称为连接模型。像人脑一样，神经网络是大量被称为权重的突触相互连接的人造神经元的集合。

就像我们通过年长者提供的例子来学习一样，人造神经网络通过向它们提供的例子来学习，这些例子被称为训练数据集。有了足够数量的训练数据集，人造神经网络可以提取信息，并用于它们没有见过的数据。

神经网络并不是最近才出现的。第一个神经网络模型 McCulloch Pitts（MCP）（http://vordenker.de/ggphilosophy/mcculloch_a-logical-calculus.pdf）早在 1943 年就被提出来了，该模型可以执行类似与、或、非的逻辑操作。

MCP 模型的权重和偏置是固定的，因此不具备学习的可能。这个问题在若干年后的 1958 年由 Frank Rosenblatt 解决（https://blogs.umass.edu/brain-wars/files/2016/03/rosenblatt-1957.pdf）。他提出了第一个具有学习能力的神经网络，称之为感知机（perceptron）。

从那时起，人们就知道添加多层神经元并建立一个深的、稠密的网络将有助于神经网络解决复杂的任务。就像母亲为孩子的成就感到自豪一样，科学家和工程师对使用神经网络（https://www.youtube.com/watch?v=jPHUlQiwD9Y）所能实现的功能做出了高度的评价。

这些评价并不是虚假的，但是由于硬件计算的限制和网络结构的复杂，当时根本无法实现。这导致了在 20 世纪 70 年代和 80 年代出现了被称为 AI 寒冬的时期。在这段时期，由于人工智能项目得不到资助，导致这一领域的进展放缓。

随着 DNN 和 GPU 的出现，情况发生了变化。今天，可以利用一些技术通过微调参数来获得表现更好的网络，比如 dropout 和迁移学习等技术，这缩短了训练时间。最后，硬件公司提出了使用专门的硬件芯片快速地执行基于神经网络的计算。

人造神经元是所有神经网络的核心。它由两个主要部分构成：一个加法器，将所有输入加权求和到神经元上；一个处理单元，根据预定义函数产生一个输出，这个函数被称为激活函数。每个神经元都有自己的一组权重和阈值（偏置），它通过不同的学习算法学习这些权重和阈值：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q62244396.gif)


当只有一层这样的神经元存在时，它被称为感知机。输入层被称为第零层，因为它只是缓冲输入。存在的唯一一层神经元形成输出层。输出层的每个神经元都有自己的权重和阈值。

当存在许多这样的层时，网络被称为多层感知机（MLP）。MLP有一个或多个隐藏层。这些隐藏层具有不同数量的隐藏神经元。每个隐藏层的神经元具有相同的激活函数：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q623032b.gif)



上图的 MLP 具有一个有 4 个输入的输入层，5 个分别有 4、5、6、4 和 3 个神经元的隐藏层，以及一个有 3 个神经元的输出层。在该 MLP 中，下层的所有神经元都连接到其相邻的上层的所有神经元。因此，MLP 也被称为全连接层。MLP 中的信息流通常是从输入到输出，目前没有反馈或跳转，因此这些网络也被称为前馈网络。

感知机使用梯度下降算法进行训练。前面章节已经介绍了梯度下降，在这里再深入一点。感知机通过监督学习算法进行学习，也就是给网络提供训练数据集的理想输出。在输出端，定义了一个误差函数或目标函数 J(W)，这样当网络完全学习了所有的训练数据后，目标函数将是最小的。

输出层和隐藏层的权重被更新，使得目标函数的梯度减小：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q6235I49.gif)


为了更好地理解它，想象一个充满山丘、高原和凹坑的地形。目标是走到地面（目标函数的全局最小值）。如果你站在最上面，必须往下走，那么很明显你将会选择下山，即向负坡度（或负梯度）方向移动。相同的道理，感知机的权重与目标函数梯度的负值成比例地变化。

梯度的值越大，权值的变化越大，反之亦然。现在，这一切都很好，但是当到达高原时，可能会遇到问题，因为梯度是零，所以权重没有变化。当进入一个小坑（局部最小值）时，也会遇到问题，因为尝试移动到任何一边，梯度都会增加，迫使网络停留在坑中。

正如前面所述，针对增加网络的收敛性提出了梯度下降的各种变种使得网络避免陷入局部最小值或高原的问题，比如添加动量、可变学习率。

[TensorFlow](http://c.biancheng.net/tensorflow/) 会在不同的优化器的帮助下自动计算这些梯度。然而，需要注意的重要一点是，由于 TensorFlow 将计算梯度，这也将涉及激活函数的导数，所以你选择的激活函数必须是可微分的，并且在整个训练场景中具有非零梯度。

感知机中的梯度下降与梯度下降的一个主要不同是，输出层的目标函数已经被定义好了，但它也用于隐藏层神经元的权值更新。这是使用反向传播（BPN）算法完成的，输出中的误差向后传播到隐藏层并用于确定权重变化。



## 21.TensorFlow常用激活函数及其特点和用法（6种）详解

每个神经元都必须有激活函数。它们为神经元提供了模拟复杂非线性数据集所必需的非线性特性。该函数取所有输入的加权和，进而生成一个输出信号。你可以把它看作输入和输出之间的转换。使用适当的激活函数，可以将输出值限定在一个定义的范围内。

如果 xi 是第 j 个输入，Wj 是连接第 j 个输入到神经元的权重，b 是神经元的偏置，神经元的输出（在生物学术语中，神经元的激活）由激活函数决定，并且在数学上表示如下：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q64IJ27.gif)

这里，g 表示激活函数。激活函数的参数 ΣWjxj+b 被称为神经元的活动。

这里对给定输入刺激的反应是由神经元的激活函数决定的。有时回答是二元的（是或不是）。例如，当有人开玩笑的时候...要么不笑。在其他时候，反应似乎是线性的，例如，由于疼痛而哭泣。有时，答复似乎是在一个范围内。

模仿类似的行为，人造神经元使用许多不同的激活函数。在这里，你将学习如何定义和使用 [TensorFlow](http://c.biancheng.net/tensorflow/) 中的一些常用激活函数。

下面认识几种常见的激活函数：

### 21.1阈值激活函数

这是最简单的激活函数。在这里，如果神经元的激活值大于零，那么神经元就会被激活；否则，它还是处于抑制状态。下面绘制阈值激活函数的图，随着神经元的激活值的改变在 TensorFlow 中实现阈值激活函数：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q64S4527.gif)

上述代码的输出如下图所示：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q64U3D3.gif)


### 21.2Sigmoid 激活函数

在这种情况下，神经元的输出由函数 g(x)=1/(1+exp(-x)) 确定。在 TensorFlow 中，方法是 tf.sigmoid，它提供了 Sigmoid 激活函数。这个函数的范围在 0 到 1 之间：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q6493C34.gif)

在形状上，它看起来像字母 S，因此名字叫 Sigmoid：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10Q64952W5.gif)


### 21.3双曲正切激活函数

在数学上，它表示为 (1-exp(-2x)/(1+exp(-2x)))。在形状上，它类似于 Sigmoid 函数，但是它的中心位置是 0，其范围是从 -1 到 1。TensorFlow 有一个内置函数 tf.tanh，用来实现双曲正切激活函数：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QA12S61.gif)


以下是上述代码的输出：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QA14S08.gif)




### 21.4线性激活函数

在这种情况下，神经元的输出与神经元的输入值相同。这个函数的任何一边都不受限制：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QA220A7.gif)


### 21.5整流线性单元（ReLU）激活函数

也被内置在 TensorFlow 库中。这个激活函数类似于线性激活函数，但有一个大的改变：对于负的输入值，神经元不会激活（输出为零），对于正的输入值，神经元的输出与输入值相同：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QA251315.gif)


以下是 ReLU 激活函数的输出：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QA3035F.gif)


### 21.6Softmax 激活函数

是一个归一化的指数函数。一个神经元的输出不仅取决于其自身的输入值，还取决于该层中存在的所有其他神经元的输入的总和。这样做的一个优点是使得神经元的输出小，因此梯度不会过大。数学表达式为 yi =exp(xi)/Σjexp(xj)：

![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QA326343.gif)


以下是上述代码的输出：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QA345G2.gif)


下面我们逐个对上述函数进行解释：

- 阈值激活函数用于 McCulloch Pitts 神经元和原始的感知机。这是不可微的，在 x=0 时是不连续的。因此，使用这个激活函数来进行基于梯度下降或其变体的训练是不可能的。

- Sigmoid 激活函数一度很受欢迎，从曲线来看，它像一个连续版的阈值激活函数。它受到梯度消失问题的困扰，即函数的梯度在两个边缘附近变为零。这使得训练和优化变得困难。

- 双曲正切激活函数在形状上也是 S 形并具有非线性特性。该函数以 0 为中心，与 Sigmoid 函数相比具有更陡峭的导数。与 Sigmoid 函数一样，它也受到梯度消失问题的影响。

- 线性激活函数是线性的。该函数是双边都趋于无穷的 [-inf，inf]。它的线性是主要问题。线性函数之和是线性函数，线性函数的线性函数也是线性函数。因此，使用这个函数，不能表示复杂数据集中存在的非线性。

- ReLU 激活函数是线性激活功能的整流版本，这种整流功能允许其用于多层时捕获非线性。

  
  使用 ReLU 的主要优点之一是导致稀疏激活。在任何时刻，所有神经元的负的输入值都不会激活神经元。就计算量来说，这使得网络在计算方面更轻便。

  ReLU 神经元存在死亡 ReLU 的问题，也就是说，那些没有激活的神经元的梯度为零，因此将无法进行任何训练，并停留在死亡状态。尽管存在这个问题，但 ReLU 仍是隐藏层最常用的激活函数之一。

- Softmax 激活函数被广泛用作输出层的激活函数，该函数的范围是 [0，1]。在多类分类问题中，它被用来表示一个类的概率。所有单位输出和总是 1。

### 21.7总结

神经网络已被用于各种任务。这些任务可以大致分为两类：函数逼近（回归）和分类。根据手头的任务，一个激活函数可能比另一个更好。一般来说，隐藏层最好使用 ReLU 神经元。对于分类任务，Softmax 通常是更好的选择；对于回归问题，最好使用 Sigmoid 函数或双曲正切函数。



## 22.TensorFlow实现单层感知机详解

简单感知机是一个单层神经网络。它使用阈值激活函数，正如 Marvin Minsky 在论文中所证明的，它只能解决线性可分的问题。虽然这限制了**单层感知机只能应用于线性可分问题**，但它具有学习能力已经很好了。

当感知机使用阈值激活函数时，不能使用 [TensorFlow](http://c.biancheng.net/tensorflow/) 优化器来更新权重。我们将不得不使用权重更新规则：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QH210240.gif)


η 是学习率。为了简化编程，当输入固定为 +1 时，偏置可以作为一个额外的权重。那么，上面的公式可以用来同时更新权重和偏置。

下面讨论如何实现单层感知机：

1. 导入所需的模块：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QH234P9.gif)
   
2. 定义要使用的超参数：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QH304U9.gif)
   
3. 指定训练数据。在这个例子中，取三个输入神经元（A，B，C）并训练它学习逻辑 AB+BC：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QH3502J.gif)
   
4. 定义要用到的变量和用于计算更新的计算图，最后执行计算图：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QH412645.gif)
   
5. 以下是上述代码的输出：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QH42BK.gif)


那么，如果使用 Sigmoid 激活函数，而不是阈值激活函数，会发生什么？你猜对了，首先，可以使用 TensorFlow 优化器来更新权重。其次，网络将表现得像逻辑回归。



## 23.TensorFlow实现反向传播算法详解

反向传播（BPN）算法是神经网络中研究最多、使用最多的算法之一，它用于将输出层中的误差传播到隐藏层的神经元，然后用于更新权重。

> 学习 BPN 算法可以分成以下两个过程：
>
> 1. 正向传播：输入被馈送到网络，信号从输入层通过隐藏层传播到输出层。在输出层，计算误差和损失函数。
> 2. 反向传播：在反向传播中，首先计算输出层神经元损失函数的梯度，然后计算隐藏层神经元损失函数的梯度。接下来用梯度更新权重。
>    这两个过程重复迭代直到收敛。

### 23.1前期准备

首先给网络提供 M 个训练对（X，Y），X 为输入，Y 为期望的输出。输入通过激活函数 g(h) 和隐藏层传播到输出层。输出 Yhat是网络的输出，得到 error=Y-Yhat。其损失函数 J(W) 如下：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QK125Q4.gif)


其中，i 取遍所有输出层的神经元（1 到 N）。然后可以使用 J(W) 的梯度并使用链式法则求导，来计算连接第 i 个输出层神经元到第 j 个隐藏层神经元的权重 Wij 的变化：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QK15C51.gif)


这里，Oj 是隐藏层神经元的输出，h 表示隐藏层的输入值。这很容易理解，但现在怎么更新连接第 n 个隐藏层的神经元 k 到第 n+1 个隐藏层的神经元 j 的权值 Wjk？过程是相同的：将使用损失函数的梯度和链式法则求导，但这次计算 Wjk：


![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QK22C60.gif)


现在已经有方程了，看看如何在 [TensorFlow](http://c.biancheng.net/tensorflow/) 中做到这一点。在这里，还是使用 MNIST 数据集（http://yann.lecun.com/exdb/MNIST/）。

### 23.2具体实现过程

现在开始使用反向传播算法：

1. 导入模块：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QK312K9.gif)
   
2. 加载数据集，通过设置 one_hot=True 来使用独热编码标签：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QK3322H.gif)
   
3. 定义超参数和其他常量。这里，每个手写数字的尺寸是 28×28=784 像素。数据集被分为 10 类，以 0 到 9 之间的数字表示。这两点是固定的。学习率、最大迭代周期数、每次批量训练的批量大小以及隐藏层中的神经元数量都是超参数。可以通过调整这些超参数，看看它们是如何影响网络表现的：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QK404E3.gif)
   
4. 需要 Sigmoid 函数的导数来进行权重更新，所以定义它：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QK424934.gif)
   
5. 为训练数据创建占位符：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QK5032K.gif)
   
6. 创建模型：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QK55DT.gif)
   
7. 定义权重和偏置变量：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QK624424.gif)
   
8. 为正向传播、误差、梯度和更新计算创建计算图：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QK640B6.gif)
   
9. 定义计算精度 accuracy 的操作：

   
   ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QKF0138.gif)
   
10. 初始化变量：

    
    ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QKG9307.gif)
    
11. 执行图：

    
    ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QKI5349.gif)
    
12. 结果如下：

    
    ![img](http://c.biancheng.net/uploads/allimg/190108/2-1Z10QKK5J3.gif)

### 23.3解读分析

在这里，训练网络时的批量大小为 10，如果增加批量的值，网络性能就会下降。另外，需要在测试数据集上检测训练好的网络的精度，这里测试数据集的大小是 1000。

单隐藏层多层感知机在训练数据集上的准确率为 84.45，在测试数据集上的准确率为 92.1。这是好的，但不够好。MNIST 数据集被用作机器学习中分类问题的基准。接下来，看一下如何使用 TensorFlow 的内置优化器影响网络性能。



## 24.TensorFlow多层感知机实现MINIST分类（详解版）

[TensorFlow](http://c.biancheng.net/tensorflow/) 支持自动求导，可以使用 TensorFlow 优化器来计算和使用梯度。它使用梯度自动更新用变量定义的张量。本节将使用 TensorFlow 优化器来训练网络。

前面章节中，我们定义了层、权重、损失、梯度以及通过梯度更新权重。用公式实现可以帮助我们更好地理解，但随着网络层数的增加，这可能非常麻烦。

本节将使用 TensorFlow 的一些强大功能，如 Contrib（层）来定义神经网络层及使用 TensorFlow 自带的优化器来计算和使用梯度。

通过前面的学习，我们已经知道如何使用 TensorFlow 的优化器。Contrib 可以用来添加各种层到神经网络模型，如添加构建块。这里使用的一个方法是 tf.contrib.layers.fully_connected，在 TensorFlow 文档中定义如下：

![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109102200Z7.gif)

这样就添加了一个全连接层。

> 提示：上面那段代码创建了一个称为权重的变量，表示全连接的权重矩阵，该矩阵与输入相乘产生隐藏层单元的张量。如果提供了 normalizer_fn（比如batch_norm），那么就会归一化。否则，如果 normalizer_fn 是 None，并且设置了 biases_initializer，则会创建一个偏置变量并将其添加到隐藏层单元中。最后，如果 activation_fn 不是 None，它也会被应用到隐藏层单元。

### 24.1具体做法

第一步是改变损失函数，尽管对于分类任务，最好使用交叉熵损失函数。这里继续使用均方误差（MSE）：

![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109102241E1.gif)

接下来，使用 GradientDescentOptimizer：

![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10910225E37.gif)

对于同一组超参数，只有这两处改变，在测试数据集上的准确率只有 61.3%。增加 max_epoch，可以提高准确性，但不能有效地发挥 TensorFlow 的能力。

这是一个分类问题，所以最好使用交叉熵损失，隐藏层使用 ReLU 激活函数，输出层使用 softmax 函数。做些必要的修改，完整代码如下所示：

![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10910231T61.gif)

### 24.2解读分析

修改后的 MNIST MLP 分类器在测试数据集上只用了一个隐藏层，并且在 10 个 epoch 内，只需要几行代码，就可以得到 96% 的精度：


![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109102350433.jpg)


由此可见 TensorFlow 的强大之处。



## 25.TensorFlow多层感知机函数逼近过程详解

Hornik 等人的工作（http://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/notes/Sonia_Hornik.pdf）证明了一句话，“只有一个隐藏层的多层前馈网络足以逼近任何函数，同时还可以保证很高的精度和令人满意的效果。”

本节将展示如何使用多层感知机（MLP）进行函数逼近，具体来说，是预测波士顿的房价。第2章使用回归技术对房价进行预测，现在使用 MLP 完成相同的任务。

### 25.1准备工作

对于函数逼近，这里的损失函数是 MSE。输入应该归一化，隐藏层是 ReLU，输出层最好是 Sigmoid。

下面是如何使用 MLP 进行函数逼近的示例：

1. 导入需要用到的模块：sklearn，该模块可以用来获取数据集，预处理数据，并将其分成训练集和测试集；pandas，可以用来分析数据集；matplotlib 和 seaborn 可以用来可视化：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109104425Y4.gif)
   
2. 加载数据集并创建 Pandas 数据帧来分析数据：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109104443393.gif)
   
3. 了解一些关于数据的细节：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091045363U.gif)
   

   下表很好地描述了数据：


   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10910455NR.gif)

4. 找到输入的不同特征与输出之间的关联：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109104A0411.gif)
   

   以下是上述代码的输出：


   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109104FQ92.jpg)

5. 从前面的代码中，可以看到三个参数 RM、PTRATIO 和 LSTAT 在幅度上与输出之间具有大于 0.5 的相关性。选择它们进行训练。将数据集分解为训练数据集和测试数据集。使用 MinMaxScaler 来规范数据集。

   需要注意的一个重要变化是，由于神经网络使用 Sigmoid 激活函数（Sigmoid 的输出只能在 0～1 之间），所以还必须对目标值 Y 进行归一化：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109104Q6260.gif)
   
6. 定义常量和超参数：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109104T3405.gif)
   
7. 创建一个单隐藏层的多层感知机模型：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109104U9139.gif)
   
8. 声明训练数据的占位符并定义损失和优化器：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109104915Y0.gif)
   
9. 执行计算图：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091049552I.gif)

### 25.2解读分析

在只有一个隐藏层的情况下，该模型在训练数据集上预测房价的平均误差为 0.0071。下图显示了房屋估价与实际价格的关系：


![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109105013210.gif)


在这里，使用 [TensorFlow](http://c.biancheng.net/tensorflow/) 操作层（Contrib）来构建神经网络层。这使得工作稍微容易一些，因为避免了分别为每层声明权重和偏置。如果使用像 Keras 这样的 API，工作可以进一步简化。

下面是 Keras 中以 TensorFlow 作为后端的代码：

![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109105042600.gif)

前面的代码给出了预测值和实际值之间的结果。可以看到，通过去除异常值（一些房屋价格与其他参数无关，比如最右边的点），可以改善结果：


![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109105104G6.gif)



## 26.TensorFlow超参数及其调整（超详细）

正如你目前所看到的，神经网络的性能非常依赖超参数。因此，了解这些参数如何影响网络变得至关重要。

常见的超参数是学习率、正则化器、正则化系数、隐藏层的维数、初始权重值，甚至选择什么样的优化器优化权重和偏置。

### 26.1超参数调整过程

1. 调整超参数的第一步是构建模型。与之前一样，在 [TensorFlow](http://c.biancheng.net/tensorflow/) 中构建模型。

2. 添加一种方法将模型保存在 model_file 中。在 TensorFlow 中，可以使用 Saver 对象来完成。然后保存在会话中：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109110429239.gif)
   
3. 确定要调整的超参数，并为超参数选择可能的值。在这里，你可以做随机的选择、固定间隔值或手动选择。三者分别称为随机搜索、网格搜索和手动搜索。例如，下面是用来调节学习率的代码：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109110452448.gif)
   
4. 选择对损失函数给出最佳响应的参数。所以，可以在开始时将损失函数的最大值定义为 best_loss（如果是精度，可以选择将自己期望得到的准确率设为模型的最低精度）：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109110549515.gif)
   
5. 把你的模型放在 for 循环中，然后保存任何能更好估计损失的模型：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109110Q5101.gif)


除此之外，贝叶斯优化也可以用来调整超参数。其中，用高斯过程定义了一个采集函数。高斯过程使用一组先前评估的参数和得出的精度来假定未观察到的参数。采集函数使用这一信息来推测下一组参数。https://github.com/lucfra/RFHO上有一个包装器用于基于梯度的超参数优化。

**拓展阅读**

- 关于超参数优化的另一个资源：http://fastml.com/optimizing-hyperparams-with-hyperopt/。
- Bengio 和其他人关于超参数优化的各种算法的详细论文：https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf



## 27.TensorFlow Keras及其用法（无师自通）

Keras 是与 [TensorFlow](http://c.biancheng.net/tensorflow/) 一起使用的更高级别的作为后端的 API。添加层就像添加一行代码一样简单。在模型架构之后，使用一行代码，你可以编译和拟合模型。之后，它可以用于预测。变量声明、占位符甚至会话都由 API 管理。

### 27.1具体做法

1. 定义模型的类型。Keras 提供了两种类型的模型：序列和模型类 API。Keras 提供各种类型的神经网络层：


   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109112Q1628.gif)

2. 在 model.add() 的帮助下将层添加到模型中。依照 Keras 文档描述，Keras 提供全连接层的选项（针对密集连接的神经网络）：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109112S0512.gif)

   注意，密集层实现的操作：output=activation(dot(input，kernel)+bias)，其中 activation 是元素激活函数，是作为激活参数传递的，kernel 是由该层创建的权重矩阵，bias 是由该层创建的偏置向量（仅在 use_bias 为 True 时适用）。

3. 可以使用它来添加尽可能多的层，每个隐藏层都由前一层提供输入。只需要为第一层指定输入维度：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109112924138.gif)
   
4. 一旦模型被定义，需要选择一个损失函数和优化器。Keras 提供了多种损失函数（mean_squared_error、mean_absolute_error、mean_absolute_percentage_error、categorical_crossentropy 和优化器（sgd、RMSprop、Adagrad、Adadelta、Adam 等）。损失函数和优化器确定后，可以使用 compile（self，optimizer，loss，metrics=None，sample_weight_mode=None）来配置学习过程：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091129564K.gif)
   
5. 使用 fit 方法训练模型：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091130193Q.gif)
   
6. 可以在 predict 方法 predict(self，x，batch_size=32，verbose=0) 的帮助下进行预测：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109113040I0.gif)

> Keras 提供选项来添加卷积层、池化层、循环层，甚至是局部连接层。每种方法的详细描述在 Keras 的官方文档中可以找到：https://keras.io/models/sequential/。



## 28.卷积神经网络(CNN,ConvNet)及其原理详解

卷积神经网络（CNN，有时被称为 ConvNet）是很吸引人的。在短时间内，它们变成了一种颠覆性的技术，打破了从文本、视频到语音等多个领域所有最先进的算法，远远超出了其最初在图像处理的应用范围。

CNN 由许多神经网络层组成。卷积和池化这两种不同类型的层通常是交替的。网络中每个滤波器的深度从左到右增加。最后通常由一个或多个全连接的层组成：


![卷积神经网络的一个例子](http://c.biancheng.net/uploads/allimg/190109/2-1Z10913095L37.gif)
图 1 卷积神经网络的一个例子


Convnets 背后有三个关键动机：局部感受野、共享权重和池化。让我们一起看一下。

### 28.1局部感受野

如果想保留图像中的空间信息，那么用像素矩阵表示每个图像是很方便的。然后，编码局部结构的简单方法是将相邻输入神经元的子矩阵连接成属于下一层的单隐藏层神经元。这个单隐藏层神经元代表一个局部感受野。请注意，此操作名为“卷积”，此类网络也因此而得名。

当然，可以通过重叠的子矩阵来编码更多的信息。例如，假设每个子矩阵的大小是 5×5，并且将这些子矩阵应用到 28×28 像素的 MNIST 图像。然后，就能够在下一隐藏层中生成 23×23 的局部感受野。事实上，在触及图像的边界之前，只需要滑动子矩阵 23 个位置。

定义从一层到另一层的特征图。当然，可以有多个独立从每个隐藏层学习的特征映射。例如，可以从 28×28 输入神经元开始处理 MNIST 图像，然后（还是以 5×5 的步幅）在下一个隐藏层中得到每个大小为 23×23 的神经元的 k 个特征图。

### 28.2共享权重和偏置

假设想要从原始像素表示中获得移除与输入图像中位置信息无关的相同特征的能力。一个简单的直觉就是对隐藏层中的所有神经元使用相同的权重和偏置。通过这种方式，每层将从图像中学习到独立于位置信息的潜在特征。

理解卷积的一个简单方法是考虑作用于矩阵的滑动窗函数。在下面的例子中，给定输入矩阵 I 和核 K，得到卷积输出。将 3×3 核 K（有时称为滤波器或特征检测器）与输入矩阵逐元素地相乘以得到输出卷积矩阵中的一个元素。所有其他元素都是通过在 I 上滑动窗口获得的：

 


![卷积运算的一个例子：用粗体表示参与计算的单元](http://c.biancheng.net/uploads/allimg/190109/2-1Z109131149432.gif)

图 2 卷积运算的一个例子：用粗体表示参与计算的单元


在这个例子中，一触及 I 的边界就停止滑动窗口（所以输出是 3×3）。或者，可以选择用零填充输入（以便输出为 5×5），这是有关填充的选择。

另一个选择是关于滑窗所采用的滑动方式的步幅。步幅可以是 1 或大于 1。大步幅意味着核的应用更少以及更小的输出尺寸，而小步幅产生更多的输出并保留更多的信息。

滤波器的大小、步幅和填充类型是超参数，可以在训练网络时进行微调。

#### [TensorFlow](http://c.biancheng.net/tensorflow/)中的ConvNet

在 TensorFlow 中，如果想添加一个卷积层，可以这样写：

![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109131344225.gif)

参数说明如下：

- input：张量，必须是 half、float32、float64 三种类型之一。
- filter：张量必须具有与输入相同的类型。
- strides：整数列表。长度是 4 的一维向量。输入的每一维度的滑动窗口步幅。必须与指定格式维度的顺序相同。
- padding：可选字符串为 SAME、VALID。要使用的填充算法的类型。
- use_cudnn_on_gpu：一个可选的布尔值，默认为 True。
- data_format：可选字符串为 NHWC、NCHW，默认为 NHWC。指定输入和输出数据的数据格式。使用默认格式 NHWC，数据按照以下顺序存储：[batch，in_height，in_width，in_channels]。或者，格式可以是 NCHW，数据存储顺序为：[batch，in_channels，in_height，in_width]。
- name：操作的名称（可选）。


下图提供了一个卷积的例子：

![卷积运算的例子](http://c.biancheng.net/uploads/allimg/190109/2-1Z109131421P1.gif)

图 3 卷积运算的例子

### 28.3池化层

假设我们要总结一个特征映射的输出。我们可以使用从单个特征映射产生的输出的空间邻接性，并将子矩阵的值聚合成单个输出值，从而合成地描述与该物理区域相关联的含义。

#### 28.3.1最大池化

一个简单而通用的选择是所谓的最大池化算子，它只是输出在区域中观察到的最大输入值。在 TensorFlow 中，如果想要定义一个大小为 2×2 的最大池化层，可以这样写：

![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091315132Y.gif)

参数说明如下：

- value：形状为 [batch，height，width，channels] 和类型是 tf.float32 的四维张量。
- ksize：长度 >=4 的整数列表。输入张量的每个维度的窗口大小。
- strides：长度 >=4 的整数列表。输入张量的每个维度的滑动窗口的步幅。
- padding：一个字符串，可以是 VALID 或 SAME。
- data_format：一个字符串，支持 NHWC 和 NCHW。
- name：操作的可选名称。


下图给出了最大池化操作的示例：


![池化操作的一个例子](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091315512W.gif)
图 4 池化操作的一个例子

#### 28.3.2平均池化

另一个选择是平均池化，它简单地将一个区域聚合成在该区域观察到的输入值的平均值。

TensorFlow 可以实现大量的池化层，并在线提供了一个完整的列表（https://www.tensorflow.org/api_guides/python/nn#Pooling）。简而言之，所有池化操作只不过是给定区域的汇总操作。

### 28.4ConvNet总结

CNN 基本上是几层具有非线性激活函数的卷积，以及将池化层应用于卷积的结果。每层应用不同的滤波器（成百上千个）。理解的关键是滤波器不是预先设定好的，而是在训练阶段学习的，以使得恰当的损失函数被最小化。已经观察到，较低层会学习检测基本特征，而较高层检测更复杂的特征，例如形状或面部。

请注意，由于有池化层，靠后的层中的神经元看到的更多的是原始图像，因此它们能够编辑前几层中学习的基本特征。

到目前为止，描述了 ConvNet 的基本概念。CNN 在时间维度上对音频和文本数据进行一维卷积和池化操作，沿（高度×宽度）维度对图像进行二维处理，沿（高度×宽度×时间）维度对视频进行三维处理。对于图像，在输入上滑动滤波器会生成一个特征图，为每个空间位置提供滤波器的响应。

换句话说，一个 ConvNet 由多个滤波器堆叠在一起，学习识别在图像中独立于位置信息的具体视觉特征。这些视觉特征在网络的前面几层很简单，然后随着网络的加深，组合成更加复杂的全局特征。



## 29.三维卷积神经网络预测MNIST数字详解

在这一节中，你将学习如何创建一个简单的三层卷积网络来预测 MNIST 数字。这个深层网络由两个带有 ReLU 和 maxpool 的卷积层以及两个全连接层组成。

MNIST 由 60000 个手写体数字的图片组成。本节的目标是高精度地识别这些数字。

### 29.1具体实现过程

1. 导入 tensorflow、matplotlib、random 和 numpy。然后，导入 mnist 数据集并进行独热编码。请注意，

   TensorFlow

    

   有一些内置的库来处理 MNIST，我们也会用到它们：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109134645493.gif)
   
2. 仔细观察一些数据有助于理解 MNIST 数据集。了解训练数据集中有多少张图片，测试数据集中有多少张图片。可视化一些数字，以便了解它们是如何表示的。这种输出可以对于识别手写体数字的难度有一种视觉感知，即使是对于人类来说也是如此。

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109134G5325.gif)
   

   上述代码的输出：

   
   ![MNIST手写数字的一个例子](http://c.biancheng.net/uploads/allimg/190109/2-1Z109134S6342.gif)

   图 1 MNIST手写数字的一个例子
   
3. 设置学习参数 batch_size和display_step。另外，MNIST 图片都是 28×28 像素，因此设置 n_input=784，n_classes=10 代表输出数字 [0-9]，并且 dropout 概率是 0.85，则：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109134915493.gif)
   
4. 设置 TensorFlow 计算图的输入。定义两个占位符来存储预测值和真实标签：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10913493JW.gif)
   
5. 定义一个输入为 x，权值为 W，偏置为 b，给定步幅的卷积层。激活函数是 ReLU，padding 设定为 SAME 模式：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109135005151.gif)
   
6. 定义一个输入是 x 的 maxpool 层，卷积核为 ksize 并且 padding 为 SAME：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091350559D.gif)
   
7. 定义 convnet，其构成是两个卷积层，然后是全连接层，一个 dropout 层，最后是输出层：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10913511I22.gif)
   
8. 定义网络层的权重和偏置。第一个 conv 层有一个 5×5 的卷积核，1 个输入和 32 个输出。第二个 conv 层有一个 5×5 的卷积核，32 个输入和 64 个输出。全连接层有 7×7×64 个输入和 1024 个输出，而第二层有 1024 个输入和 10 个输出对应于最后的数字数目。所有的权重和偏置用 randon_normal 分布完成初始化：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109135205U2.gif)
   
9. 建立一个给定权重和偏置的 convnet。定义基于 cross_entropy_with_logits 的损失函数，并使用 Adam 优化器进行损失最小化。优化后，计算精度：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109135231521.gif)
   
10. 启动计算图并迭代 training_iterats次，其中每次输入 batch_size 个数据进行优化。请注意，用从 mnist 数据集分离出的 mnist.train 数据进行训练。每进行 display_step 次迭代，会计算当前的精度。最后，在 2048 个测试图片上计算精度，此时无 dropout。

    
    ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109135309512.gif)
    
11. 画出每次迭代的 Softmax 损失以及训练和测试的精度：

    
    ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109135335M7.gif)
    

    以下是上述代码的输出。首先看一下每次迭代的 Softmax 损失：

    
    ![减少损失的一个例子](http://c.biancheng.net/uploads/allimg/190109/2-1Z10913535E17.gif)
    图 2 减少损失的一个例子

再来看一下训练和测试的精度：

 

![训练和测试精度上升的一个例子](http://c.biancheng.net/uploads/allimg/190109/2-1Z109135425244.gif)

图 3 训练和测试精度上升的一个例子

> **解读分析**
>
> 使用 ConvNet，在 MNIST 数据集上的表现提高到了近 95% 的精度。ConvNet 的前两层网络由卷积、ReLU 激活函数和最大池化部分组成，然后是两层全连接层（含dropout）。训练的 batch 大小为 128，使用 Adam 优化器，学习率为 0.001，最大迭代次数为 500 次。



## 30.卷积神经网络分类图片过程详解

在这一节中，你将学习如何对 CIFAR-10 中的图片进行分类。

CIFAR-10 数据集由 10 类 60000 张 32×3 2像素的彩色图片组成，每类有 6000 张图片。有 50000 张训练图片和 10000 张测试图片。下面的图片取自https://www.cs.toronto.edu/~kriz/cifar.html：


![CIFAR图像的例子](http://c.biancheng.net/uploads/allimg/190109/2-1Z109141033C3.jpg)
图 1 CIFAR图像的例子


在这一节，将使用 TFLearn（一个更高层次的框架），它抽象了一些 [TensorFlow](http://c.biancheng.net/tensorflow/) 的内部细节，能够专注于深度网络的定义。可以在 http://tflearn.org/ 上了解 TFLearn 的信息，这里的代码是标准发布的一部分，网址为https://github.com/tflearn/tflearn/tree/master/examples。

### 30.1具体操作过程

1. 导入几个 utils 和核心层用于实现 ConvNet、dropout、fully_connected 和 max_pool。另外，导入一些对图像处理和图像增强有用的模块。请注意，TFLearn 为 ConvNet 提供了一些已定义的更高级别的层，这能够专注于代码的定义：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10914121a28.gif)
   
2. 加载 CIFAR-10 数据，并将其分为 X_train 和 Y_train，X_test 用于测试，Y_test 是测试集的标签。对 X 和 Y 进行混洗可能是有用的，因为这样能避免训练依赖于特定的数据配置。最后一步是对 X 和 Y 进行独热编码：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10914125A11.gif)
   
3. 使用 ImagePreprocessing() 对数据集进行零中心化（即对整个数据集计算平均值），同时进行 STD 标准化（即对整个数据集计算标准差）。TFLearn 数据流旨在通过 CPU 先对数据进行预处理，然后在 GPU 上加速模型训练：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10914141H40.gif)
   
4. 通过随机左右翻转和随机旋转来增强数据集。这一步是一个简单的技巧，用于增加可用于训练的数据：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109141333F1.gif)
   
5. 使用之前定义的图片预处理和图片增强操作创建卷积网络。网络由三个卷积层组成。第一层有 32 个卷积核，尺寸是 3×3，激活函数用 ReLU，这一层后使用 max_pool 层用于缩小尺寸。然后是两个卷积核级联，卷积核的个数是 64，尺寸是 3×3，激活函数是 ReLU。之后依次是 max_pool 层，具有 512 个神经元、激活函数为 ReLU 的全连接的网络，设置 dropout 概率为 50%。最后一层是全连接层，利用 10 个神经元和激活函数 softmax 对 10 个手写数字进行分类。请注意，这种特殊类型的 ConvNet 在 CIFAR-10 中非常有效。其中，使用 Adam 优化器（categorical_crossentropy）学习率是 0.001：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109141509105.gif)
   
6. 实例化 ConvNet 并以 batch_size=96 训练 50 个 epoch：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109141532A4.gif)

### 30.2解读分析

TFLearn 隐藏了许多 TensorFlow 的实现细节，在许多情况下，它可专注于具有更高抽象级别的 ConvNet 的定义。我们的设计在 50 次迭代中达到了 88% 的准确度。以下图片是在 Jupyter notebook 中执行的快照：

 

![Jupyter 执行 CIFAR10 分类的一个例子](http://c.biancheng.net/uploads/allimg/190109/2-1Z109141F3347.gif)

图 2 Jupyter 执行 CIFAR10 分类的一个例子

要安装 TFLearn，请参阅“安装指南”（http://tflearn.org/installation），如果你想查看更多示例，可以在网上找到已经很成熟的解决方案列表（http://tflearn.org/examples/）。



## 31.迁移学习及实操（使用预训练的VGG16网络）详解

本节讨论迁移学习，它是一个非常强大的深度学习技术，在不同领域有很多应用。动机很简单，可以打个比方来解释。假设你想学习一种新的语言，比如西班牙语，那么从你已经掌握的另一种语言（比如英语）学起，可能是有用的。

按照这种思路，计算机视觉研究人员通常使用预训练 CNN 来生成新任务的表示，其中数据集可能不够大，无法从头开始训练整个 CNN。另一个常见的策略是采用在 ImageNet 上预训练好的网络，然后通过微调整个网络来适应新任务。

这里提出的例子受启于 Francois Chollet 写的关于 Keras 的一个非常有名的博客（https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html）。

这个想法是使用在像 ImageNet 这样的大型数据集上预先训练的 VGG16 网络。注意，训练的计算量可能相当大，因此使用已经预训练的网络是有意义的：


![一个 VGG16 网络](http://c.biancheng.net/uploads/allimg/190109/2-1Z10915145b56.gif)
图 1 一个 VGG16 网络


那么，如何使用 VGG16 呢？Keras 使其变得容易，因为有一个标准的 VGG16 模型可以作为一个库来使用，预先计算好的权重会自动下载。请注意，这里省略了最后一层，并将其替换为自定义层，该层将在预定义的 VGG16 的顶部进行微调。

例如，下面你将学习如何分类 Kaggle 提供的狗和猫的图片：

1. 从 Kaggle（https://www.kaggle.com/c/dogs-vs-cats/data）下载狗和猫的数据，并创建一个包含两个子目录（train 和 validation）的数据目录，每个子目录有两个额外的子目录，分别是 dogs 和 cats。

2. 导入稍后将用于计算的 Keras 模块，并保存一些有用的常量：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109151621433.gif)
   
3. 加载 ImageNet 上预训练的 VGG16 网络，省略最后一层，因为这里将在预建的 VGG16 网络的顶部添加自定义分类网络，并替换原来 VGG16 的分类层：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109151F4638.gif)
   

   上述代码的输出如下：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109151H4F1.gif)
   
4. 冻结预训练的 VGG16 网络的一定数量的较低层。在这里决定冻结最前面的 15 层：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109151P5936.gif)
   
5. 为了分类，添加一组自定义的顶层：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109151S2545.gif)
   
6. 自定义网络应该单独进行预训练，为了简单起见，这里省略了这部分，将此任务交给读者：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109151Z9493.gif)
   
7. 创建一个新的网络，这是预训练的 VGG16 网络和预训练的定制网络的组合体：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109151934I9.gif)
   
8. 重新训练组合的新模型，仍然保持 VGG16 的 15 个最低层处于冻结状态。在这个特定的例子中，也使用 Image Augumentator 来增强训练集：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109152029225.gif)
   
9. 在组合网络上评估结果：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10915204L13.gif)

**解读分析**

一个标准的 VGG16 网络已经在整个 ImageNet 上进行了预训练，并且使用了预先计算好的从网上下载的权值。这个网络和一个已经被单独训练的定制网络并置在一起。然后，并置的网络作为一个整体被重新训练，同时保持 VGG16 的 15 个低层的参数不变。

这个组合非常有效。它可以节省大量的计算能力，重新利用已经工作的 VGG16 网络进行迁移学习，该网络已经在 ImageNet 上完成了学习，可以将此学习应用到新的特定领域，通过微调去完成分类任务。

根据具体的分类任务，有几条经验法则需要考虑：

- 如果新的数据集很小，并且与ImageNet数据集相似，那么可以冻结所有的VGG16网络并仅重新训练定制网络。这样，也可以最小化组合网络过度拟合的风险。

  可运行代码 base_model.layers：layer.trainable=False 冻结所有低层参数。

- 如果新数据集很大并且与ImageNet数据集相似，那么可以重新训练整个并置网络。仍然保持预先计算的权重作为训练起点，并通过几次迭代进行微调：

  可运行代码 model.layers：layer.trainable=True 取消冻结所有低层的参数。

- 如果新数据集与ImageNet数据集有很大的不同，实际上仍然可以使用预训练模型的权值进行初始化。在这种情况下，将有足够的数据和信心通过整个网络进行微调。更多信息请访问http://cs231n.github.io/transfer-learning/。



## 32.DeepDream网络（TensorFlow创建）详解

Google 于 2014 年在 ImageNet 大型视觉识别竞赛（ILSVRC）训练了一个神经网络，并于 2015 年 7 月开放源代码。

该网络学习了每张图片的表示。低层学习低级特征，比如线条和边缘，而高层学习更复杂的模式，比如眼睛、鼻子、嘴巴等。因此，如果试图在网络中表示更高层次的特征，我们会看到从原始 ImageNet 中提取的不同特征的组合，例如鸟的眼睛和狗的嘴巴。

考虑到这一点，如果拍摄一张新的图片，并尝试最大化与网络高层的相似性，那么结果会得到一张新的视觉体验的图片。在这张新视觉体验的图片中，由高层学习的一些模式如同是原始图像的梦境一般。下图是一张想象图片的例子：


![Google Deep Dreams 的示例](http://c.biancheng.net/uploads/allimg/190109/2-1Z109155154117.jpg)
图 1 Google Deep Dreams 的示例

### 32.1准备工作

从网上下载预训练的 Inception 模型（https://github.com/martinwicke/tensorflow-tutorial/blob/master/tensorflow_inception_graph.pb）。

### 32.2具体做法

1. 导入 numpy 进行数值计算，functools 定义一个或多个参数已经填充的偏函数，Pillow 用于图像处理，matplotlib 用于产生图像：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091552505O.gif)
   
2. 设置内容图像和预训练模型的路径。从随机噪声的种子图像开始：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10915530c05.gif)
   
3. 以 graph 的形式加载从网上下载的 Inception 网络。初始化一个

    

   TensorFlow

    

   会话，用 FastGFile(..) 加载这个 graph，并用 ParseFromstring(..) 解析该 graph。之后，使用 placeholder(..) 方法创建一个占位符作为输入。 imagenet_mean 是预先计算的常数，这里的内容图像减去该值以实现数据标准化。事实上，这是训练得到的平均值，规范化使得收敛更快。该值将从输入中减去并存储在 t_preprocessed 变量中，然后用于加载 graph 定义：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10915535H21.gif)
   
4. 定义一些 util 函数来可视化图像，并将 TF-graph 生成函数转换为常规

    

   Python

    

   函数（请参阅下面的示例）：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109155422340.gif)
   
5. 计算图像的梯度上升值。为了提高效率，应用平铺计算，其中在不同的图块上计算单独的梯度上升。通过多次迭代对图像应用随机偏移以模糊图块的边界：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109155451M1.gif)
   
6. 定义用来减少输入层均值的优化对象。通过考虑输入张量，该梯度函数可以计算优化张量的符号梯度。为了提高效率，图像被分割成几块，然后调整大小并添加到块数组中。对于每个块，使用 calc_grad_tiled 函数：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109155546109.gif)
   
7. 加载特定的内容图像，并开始想象。在这个例子中，作者的脸被转化成类似于狼的模样：

   
   ![深度转换的例子，其中一个作者变成了狼](http://c.biancheng.net/uploads/allimg/190109/2-1Z10915560Y93.jpg)

   图 2 深度转换的例子，其中一个作者变成了狼

### 32.3解读分析

神经网络存储训练图像的抽象描述：较低层存储线条和边缘等特征，较高层存储较复杂的图像特征，如眼睛、脸部和鼻子。通过应用梯度上升过程，这里使损失函数最大化并促使发现类似于由较高层记忆的图案的内容图片模式。这样网络就生成了令人致幻的图片。



## 33.TensorFlow实现文本情感分析详解

前面我们介绍了如何将卷积网络应用于图像。本节将把相似的想法应用于文本。

文本和图像有什么共同之处？乍一看很少。但是，如果将句子或文档表示为矩阵，则该矩阵与其中每个单元是像素的图像矩阵没有什么区别。

接下来的问题是，如何能够将文本表示为矩阵？好吧，这很简单：矩阵的每一行都是一个表示文本的向量。当然，现在需要定义一个基本单位。一个简单方法是将基本单位表示为字符。另一种做法是将一个单词看作基本单位，将相似的单词聚合在一起，然后用表示符号表示每个聚合（有时称为聚类或嵌入）。

请注意，无论如何选择基本单位，都需要完成一个从基本单位到整数值地址的一一映射，以便可以将文本视为矩阵。例如，有10行文字，每行都是一个100维的嵌入，那么将其表示为10×100的矩阵。在这个特别的文本图像中，一个像素表示该句子x在位置y处有相应的嵌入。

你也许会注意到，文本并不是一个真正的矩阵，而是一个矢量，因为位于相邻行中的两个单词几乎没有什么关联。实际上，位于相邻列中的两个单词最有可能具有某种相关性，这是文本矩阵与图像的主要差异。

现在你可能想问：我明白你是想把文本当成一个向量，但是这样做就失去了这个词的位置信息，这个位置信息应该是很重要的，不是吗？

其实，事实证明，在很多真实的应用程序中，知道一个句子是否包含一个特定的基本单位（一个字符、一个单词或一个聚合体）是非常准确的信息，即使不去记住其在句子中的确切位置。

本节将使用 TFLearn 创建一个基于 CNN 的情感分析深度学习网络。正如前一节所讨论的，这里的 CNN 是一维的。

这里将使用 IMDb 数据集，收集 45000 个高度受欢迎的电影评论样本进行训练，并用 5000 个样本进行测试。TFLearn有从网络自动下载数据集的库，便于创建卷积网络，所以可以直接编写代码。

### 33.1文本情感分析实现过程

1. 导入 

   TensorFlow、tflearn 以及构建网络所需要的模块。然后导入 IMDb 库并执行独热编码和填充：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109163614354.gif)
   
2. 加载数据集，用 0 填充整个句子至句子的最大长度，然后在标签上进行独热编码，其中两个数值分别对应 true 和 false 值。请注意，参数 n_words 是词汇表中单词的个数。表外的单词均设为未知。此外，请注意 trainX 和 trainY 是稀疏向量，因为每个评论可能仅包含整个单词集的一个子集。

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109163A0564.gif)
   
3. 显示几个维度来检查刚刚处理的数据，并理解数据维度的含义：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109163G0E6.gif)
   
4. 为数据集中包含的文本构建一个嵌入。就目前而言，考虑这个步骤是一个黑盒子，它把这些词汇映射聚类，以便类似的词汇可能出现在同一个聚类中。请注意，在之前的步骤中，词汇是离散和稀疏的。通过嵌入操作，这里将创建一个将每个单词嵌入连续密集向量空间的映射。使用这个向量空间表示将给出一个连续的、分布式的词汇表示。如何构建嵌入，将在讨论RNN时详细讲解：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109163I2463.gif)
   
5. 创建合适的卷积网络。这里有三个卷积层。由于正在处理文本，这里将使用一维卷积网络，这些图层将并行执行。每一层需要一个 128 维的张量（即嵌入输出），并应用多个具有有效填充的滤波器（分别为 3、4、5）、激活函数 ReLU 和 L2 regularizer。然后将每个图层的输出通过合并操作连接起来。接下来添加最大池层，以 50% 的概率丢弃参数的 dropout 层。最后一层是使用 softmax 激活的全连接层：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109163PM22.gif)
   
6. 学习阶段使用 Adam 优化器以及 categorical_crossentropy 作为损失函数：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109163S2416.gif)
   
7. 在训练中，采用 batch_size=32，观察在训练和验证集上达到的准确度。正如你所看到的，在通过电影评论预测情感表达时能够获得 79% 的准确性：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z109163Z5K1.gif)

### 33.2解读分析

论文“[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)”详细阐述了用于情感分析的一维卷积网络。请注意，得益于滤波器窗口在连续单词上的操作，文章提出的模型保留了一些位置信息。文中配图给出了网络中的关键点。在开始时，文本被表示为基于标准嵌入的向量，在一维密集空间中提供了紧凑的表示，然后用多个标准的一维卷积层处理这些矩阵。

请注意，该模型使用了多个具有不同窗口大小的滤波器来获取多个特征。之后，用一个最大池化操作来保留最重要的特征，即每个特征图中具有最高值的特征。为防止过度拟合，文章提出在倒数第二层采用一个 dropout 和用权向量的 L2 范数进行约束。最后一层输出情感为正面或者负面。

为了更好地理解模型，有几个观察结果展示如下：

- 滤波器通常在连续的空间上进行卷积。对于图像来说，这个空间是指高度和宽度上连续的像素矩阵表示。对于文本来说，连续的空间不过是连续词汇自然产生的连续维度。如果只使用独热编码来表示单词，那么空间是稀疏的，如果使用嵌入，则结果空间是密集的，因为相似的单词被聚合。
- 图像通常有三个颜色通道（RGB），而文本自然只有一个通道，因为不需要表示颜色。


论文“Convolutional Neural Networks for Sentence Classification”针对句子分类开展了一系列的实验。除了对超参数的微调，具有一层卷积的简单 CNN 在句子分类中表现出色。文章还表明采用一套静态嵌入。（这将在讨论 RNN 时讨论），并在其上构建一个非常简单的 CNN，可以显著提升情感分析的性能：


![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z10916405X46.gif)
图 1 例句的两通道模型结构示例图

一个模型结构的示例的链接：https://arxiv.org/pdf/1408.5882.pdf

使用 CNN 进行文本分析是一个活跃的研究领域。我建议看[Text Understanding from Scratch，Xiang Zhang，Yann LeCun](https://arxiv.org/abs/1502.01710) 。这篇文章证明可以使用 CNN 将深度学习应用到从字符级输入一直到抽象文本概念的文本理解。作者将 CNN 应用到包括本体分类、情感分析和文本分类在内的各种大规模数据集中，并表明它们不需要人类语言中关于词语、短语、句子或任何其他句法或语义结构的先验知识就可以达到让人惊艳的效果，模型适用于英文和中文。



## 34.深入了解VGG卷积神经网络滤波器

本节将使用 [keras-vis](https://raghakot.github.io/keras-vis/)，一个用于可视化 VGG 预建网络学到的不同滤波器的 Keras 软件包。基本思路是选择一个特定的 ImageNet 类别并理解 VGG16 网络如何来学习表示它。

第一步是选择 ImageNet 上的一个特定类别来训练 VGG16 网络。比如，将下图中的美国北斗鸟类别设定为 20：


![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091H33R54.jpg)
图 1 北斗鸟示意图


ImageNet 类别在网站 https://gist.github.com/yrevar/6135f1bd8dcf2e0cc683 中可以找到，作为一个 [Python](http://c.biancheng.net/python/) 字典包，ImageNet 的 1000 个类别 ID 被处理为人类可读的标签。

### 34.1具体过程

1. 导入 matplotlib 和 keras-vis 使用的模块。另外还需要载入预建的 VGG16 模块。Keras 可以轻松处理这个预建网络：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091H44W15.gif)
   
2. 通过使用 Keras 中包含的预构建图层获取 VGG16 网络，并使用 ImageNet 权重进行训练：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091H51T42.gif)
   
3. 这是 VGG16 网络的内部结构。许多卷积层与最大池化层交替。一个平坦（flatten）层连接着三个密集层。其中最后一层被称为预测层，这个图层应该能够检测高级特征，比如面部特征，在此例中，是鸟的形状。请注意，顶层显式的包含在网络中，因为希望可视化它学到的东西：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091H539195.gif)
   
4. 网络可以进一步抽象，如下图所示：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091H601544.gif)

   图 2 一个VGG16网络
   
5. 现在重点看一下最后的预测层是如何预测出 ID 类别序列为 20 的美国北斗鸟的：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091HAO40.gif)
   
6. 显示给定特征的特定图层的生成图像，并观察网络内部中的美国北斗鸟的概念：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091HH6320.gif)
   

   神经网络内部就是这样表示一只鸟的。这是一种虚幻的形象，但这正是在没有人为干预的情况下该神经网络自然学到的东西！

7. 如果你还好奇还想了解更多，那么，选择一个网络中更浅的层将其可视化，显示美国北斗鸟的前期训练过程：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091HKE20.gif)
   
8. 运行代码的输出如下：

   
   ![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091HQ3324.jpg)


正如预期的那样，这个特定的层学习低层的特征，如曲线。然而，卷积网络的真正威力在于，模型中的网络越深入，越能推断出更复杂的特征。

### 34.2解读分析

keras-vis 可视化密集层的关键思想是生成一个输入图像以最大化对应于鸟类的最终密集层输出。所以实际上这个模块做的是反转这个过程。给定一个特定的训练密集层与它的权重，生成一个新的最适合该层本身的合成图像。

每个卷积滤波器都使用类似的思路。在这种情况下，第一个卷积层是可以通过简单地将其权重可视化来解释的，因为它在原始像素上进行操作。

随后的卷积滤波器都对先前的卷积核的输出进行操作，因此直接对它们进行可视化并不一定非常容易理解。但是，如果独立考虑每一层，可以专注于生成最大化滤波器输出的合成输入图像。

GitHub 中的 [keras-vis 存储库](https://github.com/raghakot/keras-vis)提供了一系列关于如何检查内部网络的可视化示例，包括注意力显著图，其目标是在图像中包含各种类别（例如，草）时检测图像的哪个部分对特定类别（例如，老虎）的训练贡献最大。典型文章有“[Deep Inside Convolutional Networks：Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)”，其中一个 Git 库中的图片显示如下，这个案例说明了在网络中一个老虎的显著图样本：


![img](http://c.biancheng.net/uploads/allimg/190109/2-1Z1091I109555.jpg)



## 35.VGGNet、ResNet、Inception和Xception图像分类及对比

图像分类任务是一个典型的深度学习应用。人们对这个任务的兴趣得益于 [ImageNet](http://image-net.org/) 图像数据集根据 [WordNet](http://wordnet.princeton.edu/) 层次结构（目前仅有名词）组织，其中检索层次的每个节点包含了成千上万张图片。

更确切地说，ImageNet 旨在将图像分类并标注为近 22000 个独立的对象类别。在深度学习的背景下，ImageNet 一般是指论文“[ImageNet Large Scale Visual Recognition Challenge](http://www.image-net.org/challenges/LSVRC/)”中的工作，即 ImageNet 大型视觉识别竞赛，简称 ILSVRC。

在这种背景下，目标是训练一个模型，可以将输入图像分类为 1000 个独立的对象类别。本节将使用由超过 120 万幅训练图像、50000 幅验证图像和 100000 幅测试图像预训练出的模型。

### 35.1VGG16和VGG19

VGG16 和 VGG19 网络已经被引用到“[Very Deep Convolutional Networks for Large Scale Image Recognition](https://arxiv.org/abs/1409.1556)”（由 Karen Simonyan 和 Andrew Zisserman 于2014年编写）。该网络使用 3×3 卷积核的卷积层堆叠并交替最大池化层，有两个 4096 维的全连接层，然后是 softmax 分类器。16 和 19 分别代表网络中权重层的数量（即列 D 和 E）：


![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110103331932.jpg)
图 1 深层网络配置示例


在 2015 年，16 层或 19 层网络就可以认为是深度网络，但到了 2017 年，深度网络可达数百层。请注意，VGG 网络训练非常缓慢，并且由于深度和末端的全连接层，使得它们需要较大的权重存储空间。

### 35.2ResNet

ResNet（残差网络）的提出源自论文“[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)”（由 Kaiming He、XiangyuZhang、ShaoqingRen 和 JianSun 于 2015 年编写）。这个网络是非常深的，可以使用一个称为残差模块的标准的网络组件来组成更复杂的网络（可称为网络中的网络），使用标准的随机梯度下降法进行训练。

![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11010350H43.gif)

与 VGG 相比，ResNet 更深，但是由于使用全局平均池操作而不是全连接密集层，所以模型的尺寸更小。

### 35.3Inception

Inception 网络源自文章“[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)”（由Christian Szegedy、Vincent Vanhoucke、Sergey Ioffe、Jonathon Shlens和Zbigniew Wojna于2015年编写）。其主要思想是使用多个尺度的卷积核提取特征，并在同一模块中同时计算 1×1、3×3 和 5×5 卷积。然后将这些滤波器的输出沿通道维度堆叠并传递到网络中的下一层，如下图所示：


![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11010352YX.gif)


Inception-v3 见论文“Rethinking the Inception Architecture for Computer Vision”；Inception-v4 见论文“[Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)”（由 Christian Szegedy、Sergey Ioffe、Vincent Vanhoucke 和 Alex Alemi 于 2016 年编写）。

### 35.4Xception

Xception 网络是 Inception 网络的扩展，详见论文“[Xception：Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)”（由 Fran?ois Chollet 于 2016 年编写），网址为。Xception 使用了一种叫作深度可分离卷积运算的新概念，它可以在包含 3.5 亿个图像和 17000 个类的大型图像分类数据集上胜过 Inception-v3。由于 Xception 架构具有与 Inception-v3 相同的参数数量，因此性能提升不是由于容量的增加，而是由于更高效地使用了模型参数。

### 35.5图像分类准备工作

本节使用 Keras 因为这个框架有上述模块的预处理模块。Keras 在第一次使用时会自动下载每个网络的权重，并将这些权重存储在本地磁盘上。

换句话说，你不需要重新训练网络，而是使用互联网上已有的训练参数。假设你想在 1000 个预定义类别中分类网络，这样做是没问题的。下一节将介绍如何从这 1000 个类别开始，将其扩展到一个定制的集合，这个过程称为迁移学习。

### 35.6具体做法

1. 导入处理和显示图像所需的预建模型和附加模块：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110103R1641.gif)
   
2. 定义一个用于记忆训练中图像尺寸的映射，这些是每个模型的一些常量参数：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110103S9122.gif)
   
3. 定义用于加载和转换图像的辅助函数。请注意，预先训练的网络已经在张量上进行了训练，其形状还包括 batch_size 的附加维度。所以为了图像兼容性需要补充这个维度：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110103Z0G6.gif)
   
4. 定义一个辅助函数，用于对图像进行分类并对预测结果进行循环，显示前 5 名的预测概率：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110103921E3.gif)
   
5. 测试不同类型的预训练网络，你将看到一个带有各自类别预测概率的预测列表：

   | 测试欲训练网络                                           | 预测列标                                                     | 预测示例图                                                   |
   | -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | classify_image("image/parrot.jpg","vgg16")               | 1.macaw：99.92% 2.jacamar：0.03% 3.lorikeet：0.02% 4.bee_eater：0.02% 5.toucan：0.00% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11010404W63.gif) |
   | classify_image("image/parrot.jpg","vgg19")               | 1.macaw：99.77% 2.lorikeet：0.07% 3.toucan：0.06% 4.hornbill：0.05% 5.jacamar：0.01% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11010404W63.gif) |
   | classify_image("image/parrot.jpg","resnet")              | 1.macaw：97.93% 2.peacock：0.86% 3.lorikeet：0.23% 4.jacamar：0.12% 5.jay：0.12% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11010404W63.gif) |
   | classify_image("image/parrot_cropped1.jpg","resnet")     | 1.macaw：99.98% 2.lorikeet：0.00% 3.peacock：0.00% 4.sulphur-crested_cockatoo：0.00% 5.toucan：0.00% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11010420Q59.gif) |
   | classify_image("image/incredible-hulk-180.jpg","resnet") | 1.comic_book：99.76% 2.book_jacket：0.19% 3.jigsaw_puzzle：0.05% 4.menu：0.00% 5.packet：0.00% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11010422RU.gif) |
   | classify_image("image/croppoed_panda.jpg","resnet")      | 1.giant_panda：99.04% 2.indri：0.59% 3.lesser_panda：0.17% 4.gibbon：0.07% 5.titi：0.05% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110104300120.gif) |
   | classify_image("image/space-shuttle1.jpg","resnet")      | 1.space_shuttle：92.38% 2.triceratops：7.15% 3.warplane：0.11% 4.cowboy_hat：0.10% 5.sombrero：0.04% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11010431H30.gif) |
   | classify_image("image/space-shuttle2.jpg","resnet")      | 1.space_shuttle：99.96% 2.missile：0.03% 3.projectile：0.00% 4.steam_locomotive：0.00% 5.warplane：0.00% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z1101043361T.gif) |
   | classify_image("image/space-shuttle3.jpg","resnet")      | 1.space_shuttle：93.21% 2.missile：5.53% 3.projectile：1.26% 4.mosque：0.00% 5.beacon：0.00% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110104401445.gif) |
   | classify_image("image/space-shuttle4.jpg","resnet")      | 1.space_shuttle：49.61% 2.castle：8.17% 3.crane：6.46% 4.missile：4.62% 5.aircraft_carrier：4.24% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z1101044192U.gif) |

   注意可能报出一些错误，比如：

   | 测试欲训练网络                                 | 预测列标                                                     | 预测示例图                                                   |
   | ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | classify_image("image/parrot.jpg","inception") | 1.stopwatch：100.00% 2.mink：0.00% 3.hammer：0.00% 4.black_grouse：0.00% 5.web_site：0.00% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11010404W63.gif) |
   | classify_image("image/parrot.jpg","xception")  | 1.backpack：56.69% 2.military_uniform：29.79% 3.bib：8.02% 4.purse：2.14% 5.ping-pong_ball：1.52% | ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11010404W63.gif) |

6. 定义用于显示每个预建和预训练网络的内部架构的函数：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110104619524.gif)

### 35.7解读分析

我们已经使用了 Keras 应用，带有预训练权重的预训练 Keras 学习模型是可以获取的，这些模型可用于预测、特征提取以及参数微调。

在本例中，使用的是预测模型。将在下一个例子中看到如何使用该模型进行参数微调，以及如何在数据集上构建自定义的分类器，这些分类器在最初训练模型时是不可用的。

需要注意的是，Inception-v4 在 2017 年 7 月之前不能在 Keras 中直接使用，但可以在线上单独下载（https://github.com/kentsommer/keras-inceptionV4）。安装完成后，模块将在第一次使用时自动下载其权重参数。

AlexNet 是最早的堆叠深度网络之一，它只包含八层，前五层是卷积层，后面是全连接层。该网络于 2012 年提出，当年凭借其优异的性能获得冠军（其误差约为 16%，而亚军误差为 26%）。

最近对深度神经网络的研究主要集中在提高精度上。具有相同精度的前提下，轻量化 DNN 体系结构至少有以下三个优点：

1. 轻量化CNN在分布式训练期间需要更少的服务器通信。
2. 轻量化CNN需要较少的带宽将新模型从云端导出到模型所在的位置。
3. 轻量化CNN更易于部署在FPGA和其他有限内存的硬件上。


为了提供以上优点，论文“[SqueezeNet：AlexNet-level accuracy with 50x fewer parameters and<0.5MB model size](https://arxiv.org/abs/1602.07360)”（Forrest N.Iandola，Song Han，Matthew W.Moskewicz，Khalid Ashraf，William J.Dally，Kurt Keutzer，2016）提出的 SqueezeNet 在 ImageNet 上实现了 AlexNet 级别的准确性，参数少了 50 倍。

另外，由于使用模型压缩技术，可以将 SqueezeNet 压缩到小于 0.5 MB（比 AlexNet 小 510 倍）。Keras 实现的 SqueezeNet 作为一个单独的模块，已在网上开源(https://github.com/DT42/squeezenet_demo)。



## 36.预建深度学习提取特征及实现（详解版）

本节将介绍如何使用深度学习来提取相关的特征。

一个非常简单的想法是使用 VGG16 和一般的 DCNN 模型来进行特征提取。这段代码通过从特定图层中提取特征来实现这个想法。

### 36.1具体实现过程

1. 导入处理和显示图像所需的预建模型和附加模块：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110110520U1.gif)
   
2. 从网络中选择一个特定的图层，并获取输出的特征：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110110541T2.gif)
   
3. 提取给定图像的特征，代码如下所示：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110110FEW.gif)

### 36.2解读分析

现在，你可能想知道为什么要从 CNN 中的中间层提取特征。一个直觉是：随着网络的学习将图像分类成不同类别，每一层将学习到进行最终分类所必需的特征。

较低层识别诸如颜色和边缘等较低阶特征，高层将这些较低阶特征组合成较高阶特征，诸如形状或对象。因此，中间层具有从图像中提取重要特征的能力，这些特征有助于不同种类的分类。

这种结构具有以下几个优点：

1. 可以依靠公开的大规模数据集训练，将学习参数迁移到新的领域。
2. 可以节省大量训练时间。
3. 即使在自己的数据集中没有大量的训练数据，也可以提供合理的解决方案。我们也可以为手头任务准备一个较好的起始网络形状，而不用去猜测它。



## 37.TensorFlow实现InceptionV3详解

**迁移学习是一种非常强大的深度学习技术**，在不同的领域有着各种应用。迁移学习的思想很简单，可以用类比来解释。假设你想学习一种新的语言，比如西班牙语，那么从你已经知道的另一种语言，比如说英语开始学起，可能会有所帮助。

遵循这一思路，计算机视觉研究人员通常**使用预先训练的 CNN 为新任务生成表示**，其中新任务数据集可能不够大，无法从头开始训练整个 CNN。另一个常见的策略是**采用预先训练好的 ImageNet 网络**，然后对整个网络进行微调以完成新任务。

InceptionV3 网络是由 Google 开发的一个非常深的卷积网络。Keras 实现了完整的网络，如下图所示，它是在 ImageNet 上预先训练好的。这个模型的默认输入尺寸是 299×299，有三个通道。

![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110112HMb.gif)
图1 ImageNet v3 网路结构示意图


这个框架的例子受 Keras 网站上的在线模型(https://keras.io/applications/)启发。假设在一个域中有一个与 ImageNet 不同的训练数据集 D。D 具有 1024 个输入特征和 200 个输出类别。

### 37.1具体实现过程

1. 导入预处理模型和处理所需的库：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z1101129114N.gif)
    

2. 使用一个训练过的 Inception-v3 网络，但是不包括顶层模型，因为想要在 D 上进行微调。顶层是一个密集层，有 1024 个输入，最后一个输出层是一个 softmax 密集层，有 200 个输出类。

   
   x=GlobalAveragePooling2D()(x) 用于将输入转换为密集层处理的正确形状。实际上，base_model.output tensor 的形态有 dim_ordering="th"（对应样本...通道）或者 dim_ordering="tf"（对应样本，通道，行，列），但是密集层需要将其调整为（样本，通道）GlobalAveragePooling2D 按（行，列）平均。所以如果你看最后四层（include_top=True），你会看到这些形状：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110113004960.gif)
    

3. 如果包含 _top=False，则会删除最后三层并显示 mixed_10 层，因此 GlobalAveragePooling2D 层将（None...2048）转换为（None，2048），其中（None，2048）张量是（None...2048）张量中每个对应的（8，8）子张量的平均值：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110113052448.gif)
    

4. 所有卷积层都是预先训练好的，所以在整个模型的训练过程中冻结它们：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110113115430.gif)
    

5. 对模型进行编译并训练几批次，以便对顶层进行训练：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110113143145.gif)
    

6. 接下来冻结 Inception 中的顶层并微调 Inception 层。在这个例子中，冻结了前 172 层（一个用来微调的超参数）：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110113232302.gif)
    

7. 重新编译模型进行微调优化。需要重新编译模型以使这些修改生效：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11011324O96.gif)

### 37.2解读分析

现在我们有了一个新的深度网络，它重新使用了标准的 Inception-v3 网络，但是它通过迁移学习在一个新的领域 D 上进行了训练。

当然，有许多参数可以精确调整以达到较好的精度。但是，现在正在通过迁移学习重新使用一个非常大的预训练网络作为起点。这样做可以通过重新使用 Keras 中已有的功能来节省训练成本。

**知识扩展**

截至 2017 年，“计算机视觉”问题（在图像中找到模式的问题）被认为已经解决了，这个问题对生活有很大影响。例如：

论文“[Dermatologist-level classification of skin cancer with deep neural networks](https://www.nature.com/nature/journal/v542/n7639/full/nature21056.html)”（Andre Esteva，Brett Kuprel，Roberto A.Novoa，Justin Ko，Susan M.Swetter，Helen M.Blau & Sebastian Thrun，2017）使用由 2032 种不同疾病组成的 129450 张临床图像的数据集训练 CNN。他们通过 21 位经过认证的皮肤科医师对活检证实的临床图像进行二元分类，分别区分角质形成单元癌与良性脂溢性角化病、恶性黑色素瘤与良性痣。CNN 与人类专家在这两项任务上都达到了同样的水平，证明了人工智能在进行皮肤癌分类中能够与皮肤科医生相媲美。

论文“[High-Resolution Breast Cancer Screening with Multi-View Deep Convolutional Neural Networks](https://arxiv.org/abs/1703.07047)”（Krzysztof J.Geras，Stacey Wolfson，S.Gene Kim，LindaMoy，KyunghyunCho）提出一种有望提高乳腺癌筛查过程效率的新架构，可以处理四种标准的视图或角度。与常用的自然图像 DCN 架构（这种架构适用于 224×224 像素的图像）相比，MV-DCN 还能够使用 2600×2000 像素的分辨率。



## 38.TensorFlow WaveNet声音合成详解

WaveNet 是生成原始音频波形的深层生成模型。这项突破性的技术已经被 [Google DeepMind](https://deepmind.com/) 引入（https://deepmind.com/blog/generate-mode-raw-audio/），用于教授如何与计算机对话。结果确实令人惊讶，在网上你可以找到合成声音的例子，电脑学习如何用名人的声音与人们谈话。

所以，你可能想知道为什么学习合成音频是如此困难。听到的每个数字声音都是基于每秒 16000 个样本（有时是 48000 个或更多）建立一个预测模型，在这个模型中学习基于以前所有的样本来重现样本，这是一个非常困难的挑战。

尽管如此，有实验表明，WaveNet 已经改进了当前最先进的文本到语音（Text-To-Speech，TTS）系统，降低了英语和普通话之间 50% 的差异。

更酷的是，DeepMind 证明了 WaveNet 可以教会电脑如何产生乐器的声音，比如钢琴音乐。

下面给出一些定义。TTS 系统通常分为两个不同的类别：

- 连续TTS：其中单个语音片段首先被记忆，然后在语音再现时重新组合。这种方法没有大规模应用，因为它只能再现记忆过的语音片段，并且不可能在没有记忆片段的情况下再现新的声音或不同类型的音频。
- 参数TTS：其中创建模型用于存储要合成的音频的所有特征。在 WaveNet 之前，使用参数 TTS 生成的音频不如连续 TTS 自然。WaveNet 通过直接建模音频声音的生成来改进现有技术，而不是使用过去常用的中间信号去处理算法。


原则上，WaveNet 可以看作是一堆卷积层（已经在前面章节中看到了二维卷积图像），而且步长恒定，没有池化层。请注意，输入和输出的结构具有相同的尺寸，所以 ConvNet 非常适合对音频声音等连续数据进行建模。

然而，实验表明，为了达到输出神经元中的感受野大尺寸，有必要使用大量的大型滤波器或者不可避免地增加网络的深度。请记住，一个网络中一层神经元的感受野是前一层神经元对其提供输入的横截面。由于这个原因，纯粹的卷积网络在学习如何合成音频方面效率不高。

WaveNet 的关键在于所谓的扩张因果卷积（有时称为带孔卷积），这就意味着当应用卷积层的滤波器时，一些输入值被跳过。例如，在一个维度上，一个具有扩张 1、大小为 3 的滤波器 w 将计算如下所示的加权和。

简而言之，在扩张值为 D 的扩张卷积中，通常步长是 1，你也可使用其他的步长。下图给出了一个例子，扩大（孔）尺寸为 0，1，2：


![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110134T44b.gif)
图 1 扩张网络的一个例子


由于引入“孔”这个简单的想法，使得堆叠多个扩张的卷积层与指数增加的过滤器、学习长距离输入而不用担心有一个过深的网络成为可能。

因此，WaveNet 属于卷积网络，其卷积层具有各种扩张因子，使得感受野随深度呈指数增长，有效地覆盖了数千个音频时间步长。

当训练时，输入是来自人类说话者的录音。这些波形量化为一个固定的整数范围。WaveNet 定义了一个初始卷积层，只访问当前和之前的输入。然后，有一堆扩大的卷积层，仍然只能访问当前和之前的输入。最后，有一系列密集层结合了前面的结果，接下来是分类输出的 softmax 激活函数。

在每个步骤中，从网络预测一个值并将其反馈到输入中。同时，计算下一步的新预测。损失函数是当前步骤的输出和下一步的输入之间的交叉熵。

[NSynth](https://magenta.tensorflow.org/nsynth)是最近由 Google Brain 小组发布的一个 WaveNet 的演变，它不是因果关系，而是旨在看到输入块的整个上下文。如下图所示，神经网络确实是复杂的，但是作为介绍性讨论，知道网络学习如何通过使用基于减少编码/解码期间的误差的方法来再现其输入就足够了：

![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110134945327.gif)
图 2 NSynth架构的一个案例


本节直接使用网上的代码来演示(https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth)，你还可以从 [Google Brain](https://aiexperiments.withgoogle.com/sound-maker) 中找到一些。

感兴趣的读者也可以阅读论文“Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders”（Jesse Engel，Cinjon Resnick，Adam Roberts，Sander Dieleman，Douglas Eck，Karen Simonyan，Mohammad Norouzi，2017.4，https://arxiv.org/abs/1704.01279）。

### 38.1具体做法

1. 通过创建单独的 conda 环境来安装 NSynth。使用支持 Jupyter Notebook 的Python2.7 创建并激活 Magenta conda 环境：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z1101350535F.gif)
    

2. 安装用于读取音频格式的 Magenta pip 软件包和 librosa：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11013511N00.gif)
    

3. 从网上下载安装一个预先建立的模型（

   http://download.magenta.tensorflow.org/models/nsynth/wavenet-ckpt.tar

   ）并下载示例声音（

   https://www.freesound.org/people/MustardPlug/sounds/395058/

   ），然后运行 demo 目录中的笔记。第一部分包含了稍后将在计算中使用的模块：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11013521MH.gif)
    

4. 加载从互联网下载的演示声音，并将其放在与笔记本电脑相同的目录中。这将在约 2.5 秒内将 40000 个样品装入机器：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110135245V2.gif)
    

5. 使用从互联网上下载的预先训练的 NSynth 模型以非常紧凑的表示方式对音频样本进行编码。每 4 秒给一个 78×16 的尺寸编码，然后可以解码或重新合成。编码是张量(#files=1x78x16)：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z1101353214E.gif)
    

6. 保存稍后用于重新合成的编码。另外，用图形表示快速查看编码形状，并将其与原始音频信号进行比较。如你所见，编码遵循原始音频信号中的节拍：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11013534X60.gif)
    

7. 我们观察如下图所示的音频信号和 NSynth 编码：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110135412961.gif)
    

8. 现在对刚刚制作的编码进行解码。换句话说，如果重新合成的声音类似于原来的，这里试图以紧凑表示再现对原始音频的理解。事实上，如果你运行实验并听取原始音频和重新合成的音频，会感觉它们听起来非常相似：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110135431T8.gif)

### 38.2总结

WaveNet 是一种卷积网络，卷积层具有各种扩张因子，使得感受野随深度呈指数级增长，因此可以有效地覆盖数千个音频时间步长。NSynth 是 WaveNet 的一种演变，其中原始音频使用类似 WaveNet 的处理来编码，以学习紧凑的表示。然后，这个紧凑的表示被用来再现原始音频。

一旦学习如何通过扩张卷积创建一段紧凑的音频，可以发现其中的乐趣，比如说：

- 你会发现在互联网上很酷的演示。例如，可以看到模型如何学习不同乐器的声音(

  https://magenta.tensorflow.org/nsynth

  )：

  ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11013550WE.gif)

- 你可以看到一个环境中学习的模型如何在另一个环境中重新混合。例如，通过改变扬声器的身份，可以使用 WaveNet 用不同的声音描述同样的事情（https://deepmind.com/blog/wavenet-generative-model-raw-audio/）。

- 另一个非常有趣的实验是学习乐器的模型，然后重新混合，这样就可以创造出以前从未听过的新乐器。这真的很酷，它打开了一个新世界。

  例如，在这个例子中，把西塔琴和电吉他结合起来，形成一种很酷的新乐器。还不够酷？那么把低音贝斯和狗的叫声结合起来怎么样(

  https://aiexperiments.withgoogle.com/sound-maker/view/

  )？


  ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110135629554.gif)



## 39.TensorFlow实现视频分类的6种方法

对视频进行分类是一个活跃的研究领域，因为处理这种类型的问题需要大量的数据。内存需求经常达到现代GPU的极限，可能需要在多台机器上进行分布式的训练。

目前学者们正在探索复杂度不断增加的几个方向，来回顾一下：

1. 第一种方法是通过将视频的每一帧视为一幅单独的图像，利用二维 CNN 进行处理。这种方法将视频分类问题简化为图像分类问题。每帧视频图像都有类别输出，并且根据各帧输出的类别，选择频率最高的类别作为视频的分类结果。
2. 第二种方法是创建一个单一的网络，将二维 CNN 与一个 RNN 结合在一起。这个想法是，CNN 将考虑到图像分量，而 RNN 将考虑每个视频的序列信息。这种类型的网络可能非常难以训练，因为要优化的参数数量非常大。
3. 第三种方法是使用三维卷积网络，其中三维卷积网络是二维 CNN 的在 3D 张量（时间，图像宽度，图像高度）上运行的扩展。这种方法是图像分类的另一个自然延伸，但三维卷积网络可能很难训练。
4. 第四种方法基于智能方法的直觉。它们可以用于存储视频中每个帧的离线功能，而不是直接使用 CNN 进行分类。这个想法基于，特征提取可以非常有效地进行迁移学习，如前面章节所示。在提取所有的特征之后，可以将它们作为一组输入传递给RNN，其将在多个帧中学习序列并输出最终的分类。
5. 第五种方法是第四种方法的简单变体，其中最后一层是 MLP 而不是 RNN。在某些情况下，就计算需求而言，这种方法可以更简单并且成本更低。
6. 第六种方法也是第四种方法的变体，其中特征提取阶段采用三维 CNN 来提取空间和视觉特征，然后将这些特征传递给 RNN 或 MLP。


使用哪种方法取决于具体应用，并没有统一的答案。前三种方法通常计算量更大，而后三种方法计算成本更低，而且性能更好。

本节将展示如何利用论文“[Temporal Activity Detection in Untrimmed Videos with Recurrent Neural Networks](https://arxiv.org/abs/1608.08128)”（Montes，Alberto and Salvador，Amaia and Pascual，Santiago and Giro-i-Nieto，Xavier，2016）中的实验结果实现第六种方法。

这项工作旨在解决 ActivityNet 挑战赛中的问题（http://activity-net.org/challenges/2016/），重点是从用户生成的视频中识别高层次和目标导向的活动，类似于互联网门户中的活动。面临的挑战是如何在两个不同的任务中生成 200 个活动类别：

- 分类挑战：给定一个长视频，预测视频中的活动标签。
- 检测挑战：给定一个长视频，预测视频中活动的标签和时间范围。

![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11014133M23.gif)
图 1 C3D+RNN示例


提出的架构由两个阶段组成（如图 1 所示）：

1. 第一阶段将视频信息编码成小视频剪辑的单个矢量表示。为了达到这个目的，使用C3D网络。C3D网络使用3D卷积来从视频中提取时空特征，这些特征在前面已被分成16帧的剪辑。
2. 第二阶段，一旦提取到视频特征，就要对每个片段上的活动进行分类。为了执行这种分类，使用RNN。具体来说，使用LSTM网络，它尝试利用长期相关性，并且执行视频序列的预测。这是一个训练阶段：

### 39.1具体做法

本节简单总结了网站（https://github.com/imatge-upc/activitynet-2016-cvprw/blob/master/misc/nstep_by_step_guide.md）中的结果：

1. 从 git 库中克隆压缩包：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110141433G7.gif)
    

2. 下载 ActivityNet v1.3 数据集，大小为 600GB：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z1101415011a.gif)
    

3. 下载 CNN3d 和 RNN 的预训练权重：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11014152X56.gif)
    

4. 进行视频分类：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11014154T20.gif)

如果你对在自己的机器上训练 CNN3D 和 RNN 网络感兴趣，那么可以在互联网上找到本机训练需要使用的特定命令。目的是提供可用于视频分类的不同方法的高级视图。同样，不是仅有一种方法，而是有多种选择，应该根据具体需求选择最佳方案。

CNN-LSTM 体系结构是一个新的 RNN 层，输入变换和循环变换的输入都是卷积的。尽管命名非常相似，但 CNN-LSTM 层不同于 CNN 和 LSTM 的组合。

该模型在论文“[Convolutional LSTM Network：A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)”（Xingjian Shi，Zhourong Chen，Hao Wang，Dit-Yan Yeung，Wai-kin Wong，Wang-chun Woo，2015)中被提出。2017 年一些人开始使用此模块的视频进行实验，但这仍是一个非常活跃的研究领域。



## 40.RNN循环神经网络及原理（详解版）

循环神经网络（Recurrent Neural Network，RNN）很多实时情况都能通过时间序列模型来描述。

例如，如果你想写一个文档，单词的顺序很重要，当前的单词肯定取决于以前的单词。如果把注意力放在文字写作上...一个单词中的下一个字符取决于之前的字符（例如，The quick brown f...，下一个字母是 o 的概率很高），如下图所示。关键思想是在给定上下文的情况下产生下一个字符的分布，然后从分布中取样产生下一个候选字符：

 

![关于“The quick brown fox”句子的预测示例](http://c.biancheng.net/uploads/allimg/190110/2-1Z110151309492.gif)

图 1 关于“The quick brown fox”句子的预测示例


一个简单的变体是存储多个预测值，并创建一个预测扩展树，如下图所示：


![关于“The quick brown fox”句子的预测树示例](http://c.biancheng.net/uploads/allimg/190110/2-1Z110151333G3.gif)
图 2 关于“The quick brown fox”句子的预测树示例


基于序列的模型可以用在很多领域中。在音乐中，一首曲子的下一个音符肯定取决于前面的音符，而在视频领域，电影中的下一帧肯定与先前的帧有关。此外，在某些情况下，视频的当前帧、单词、字符或音符不仅仅取决于过去的信号，而且还取决于未来的信号。

基于时间序列的模型可以用RNN来描述，其中，时刻 i 输入为 Xi，输出为 Yi，时刻 [0，i-1] 区间的状态信息被反馈至网络。这种反馈过去状态的思想被循环描述出来，如下图所示：


![反馈的描述](http://c.biancheng.net/uploads/allimg/190110/2-1Z110151410441.gif)

图 3 反馈的描述


展开（unfolding）网络可以更清晰地表达循环关系，如下图所示：


![循环单元的展开](http://c.biancheng.net/uploads/allimg/190110/2-1Z110151440G4.gif)
图 4 循环单元的展开


最简单的 RNN 单元由简单的 tanh 函数组成，即双曲正切函数，如下图所示：

![简单的 tanh 单元](http://c.biancheng.net/uploads/allimg/190110/2-1Z110151502C0.gif)
图 5 简单的 tanh 单元

### 40.1梯度消失与梯度爆炸

由于存在两个稳定性问题，训练 RNN 是很困难的。由于反馈环路的缘故，梯度可以很快地发散到无穷大，或者迅速变为 0。如下图所示：


![梯度示例](http://c.biancheng.net/uploads/allimg/190110/2-1Z110151540U6.gif)
图 6 梯度示例


在这两种情况下，网络将停止学习任何有用的东西。梯度爆炸的问题可以通过一个简单的策略来解决，就是梯度裁剪。梯度消失的问题则难以解决，它涉及更复杂的 RNN 基本单元（例如长短时记忆（LSTM）网络或门控循环单元（GRU））的定义。先来讨论梯度爆炸和梯度裁剪：

梯度裁剪包括对梯度限定最大值，以使其不能无界增长。如下图所示，该方法提供了一个解决梯度爆炸问题的简单方案：


![梯度裁剪示例](http://c.biancheng.net/uploads/allimg/190110/2-1Z1101516395G.gif)
图 7 梯度裁剪示例


解决梯度消失需要一个更复杂的记忆模型，它可以有选择地忘记以前的状态，只记住真正重要的状态。如下图所示，将输入以概率 p∈[0，1] 写入记忆块 M，并乘以输入的权重。

以类似的方式，以概率 p∈[0，1] 读取输出，并乘以输出的权重。再用一个概率来决定要记住或忘记什么：


![记忆单元示例](http://c.biancheng.net/uploads/allimg/190110/2-1Z110151G3112.gif)
图 8 记忆单元示例

### 40.2长短时记忆网络（LSTM）

长短时记忆网络可以控制何时让输入进入神经元，何时记住之前时序中学到的东西，以及何时让输出传递到下一个时间戳。所有这些决策仅仅基于输入就能自我调整。

乍一看，LSTM 看起来很难理解，但事实并非如此。我们用下图来解释它是如何工作的：


![一个 LSTM 单元的示例](http://c.biancheng.net/uploads/allimg/190110/2-1Z110151J9502.gif)
图 9 一个 LSTM 单元的示例


首先，需要一个逻辑函数 σ 计算出介于 0 和 1 之间的值，并且控制哪个信息片段流经 LSTM 门。请记住，logisitic 函数是可微的，所以它允许反向传播。

然后需要一个运算符 ⊗ 对两个相同维数的矩阵进行点乘产生一个新矩阵，其中新矩阵的第 ij 个元素是两个原始矩阵第 ij 个元素的乘积。同样，需要一个运算符 ⊕ 将两个相同维数的矩阵相加，其中新矩阵的第 ij 个元素是两个原始矩阵第 ij 个元素的和。在这些基本模块中，将 i 时刻的输入 xi 与前一步的输出 yi 放在一起。

方程 fi=σ(Wf·[yi-1,xi]+bf) 是逻辑回归函数，通过控制激活门 ⊗ 决定前一个单元状态 Ci-1 中有多少信息应该传输给下一个单元状态 Ci（Wf 是权重矩阵，bf 是偏置）。逻辑输出 1 意味着完全保留先前单元状态 Ct-1，输出 0 代表完全忘记 Ci-1 ，输出（0，1）中的数值则代表要传递的信息量。

接着，方程![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110152030911.gif)根据当前输入产生新信息，方程 si=σ(Wc·[Yi-1，Xi]+bc) 则能控制有多少新信息![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110152144Y2.gif)通过运算符 ⊕ 被加入到单元状态 Ci 中。利用运算符 ⊗ 和 ⊕，给出公式![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110152215b6.gif)对单元状态进行更新。

最后，需要确定当前单元状态的哪些信息输出到 Yi。很简单，再次采用逻辑回归方程，通过 ⊗ 运算符控制候选值的哪一部分应该输出。在这里有一点需要注意，单元状态是通过 tanh 函数压缩到 [-1，1]。这部分对应的方程是 Yi=ti*tanh(Ci)。

这看起来像很多数学理论，但有两个好消息。首先，如果你明白想要达到的目标，那么数学部分就不是那么难；其次，你可以使用 LSTM 单元作为标准 RNN 元的黑盒替换，并立即解决梯度消失问题。因此你真的不需要知道所有的数学理论，你只需从库中取出 [TensorFlow](http://c.biancheng.net/tensorflow/) LSTM 并使用它。

### 40.3门控循环单元和窥孔LSTM

近年来已经提出了许多 LSTM 的变种模型，其中有两个很受欢迎：窥孔（peephole）LSTM 允许门层查看单元状态，如下图中虚线所示；而门控循环单元（GRU）将隐藏状态和单元状态合并为一个信息通道。

同样，GRU 和窥孔 LSTM 都可以用作标准 RNN 单元的黑盒插件，而不需要知道底层数学理论。这两种单元都可以用来解决梯度消失的问题，并用来构建深度神经网络。


![标准LTSM、窥孔LTSM、GRU示例](http://c.biancheng.net/uploads/allimg/190110/2-1Z1101523149C.gif)

图 10 标准LTSM、窥孔LTSM、GRU示例（[点此查看高清大图](http://c.biancheng.net/uploads/allimg/190110/2-1Z11015234a37.gif)）

### 40.4处理向量序列

真正使 RNN 强大的是它能够处理向量序列，其中 RNN 的输入和输出可以是序列，下图很好地说明了这一点，最左边的例子是一个传统（非递归）网络，后面跟着一个序列输出的 RNN，接着跟着一个序列输入的 RNN，其次跟着序列输入和序列输出不同步的 RNN，最后是序列输入和序列输出同步的 RNN。


![RNN序列示例](http://c.biancheng.net/uploads/allimg/190110/2-1Z1101524344O.gif)
图 11 RNN序列示例


机器翻译是输入序列和输出序列中不同步的一个例子：网络将输入文本作为一个序列读取，读完全文后输出目标语言。

视频分类是输入序列和输出序列同步的一个例子：视频输入是一系列帧，对于每一帧，在输出中提供分类标签。

如果你想知道更多关于 RNN 的有趣应用，则必读 Andrej Karpathy 的博客 http://karpathy.github.io/2015/05/21/rnn-effectiveness/，他训练网络写莎士比亚风格的散文（用 Karpathy 的话说：能勉强承认是莎士比亚的真实样品），写玄幻主题的维基百科文章，写愚蠢不切实际问题的定理证明（用 Karpathy 的话说：更多幻觉代数几何），并写出 Linux 代码片段（用 Karpathy 的话说：模型首先逐字列举了 GNU 许可字符，产生一些宏然后隐藏到代码中）。



## 41.神经机器翻译（seq2seq RNN）实现详解

seq2seq 是一类特殊的 RNN，在机器翻译、文本自动摘要和语音识别中有着成功的应用。本节中，我们将讨论如何实现神经机器翻译，得到类似于谷歌神经机器翻译系统得到的结果（https://research.googleblog.com/2016/09/a-neural-network-for-machine.html）。

关键是输入一个完整的文本序列，理解整个语义，然后输出翻译结果作为另一个序列。阅读整个序列的想法与以前的架构截然不同，在该架构中，一组固定词汇从一种源语言翻译成目标语言。

本节受到 Minh-Thang Luong 于 2016 年所写的博士论文“[Neural Machine Translation](https://github.com/lmthang/thesis/blob/master/thesis.pdf)”的启发。第一个关键概念是编码器–解码器架构，其中编码器将源语句转换为表示语义的向量，然后这个向量通过解码器产生翻译结果。

编码器和解码器都是 RNN，它们可以捕捉语言中的长距离依赖关系，例如性别一致性和语法结构，而不必事先知道它们，也不需要跨语言进行 1:1 映射。它能够流利地翻译并且具有强大的功能。


![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110160T3W3.gif)
图 1 编码器–解码器示例


来看一个 RNN 例子：将 She loves cute cats 翻译成 Elle aime les chats mignons。有两个 RNN：一个充当编码器，一个充当解码器。源语句 She loves cute cats 后面跟着一个分隔符“-”和目标语句 Elle aime les chats mignon。这两个关联语句被输入给编码器用于训练，并且解码器将产生目标语句 Elle aime les chats mignon。当然，需要大量类似例子来获得良好的训练。


![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110160931146.gif)
图 2 NMT的序列模型示例

NUM序列模型，一个深度循环结构示例，用于将源语句“She loves cute cats” 翻译成目标语句“Elle aimel les chats mignons”。解码器侧，前面时序中产生的单词作为输出下一个单词的输入，“_”代表语句的结束。

现在有一些可以使用的 RNN 变体，具体介绍其中的一些：

- RNN 可以是单向的或双向的，后者将捕捉双向的长时间依赖关系。
- RNN 可以有多个隐藏层，层数的选择对于优化来说至关重要...更深的网络可以学到更多知识，另一方面，训练需要花费很长时间而且可能会过度拟合。
- RNN 可以有多个隐藏层，层数的选择对于优化来说至关重要...更深的网络可以学到更多知识，另一方面，训练需要花费很长时间而且可能会过度拟合。
- RNN 可以具有嵌入层，其将单词映射到嵌入空间中，在嵌入空间中相似单词的映射恰好也非常接近。
- RNN 可以使用简单的重复性单元、LSTM、窥孔 LSTM 或者 GRU。


仍然考虑博士论文“[Neural Machine Translation](https://github.com/lmthang/thesis/blob/master/thesis.pdf)”，可以使用嵌入层将输入的句子映射到一个嵌入空间。然后，存在两个连接在一起的 RNN——源语言的编码器和目标语言的解码器。如下图所示，有多个隐藏层和两个流动方向：前馈垂直方向连接隐藏层，水平方向是将知识从上一步转移到下一步的循环部分。


![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110161030917.gif)
图 3 神经机器翻译示例


本节使用 NMT（Neural Machine Translation，神经机器翻译），这是一个在 [TensorFlow](http://c.biancheng.net/tensorflow/) 上在线可得的翻译演示包。

NMT 可通过https://github.com/tensorflow/nmt/ 获取，具体代码可通过 GitHub 获取。

### 41.1具体实现过程

1. 从 GitHub 克隆 NMT：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11016114E63.gif)
    

2. 下载一个训练数据集。在这个例子中，使用训练集将越南语翻译成英语，其他数据集可以在

   https://nlp.stanford.edu/projects/nmt/

   上获得，如德语和捷克语：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110161230X6.gif)
    

3. 参考

   https://github.com/tensorflow/nmt/

   ，这里将定义第一个嵌入层，嵌入层将输入、词汇量尺寸 V 和期望的输出尺寸嵌入到空间中。词汇量尺寸 V 中只有最频繁的单词才考虑被嵌入，所有其他单词则被打上 unknown 标签。在本例中，输入是 time-major，这意味着 max time 是第一个输入参数（

   https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn

   ）：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110161314417.gif)
    

4. 仍然参考

    

   https://github.com/tensorflow/nmt/

   ，这里定义一个简单的编码器，它使用 tf.nn.rnn_cell.BasicLSTMCell(num_units) 作为基本的 RNN 单元。虽然很简单，但要注意给定基本 RNN 单元，我们利用 tf.nn.dynamic_rnn 构建了 RNN 的（见

   https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn

   ）：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110161403209.gif)
    

5. 定义解码器。首先要有一个基本的 RNN 单元：tf.nn.rnn_cell.BasicLSTMCell，以此来创建一个基本的采样解码器 tf.contrib.seq2seq.BasicDecoder，将结果输入到解码器 tf.contrib.seq2seq.dynamic_decode 中进行动态解码。

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11016143cZ.gif)
    

6. 网络的最后一个阶段是 softmax dense 阶段，将最高隐藏状态转换为 logit 向量：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z1101615052W.gif)
    

7. 定义在训练阶段使用的交叉熵函数和损失：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11016152S16.gif)
    

8. 定义反向传播所需的步骤，并使用适当的优化器（本例中使用 Adam）。请注意，梯度已被剪裁，Adam 使用预定义的学习率：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11016160K51.gif)
    

9. 运行代码并理解不同的执行步骤。首先，创建训练图，然后开始迭代训练。评价指标是 BLEU（bilingual evaluation understudy），这个指标是评估将一种自然语言机器翻译为另一种自然语言的文本质量的标准，质量被认为是算法的结果和人工操作结果的一致性。正如你所看到的，指标值随着时间而增长：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110161IUQ.gif)

### 41.2解读分析

所有上述代码已经在 https://github.com/tensorflow/nmt/blob/master/nmt/model.py 上给出。关键是将两个 RNN 打包在一起，第一个是嵌入空间的编码器，将相似的单词映射得很接近，编码器理解训练样例的语义，并产生一个张量作为输出。然后通过将编码器的最后一个隐藏层连接到解码器的初始层可以简单地将该张量传递给解码器。

请注意，学习能够进行是因为损失函数基于交叉熵，且labels=decoder_outputs。

如下图所示，代码学习如何翻译，并通过BLEU指标的迭代跟踪进度：

![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110161QN57.gif)
图 4


下面我们将源语言翻译成目标语言。这个想法非常简单：一个源语句作为两个组合的 RNN（编码器+解码器）的输入。一旦句子结束，解码器将发出 logit 值，采用贪婪策略输出与最大值相关的单词。

例如，单词 moi 作为来自解码器的第一个标记被输出，因为这个单词具有最大的 logit 值。此后，单词 suis 输出，等等：

![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z11016192J09.gif)

图 5 带概率分布的NMT序列模型示例


解码器的输出有多种策略：

1. 贪婪：输出对应最大logit值的单词。
2. 采样：通过对众多logit值采样输出单词。
3. 集束搜索：有多个预测，因此创建一个可能结果的扩展树。

### 41.3翻译实现过程

1. 制定解码器采样的贪婪策略。这很简单，因为可以使用 tf.contrib.seq2seq.GreedyEmbeddingHelper 中定义的库，由于不知道目标句子的准确长度，因此这里使用启发式方法将其限制为源语句长度的两倍。

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z1101620362X.gif)
    

2. 现在可以运行网络，输入一个从未见过的语句（inference_input_file=/tmp/my_infer_file），并让网络转换结果（inference_output_file=/tmp/nmt_model/output_infer）：

   
   ![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110162133292.gif)


两个 RNN 封装在一起形成编码器–解码器 RNN 网络。解码器发出的 logits 被贪婪策略转换成目标语言的单词。作为一个例子，下面显示了一个从越南语到英语的自动翻译的例子：

![img](http://c.biancheng.net/uploads/allimg/190110/2-1Z110162155A0.gif)



## 42.注意力机制(基于seq2seq RNN)详解