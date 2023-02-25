# **基于原生C++的通用矩阵——Numcpp**

原生的C++矩阵类封装，基本使用：

```c++
#include <iostream>
#include "Numcpp.hpp"

int main(int argc, char const *argv[])
{
    /*使用Numcpp<type> np(row,col)创建一个row * col的矩阵*/
    Numcpp<int> nc(4, 4), m(4, 4);
	/*矩阵中所有元素的默认值为1*/
    nc *= 2;
    m *= 3;
    try
    {
        //矩阵乘法：
        Numcpp<int> result = nc * m;
        std::cout << result << std::endl;
        //矩阵运算：
        result = nc + m;
        std::cout << result << std::endl;
        //哈达马乘积：
        result = nc.Hadamard(m);
        std::cout << result << std::endl;
        //矩阵转置：
        result = nc.transpose();
        std::cout << result << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}
```

# **矩阵的运算**

### 下面是支持的操纵符的类型：

```c++
Numcpp<typename> nc(4, 4),matrix(4,4);
```

### 赋值运算符：

```c++
//矩阵简单运算的赋值运算
nc += matrix;
nc -= matrix;
nc = matrix;
//矩阵数乘广播的赋值运算
nc *= number//number为一个数值
```

### 矩阵的基本运算

```c++
Numcpp<typename> result = nc + matrix;
Numcpp<typename> result = nc - matrix;

Numcpp<typename> result = nc * matrix;//数乘
Numcpp<typename> result = nc * matrix;//矩阵乘法
```

### 解矩阵运算（下标访问）

```c++
typename data = nc[x][y];
```

### 矩阵的哈达马乘积（Hadamard product）

```c++
//哈达马乘积的基本运算
Numcpp<typename> result = nc.Hadamard(matrix);
//哈达马乘积的赋值运算
nc.Hadamard_self(matrix);
```

### 矩阵的置转运算

```c++
//获取矩阵的转置
Numcpp<typename> result = nc.transpose();
//转置矩阵
nc.transposed();
```

# 相关信息

作者：[yauntyour]([yauntyour (yauntyour) · GitHub](https://github.com/yauntyour/))

授权协议：MIT开源协议

参考：[数学基础 - 矩阵的基本运算（Matrix Operations）_沙沙的兔子的博客-CSDN博客_矩阵运算](https://blog.csdn.net/darkrabbit/article/details/80025935)