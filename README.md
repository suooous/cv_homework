# cv_homework
**存储计算机视觉的作业的程序**

## 环境准备
* 使用conda创建虚拟环境进行实验(cvv替换成你自己的命名，可以选择自己所需的python版本)
```bash
conda create --name cvv python=3.9
```
* 激活虚拟环境
```bash
conda activate cvv
```
* 安装openCV
```bash
pip install opencv-python
```
* 其余所需的库，如matplolib等，按照所需使用pip安装

## 第一次实验
* 拍一张A4纸文稿的图片，利用角点检测、边缘检测等，再通过投影变换完成对文档的对齐扫描
* 运行程序
```bash
python project.py
```
* 运行之后将在同一个文件夹之下，生成对应处理过程的图像处理结果
