# 创建并切换jupyter kernel

如果在Conda环境中创建了Python环境，并且该环境在执行conda info -e时可见，但在Jupyter Notebook中找不到对应的kernel，这通常是因为该环境没有被添加为Jupyter的一个可用kernel。要解决这个问题，你需要确保已经安装了ipykernel包，并且为你的环境创建了一个Jupyter kernel。

以下是步骤：

```Bash
#激活你的Conda环境：打开命令行或终端，激活你的Conda环境。假设你的环境名为myenv，你可以使用以下命令来激活环境：
conda activate myenv

#安装ipykernel：在激活的环境中安装ipykernel包，这个包允许你将环境作为一个kernel添加到Jupyter中。
conda install ipykernel
#或者使用pip（如果conda命令不起作用）：
pip install ipykernel

#为环境添加Jupyter Kernel：安装ipykernel之后，使用以下命令为你的环境创建一个新的Jupyter kernel：
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
#这里，--name后面跟的是你的环境名，--display-name后面跟的是在Jupyter Notebook中显示的名称。

#检查Jupyter Notebook中的kernel：
#完成以上步骤后，重新启动Jupyter Notebook，然后检查kernel列表，你应该能够看到新添加的环境。
```

