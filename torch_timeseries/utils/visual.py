from typing import Any
import matplotlib.pyplot as plt

def mvts_fig(data:Any, num_nodes:int,start:int , end:int ):
    # 创建图形和子图
    fig, axs = plt.subplots(num_nodes,figsize=(80, 6*num_nodes))
    for i in range(num_nodes):
        # 绘制时序数据
        axs[i].plot(data[start:end,i])
    return fig