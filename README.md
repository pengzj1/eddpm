# eddpm

# EDDPM Training for IC-Light


- **高效时间步采样**: 通过学习的掩码策略跳过冗余的时间步，显著提高训练效率
- **8通道潜在空间**: 结合背景和前景信息的8通道潜在表示
- **自动背景移除**: 集成BriaRMBG模型进行高质量前景提取
- **IC-Light集成**: 完全兼容IC-Light预训练权重
- **EMA模型**: 指数移动平均确保训练稳定性
- **策略梯度优化**: 基于策略梯度的掩码参数学习



## 🏗️ 项目结构

```
eddpm/
├── train.py           # 主训练脚本
├── briarmbg.py        # BriaRMBG背景移除模型（原iclight内容不用做修改）
├── models/            # 模型权重目录
│   └── iclight_sd15_fc.safetensors
├── data/              # 数据集目录
├── db_examples.py     # gradio_demo所需的前端图像参数  （原iclight内容不用做修改）
├── eddpm_scheduler.py # 依据ddpm scheduler加入掩码s逻辑的eddpm scheduler，目前s初始化为1功能与ddpm相同可以正常生成图像
├── gradio_demo.py     # 原iclight内容将scheduler改为eddpm scheduler的演示demo
└── README.md          # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
git clone https://github.com/pengzj1/eddpm.git
cd eddpm
conda create -n iclight python=3.10
conda activate iclight
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 模型准备

确保以下模型文件在正确位置：

```bash
# IC-Light权重文件
./models/iclight_sd15_fc.safetensors

# BriaRMBG模型会自动下载
```

### 3. 开始训练

```bash
python train.py
```





## unet设置：
参考gradio_demo.py里对iclight模型的调用，将数据集图像处理为8通道输入
通道 0-3: 图像通过vae的潜在表示
通道 4-7: 利用briarmbg移除背景后的前景通过vae潜在表示



          

      12.结束迭代  

      13.结束训练  
