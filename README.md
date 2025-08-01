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
# 克隆项目
git clone https://github.com/pengzj1/eddpm.git
cd eddpm

# 安装依赖
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





### unet设置：
参考gradio_demo.py里对iclight模型的调用，将数据集图像处理为8通道输入
# 通道 0-3: 图像通过vae的潜在表示
# 通道 4-7: 利用briarmbg移除背景后的前景通过vae潜在表示


#### 算法1：基于概率掩码的高效去噪扩散（EDDPM）  

**输入**：随机初始化的扩散模型$F_\theta$，训练轮数$N$，最终掩码率$\gamma_f$。  
**初始化**：各时间步采样概率$s = 1 \in \mathbb{R}^T$（初始全激活）  

1. 对于轮次$e = 1, 2, ..., N$：  

   2. 根据式(10)计算当前掩码率$\gamma_e$。  

      #### 【掩码率的渐进增加】  

      为控制模型复杂度，定义最终掩码率$\gamma_f$（即$K = \gamma_f T$）。为稳定训练，掩码率从1（全步骤）渐进过渡到$\gamma_f$，采用（Zhu & Gupta, 2017）的增长函数：  
      $$
      \gamma_e = 
      \begin{cases} 
      1, & \text{若 } e < e_1, \\
      \gamma_f + (1 - \gamma_f) \left(1 - \frac{e - e_1}{N - e_1}\right)^3, & \text{否则},
      \end{cases} \quad (10)
      $$
      其中$e_1$为初始全步骤训练轮数，$\gamma_e$为当前轮次$e$的剩余步骤比例。  

      #### 

   3. 对于每个训练迭代：  

      4. 采样数据 mini-batch $X_B$。  

      5. 基于分数$s$进行伯努利采样，生成扩散步骤掩码。  

      6. 根据式(6)，基于采样掩码更新方差调度。  

         $x_t \sim \mathcal{N}\left(\sqrt{\alpha_t(m)}x_0,\ (1-\alpha_t(m))\mathbf{I}\right)$  
         其中$\alpha_t(m) = \prod_{i=1}^t \hat{\alpha}_i(m)$，且$\hat{\alpha}_t(m) = 1 - \beta_t m_t$。(6)  

      7. 随机采样未掩码的扩散步骤用于训练。  

      8. 计算扩散模型损失$\mathcal{L}_\theta^t$。  

         其中损失函数$\mathcal{L}_\theta^t(x_0, \epsilon, m)$的形式为：  
         $\mathcal{L}_\theta^t(x_0, \epsilon, m) = C_t \left\|\epsilon - \epsilon_\theta\left(\sqrt{\alpha_t(m)}x_0 + \sqrt{1 - \alpha_t(m)}\epsilon,\ t\right)\right\|^2$  

      9. 对$F_\theta$反向传播，估计$\nabla_\theta \Phi(\theta, s)$。 

         $\nabla_\theta \Phi(\theta, s) = \mathbb{E}_{m \sim p(m|s)} \mathbb{E}_{x_0,\epsilon,t|m} \nabla_\theta \mathcal{L}_\theta^t(x_0, \epsilon, m) \quad (8)$   

      10. 根据式(9)估计$\nabla_s \Phi(\theta, s)$。  

          采用策略梯度法估计梯度：  
          $$
          \begin{aligned}
          \nabla_s \Phi(\theta, s) &= \nabla_s \sum_{m} \left[ \mathbb{E}_{x_0,\epsilon,t|m} \mathcal{L}_\theta^t(x_0, \epsilon, m) \right] p(m|s) \\
          &= \sum_{m} \left[ \mathbb{E}_{x_0,\epsilon,t|m} \mathcal{L}_\theta^t(x_0, \epsilon, m) \right] \nabla_s p(m|s) \\
          &= \sum_{m} \mathbb{E}_{x_0,\epsilon,t|m} \mathcal{L}_\theta^t(x_0, \epsilon, m) \nabla_s \ln p(m|s) \cdot p(m|s) \\
          &= \mathbb{E}_{m \sim p(m|s)} \mathbb{E}_{x_0,\epsilon,t|m} \mathcal{L}_\theta^t(x_0, \epsilon, m) \nabla_s \ln p(m|s). \quad (9)
          \end{aligned}
          $$
          因此，$\mathcal{L}_\theta^t(x_0, \epsilon, m) \nabla_s \ln p(m|s)$是$\Phi(\theta, s)$的随机梯度。  

      11. 根据式(11)更新$\theta$和$s$。  

          通过投影梯度下降更新$\theta$和$s$：  
          $\theta = \theta - \eta \nabla_\theta \Phi(\theta, s), \quad s = \text{proj}_S\left(s - \eta \nabla_s \Phi(\theta, s)\right), \quad (11)$  
          其中$S = \{s \in \mathbb{R}^T : \|s\|_1 \leq K_e,\ s \in [0,1]^T\}$（$K_e = \gamma_e T$）

          投影计算细节

          给定向量 $z$，其在约束区域 $\{s \in \mathbb{R}^T：\|s\|_1 \leq Ke，s \in [0, 1]^T\}$ 上的投影 $s$ 可按如下方式计算：

          $s = \min(1, \max(0, z - v_2^*1))$。

          其中 $v_2^* = \max(0, v_1^*)$，且 $v_1^*$ 是以下方程的解：

          $1^\top[\min(1, \max(0, z - v_1^*1))] - Ke = 0$。(12)

          方程（12）可使用二分法高效求解。

          

      12.结束迭代  

      13.结束训练  
