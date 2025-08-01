import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import functional as F
import logging
from PIL import Image
import cv2
import copy

# 导入您的U-Net配置
import safetensors.torch as sf
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

# 导入BriaRMBG模型
from briarmbg import BriaRMBG

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resize_without_crop(image, target_width, target_height):
    """从gradio_demo1.py复制的resize函数"""
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

def numpy2pytorch(imgs):
    """从gradio_demo1.py复制的转换函数"""
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h

def pytorch2numpy(imgs, quant=True):
    """从gradio_demo1.py复制的转换函数"""
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results

@torch.inference_mode()
def run_rmbg(img, rmbg_model, device, sigma=0.0):
    """从gradio_demo1.py复制的背景移除函数"""
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg_model(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha

class ImageProcessor:
    def __init__(self, vae, device):
        self.vae = vae
        self.device = device
        
        # 加载BriaRMBG模型用于前景提取
        self.rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
        self.rmbg = self.rmbg.to(device=device, dtype=torch.float32)
        self.rmbg.eval()
    
    def remove_background(self, image):
        """使用BriaRMBG移除背景 - 使用gradio_demo1.py的方法"""
        # 将tensor转换为numpy图像
        if isinstance(image, torch.Tensor):
            # 从[-1,1]范围转换到[0,255]
            image_np = pytorch2numpy([image], quant=True)[0]
        else:
            image_np = image
        
        # 使用run_rmbg函数移除背景
        fg_image, alpha = run_rmbg(image_np, self.rmbg, self.device)
        
        # 将结果转换回tensor格式
        fg_tensor = numpy2pytorch([fg_image])[0]  # 转换为[-1,1]范围的tensor
        
        return fg_tensor
    
    def encode_image_to_latent(self, image):
        """将图像编码为潜在表示"""
        # 确保图像在正确的设备和数据类型上
        image = image.to(self.device, dtype=self.vae.dtype)
        
        # 如果是单张图像，添加batch维度
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 编码为潜在表示
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        
        return latent
    
    def process_batch_to_8channel(self, rgb_batch):
        """将RGB批次转换为8通道潜在表示"""
        batch_size = rgb_batch.shape[0]
        device = rgb_batch.device
        
        # 存储结果的列表
        bg_latents = []
        fg_latents = []
        
        for i in range(batch_size):
            rgb_image = rgb_batch[i]  # [3, H, W]
            
            # 1. 编码原始RGB图像（背景）
            bg_latent = self.encode_image_to_latent(rgb_image)
            bg_latents.append(bg_latent)
            
            # 2. 提取前景并编码
            fg_image = self.remove_background(rgb_image)
            fg_latent = self.encode_image_to_latent(fg_image)
            fg_latents.append(fg_latent)
        
        # 拼接所有结果
        bg_latents = torch.cat(bg_latents, dim=0)  # [batch_size, 4, H, W]
        fg_latents = torch.cat(fg_latents, dim=0)  # [batch_size, 4, H, W]
        
        # 拼接为8通道
        combined_latents = torch.cat([bg_latents, fg_latents], dim=1)  # [batch_size, 8, H, W]
        
        return combined_latents

class EDDPMTrainer:
    def __init__(self, 
                 unet,
                 text_encoder,
                 vae,
                 tokenizer,
                 image_processor,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 final_mask_ratio=0.5,
                 warmup_epochs=100,  # 预热阶段轮数
                 device='cuda'):
        
        self.device = device
        self.unet = unet
        self.text_encoder = text_encoder
        self.vae = vae
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_timesteps = num_timesteps
        self.final_mask_ratio = final_mask_ratio
        self.warmup_epochs = warmup_epochs
        
        # 初始化噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 初始化采样概率 s (所有时间步初始为1)
        self.s = torch.ones(num_timesteps, requires_grad=True, device=device)
        
        # 优化器
        self.unet_optimizer = optim.Adam(self.unet.parameters(), lr=0.0001)
        self.s_optimizer = optim.Adam([self.s], lr=0.001)
        
        # EMA
        self.ema_decay = 0.9999
        self.ema_unet = self._create_ema_model()
        
    def _create_ema_model(self):
        """创建EMA模型"""
        # 不要从配置重新创建，而是深拷贝当前的unet
        ema_unet = copy.deepcopy(self.unet)
        ema_unet.eval()
        for param in ema_unet.parameters():
            param.requires_grad = False
        return ema_unet
    
    def update_ema(self):
        """更新EMA模型参数"""
        for ema_param, param in zip(self.ema_unet.parameters(), self.unet.parameters()):
            ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * param.data
    
    def compute_mask_ratio(self, epoch, total_epochs):
        """计算当前epoch的掩码率 (公式10)"""
        if epoch < self.warmup_epochs:
            return 1.0
        else:
            progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
            gamma_e = self.final_mask_ratio + (1 - self.final_mask_ratio) * (1 - progress) ** 3
            return gamma_e
    
    def sample_mask(self, batch_size):
        """基于采样概率s进行伯努利采样生成掩码
        返回: mask，其中mask[t]=1表示保留时间步t，mask[t]=0表示跳过时间步t
        """
        # 将s转换为概率
        probs = torch.sigmoid(self.s)
        # 伯努利采样
        mask = torch.bernoulli(probs.unsqueeze(0).expand(batch_size, -1))
        return mask.bool()
    
    def sample_unmasked_timesteps(self, mask):
        """从未掩码的时间步中随机采样用于训练
        mask: [batch_size, num_timesteps], mask[t]=1表示保留，mask[t]=0表示跳过
        返回: 每个样本对应的采样时间步 [batch_size]
        """
        batch_size = mask.shape[0]
        t_list = []
        
        for b in range(batch_size):
            current_mask = mask[b]  # [T]
            # 找到未掩码的时间步 (mask[t] == 1 表示保留/未掩码)
            unmasked_steps = torch.where(current_mask == 1)[0]
            
            if len(unmasked_steps) > 0:
                # 从未掩码步骤中随机选择一个
                selected_idx = torch.randint(0, len(unmasked_steps), (1,))
                t_val = unmasked_steps[selected_idx]
            else:
                # 如果没有未掩码步骤，随机选择一个（降级处理）
                t_val = torch.randint(0, self.num_timesteps, (1,))
                logger.warning(f"No unmasked steps for sample {b}, falling back to random sampling")
            
            t_list.append(t_val)
        
        t = torch.stack(t_list).squeeze().to(self.device)
        return t
    
    def compute_alpha_t_with_mask(self, t, mask):
        """根据掩码计算修正的alpha_t (公式6)
        mask: [batch_size, num_timesteps], mask[t]=1表示保留，mask[t]=0表示跳过
        
        公式6: α̂ₜ(m) = 1 - βₜmₜ
        - 当mₜ=1（保留）时: α̂ₜ = 1 - βₜ（正常alpha值）
        - 当mₜ=0（跳过）时: α̂ₜ = 1（完全跳过，无噪声）
        """
        batch_size = mask.shape[0]
        alpha_t_list = []
        
        for b in range(batch_size):
            current_mask = mask[b]  # [T] mask=1表示保留，mask=0表示跳过
            t_val = int(t[b].item())  # 确保转换为整数
            
            # 直接使用mask，无需转换语义
            # 公式6: α̂ₜ(m) = 1 - βₜmₜ
            # 保留的步骤(mask=1): α̂ₜ = 1 - βₜ（正常值）
            # 跳过的步骤(mask=0): α̂ₜ = 1（无噪声）
            modified_alphas = 1.0 - self.betas * current_mask.float()
            
            # 计算累积乘积直到时间步t: αₜ(m) = ∏ᵢ₌₁ᵗ α̂ᵢ(m)
            alpha_t = torch.prod(modified_alphas[:t_val+1])
            alpha_t_list.append(alpha_t)
        
        result = torch.stack(alpha_t_list).to(self.device)
        # 确保返回的数据类型与输入数据一致
        return result.to(dtype=self.unet.dtype)
    
    def project_s(self, s_grad, K_e):
        """投影操作 - 将s投影到约束区域"""
        # 更新s
        s_new = self.s - 0.001 * s_grad
        
        # 投影到[0,1]
        s_new = torch.clamp(s_new, 0.0, 1.0)
        
        # 投影到L1约束
        if s_new.sum() > K_e:
            # 使用二分法求解投影
            def objective(v):
                projected = torch.clamp(s_new - v, 0.0, 1.0)
                return projected.sum() - K_e
            
            # 二分法
            v_low, v_high = -1.0, 1.0
            for _ in range(50):  # 最多50次迭代
                v_mid = (v_low + v_high) / 2
                if objective(v_mid) > 0:
                    v_low = v_mid
                else:
                    v_high = v_mid
            
            s_new = torch.clamp(s_new - v_mid, 0.0, 1.0)
        
        return s_new
    
    def compute_loss(self, rgb_batch, mask, prompt_embeds=None):
        """计算扩散损失
        只对未掩码的时间步计算损失
        """
        batch_size = rgb_batch.shape[0]
        
        # 将RGB图像转换为8通道潜在表示
        x0 = self.image_processor.process_batch_to_8channel(rgb_batch)
        
        # 步骤4: 随机采样未掩码的扩散步骤用于训练
        t = self.sample_unmasked_timesteps(mask)
        
        # 生成噪声
        noise = torch.randn_like(x0)
        
        # 计算修正的alpha_t (传入整数时间步)
        alpha_t = self.compute_alpha_t_with_mask(t, mask)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        
        # 添加噪声 - 确保数据类型一致
        x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        x_t = x_t.to(dtype=self.unet.dtype)
        
        # 确保时间步的数据类型与U-Net一致 (转换为模型需要的类型)
        t_model = t.to(dtype=self.unet.dtype)
        
        # 预测噪声
        if prompt_embeds is None:
            # 使用空的prompt - 确保数据类型一致
            prompt_embeds = torch.zeros(batch_size, 77, 768, 
                                    device=self.device, 
                                    dtype=self.unet.dtype)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype)
        
        predicted_noise = self.unet(x_t, t_model, encoder_hidden_states=prompt_embeds).sample

        # 目标噪声是前4个通道（原始的噪声）
        target_noise = noise[:, :4, :, :]  # 只取前4个通道作为目标
        
        # 计算损失 - 转换回float32进行损失计算以避免精度问题
        loss = F.mse_loss(predicted_noise.float(), target_noise.float(), reduction='none')
        loss = loss.mean(dim=[1, 2, 3])  # 按样本求平均
        
        return loss, t  # 同时返回采样的时间步
    
    def train_step(self, batch, mask, epoch):
        """单步训练"""
        rgb_images, _ = batch
        rgb_images = rgb_images.to(self.device)
        batch_size = rgb_images.shape[0]
        
        # 计算损失
        losses, sampled_t = self.compute_loss(rgb_images, mask)
        unet_loss = losses.mean()
        
        # 更新U-Net参数
        self.unet_optimizer.zero_grad()
        unet_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.unet_optimizer.step()
        
        # 计算策略梯度更新s (公式9)
        with torch.no_grad():
            # 计算log概率的梯度
            probs = torch.sigmoid(self.s)
            log_prob_grad = torch.zeros_like(self.s)
            
            for b in range(batch_size):
                current_mask = mask[b]
                current_loss = losses[b]
                
                # 计算log p(m|s)的梯度
                # 对于mask=1的时间步: ∂log p(m|s)/∂s = (1-p(s))
                # 对于mask=0的时间步: ∂log p(m|s)/∂s = -p(s)
                for t in range(self.num_timesteps):
                    if current_mask[t]:  # mask=1, 保留该时间步
                        log_prob_grad[t] += current_loss * (1 - probs[t])
                    else:  # mask=0, 跳过该时间步
                        log_prob_grad[t] += current_loss * (-probs[t])
            
            log_prob_grad /= batch_size
        
        # 更新s
        current_mask_ratio = self.compute_mask_ratio(epoch, 500000 // (50000 // 128))  # 总epoch数
        K_e = current_mask_ratio * self.num_timesteps
        
        self.s.data = self.project_s(log_prob_grad, K_e)
        
        # 更新EMA
        self.update_ema()
        
        return unet_loss.item()
    
    def compute_fid(self, real_images, generated_images):
        """计算FID分数"""
        def calculate_activation_statistics(images):
            # 简化的FID计算，实际应用中需要使用预训练的Inception网络
            images = images.view(images.shape[0], -1)
            mu = torch.mean(images, dim=0)
            sigma = torch.cov(images.T)
            return mu, sigma
        
        mu1, sigma1 = calculate_activation_statistics(real_images)
        mu2, sigma2 = calculate_activation_statistics(generated_images)
        
        # 计算FID
        diff = mu1 - mu2
        covmean = torch.sqrt(torch.mm(sigma1, sigma2))
        fid = torch.sum(diff ** 2) + torch.trace(sigma1 + sigma2 - 2 * covmean)
        
        return fid.item()

def main():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载您的U-Net配置
    sd15_name = 'stablediffusionapi/realistic-vision-v51'
    tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
    
    # 修改U-Net输入通道
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, 
                                     unet.conv_in.kernel_size, 
                                     unet.conv_in.stride, 
                                     unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in
    
    # 加载IC-Light权重
    model_path = './models/iclight_sd15_fc.safetensors'
    if os.path.exists(model_path):
        sd_offset = sf.load_file(model_path)
        sd_origin = unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged
    
    # 移动到设备
    text_encoder = text_encoder.to(device=device, dtype=torch.float16)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    unet = unet.to(device=device, dtype=torch.float16)
    
    # 设置注意力处理器
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())
    
    # 初始化图像处理器
    image_processor = ImageProcessor(vae, device)
    
    # 准备CIFAR-10数据集 - 调整到更高分辨率以适配VAE
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 调整到512x512以适配VAE
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # 减小batch size因为图像更大且需要额外处理
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化训练器
    trainer = EDDPMTrainer(
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_timesteps=1000,
        final_mask_ratio=0.5,
        warmup_epochs=100,  # 预热阶段轮数
        device=device
    )
    
    # 训练循环
    num_epochs = 500000 // len(train_loader)
    total_iterations = 0
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Mask logic: mask=1 means KEEP timestep, mask=0 means SKIP timestep")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            # 生成掩码
            batch_size = batch[0].shape[0]
            mask = trainer.sample_mask(batch_size)
            
            # 训练步骤
            loss = trainer.train_step(batch, mask, epoch)
            epoch_losses.append(loss)
            
            total_iterations += 1
            
            # 计算掩码统计信息
            kept_steps = mask.sum().float().mean().item()  # 平均每个样本保留的时间步数
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Mask_Ratio': f'{trainer.compute_mask_ratio(epoch, num_epochs):.3f}',
                'Kept_Steps': f'{kept_steps:.1f}/{trainer.num_timesteps}',
                'Efficiency': f'{(1-kept_steps/trainer.num_timesteps)*100:.1f}%'
            })
            
            # 保存检查点
            if total_iterations % 10000 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'iteration': total_iterations,
                    'unet_state_dict': trainer.unet.state_dict(),
                    'ema_unet_state_dict': trainer.ema_unet.state_dict(),
                    's': trainer.s,
                    'optimizer_state_dict': trainer.unet_optimizer.state_dict(),
                }
                torch.save(checkpoint, f'checkpoint_{total_iterations}.pt')
                
                # 单独保存掩码参数s
                sigmoid_s = torch.sigmoid(trainer.s.detach().cpu())
                mask_params = {
                    'mask_probabilities': sigmoid_s,
                    'raw_s': trainer.s.detach().cpu(),
                    'num_timesteps': trainer.num_timesteps,
                    'final_mask_ratio': trainer.final_mask_ratio,
                    'iteration': total_iterations,
                    'epoch': epoch,
                    'expected_kept_steps': sigmoid_s.sum().item(),
                    'total_steps': trainer.num_timesteps,
                    'mask_logic': 'mask=1 means KEEP, mask=0 means SKIP'
                }
                torch.save(mask_params, f'mask_params_{total_iterations}.pt')
                logger.info(f"Saved checkpoint and mask parameters at iteration {total_iterations}")
                logger.info(f"Expected kept steps: {sigmoid_s.sum().item():.1f}/{trainer.num_timesteps}")
            
            if total_iterations >= 500000:
                break
        
        avg_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        if total_iterations >= 500000:
            break
    
    # 保存最终模型
    final_checkpoint = {
        'unet_state_dict': trainer.unet.state_dict(),
        'ema_unet_state_dict': trainer.ema_unet.state_dict(),
        's': trainer.s,
        'final_mask_ratio': trainer.final_mask_ratio,
        'num_timesteps': trainer.num_timesteps,
    }
    torch.save(final_checkpoint, 'eddpm_final_model.pt')
    
    # 单独保存最终掩码参数
    sigmoid_s = torch.sigmoid(trainer.s.detach().cpu())
    binary_mask = torch.bernoulli(sigmoid_s)  # 生成二进制掩码示例
    
    final_mask_params = {
        'mask_probabilities': sigmoid_s,
        'raw_s': trainer.s.detach().cpu(),
        'binary_mask_example': binary_mask,
        'num_timesteps': trainer.num_timesteps,
        'final_mask_ratio': trainer.final_mask_ratio,
        'total_iterations': total_iterations,
        'final_epoch': num_epochs,
        'expected_kept_steps': sigmoid_s.sum().item(),
        'total_steps': trainer.num_timesteps,
        'training_completed': True,
        'mask_logic': 'mask=1 means KEEP timestep, mask=0 means SKIP timestep'
    }
    torch.save(final_mask_params, 'final_mask_params.pt')
    
    print("Training completed!")
    print(f"Final expected kept steps: {sigmoid_s.sum().item():.2f}/{trainer.num_timesteps}")
    print(f"Final mask ratio: {trainer.compute_mask_ratio(num_epochs, num_epochs):.3f}")
    print(f"Training efficiency: {(1 - sigmoid_s.sum().item()/trainer.num_timesteps)*100:.1f}% timestep reduction")
    print("Mask parameters saved separately as 'final_mask_params.pt'")
    
    # 显示掩码统计信息
    print(f"Example binary mask kept steps: {binary_mask.sum().item()}/{trainer.num_timesteps}")
    print(f"Mask logic: mask=1 means KEEP timestep, mask=0 means SKIP timestep")

if __name__ == "__main__":
    main()
