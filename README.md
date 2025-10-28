```markdown
# PSINet-like Stripe Denoiser (RDB + U-Net + FFT/VGG/SSIM losses)

这是一个为条纹/线性噪声去除任务定制的 PyTorch 项目，包含：
- 模型：U-Net 风格 encoder-decoder，每尺度使用 Residual Dense Block (RDB)
- 数据：支持真实 clean/noisy 配对训练与内置 realistic stripe 合成增强
- 损失：L1 + FFT magnitude + VGG perceptual + SSIM
- 训练脚本：支持 AMP、ReduceLROnPlateau、可视化、显存自测

快速开始
1. 安装依赖（建议在 conda 环境或 venv 中）：
   - pip install -r requirements.txt

2. 准备数据：
   - 如果有 clean + noisy 配对：把干净图放在 /data/clean，带噪图放在 /data/noisy（同名文件）
   - 如果只有 clean，可仅传入 --clean_dir，脚本会使用合成噪声生成 noisy

3. 显存自测（估算 batch）：
   - python train.py --clean_dir /path/to/clean --patch 128 --test_batch

4. 训练（示例）：
   - 使用真实配对训练（优先使用真实 noisy）：
     python train.py --clean_dir /data/clean --noisy_dir /data/noisy --batch_size 8 --patch 128 --epochs 120 --out_dir ./outputs

   - 使用合成预训练再微调：
     1) 预训练（合成）：
        python train.py --clean_dir /data/clean --synth_prob 1.0 --batch_size 16 --patch 128 --epochs 60 --out_dir ./pretrain
     2) 微调（真实配对）：
        python train.py --clean_dir /data/clean --noisy_dir /data/noisy --synth_prob 0.0 --batch_size 8 --patch 128 --epochs 60 --out_dir ./finetune

5. 调整损失权重：
   - train.py 支持命令行参数:
     --lambda_fft (默认 0.2), --lambda_perc (默认 0.01), --lambda_ssim (默认 0.1)

将项目推送到 GitHub（本地操作示例）
1. 初始化本地仓库并提交：
   git init
   git add .
   git commit -m "Initial commit: PSINet-like stripe denoiser"

2. 创建远程仓库（在 GitHub 网站或使用 gh CLI），然后推送：
   git remote add origin git@github.com:<your-username>/<repo>.git
   git branch -M main
   git push -u origin main


其他说明
- 若启用 VGG perceptual loss，会自动加载 torchvision 的 VGG16 权重（需联网下载一次）
- 若使用 AMP（默认启用），请确保 PyTorch 与 CUDA 环境配置正确
- 如果你的真实 noisy 样本具有某些特定角度/频率的条纹，建议用一到几张代表图做 FFT 分析后把 default_synth_params_generator 调整为更贴合真实分布
```
