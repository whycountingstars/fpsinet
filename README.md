# PSINet-like Stripe Denoiser (RDB + U-Net + FFT/VGG/SSIM + LowFreq/TV)

这是为条纹/线性噪声去除任务定制的 PyTorch 项目：
- 单通道（二值 / 单值）输出支持（binary modes）
- Low-frequency branch + TV 正则，用于学习平滑背景（适合需要“下半部分平整”）
- 轻量注意力（CBAM）用于自动关注关键区域
- 可在 train.yml 中一键调整模型容量与损失权重

快速开始（环境）
1. 建议创建并激活 conda 环境，然后安装依赖：
   conda create -n fpsinet python=3.10 -y
   conda activate fpsinet
   pip install -r requirements.txt

2. 如果希望在 GPU 上训练，请确保安装了 CUDA-enabled PyTorch compatible with your driver. Example (conda, CUDA 11.8):
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

数据准备（重要）
- clean_dir：建议放置单值/真值（single-channel）或二值（0/1）标签。如果你已经把 clean 图像保存为二值 PNG，可直接使用。
- noisy_dir：可选。如果提供 noisy_dir 且文件名与 clean 相同，会优先使用真实配对数据；若不提供则用内置合成器生成 noisy。
- train.yml 中可启用 dataset.force_gray=true 和 dataset.binarize_threshold=0.5 来在加载时自动转换与二值化 clean。

修改要点
1) 直接输出单值 / 二值
- train.yml: extra.binary_output: true （训练时使用 BCEWithLogitsLoss）
- 数据：确保 clean_dir 存放二值图像，或在 train.yml dataset.force_gray=true 和 dataset.binarize_threshold=0.5
- 推理：在 test.yml 设置 threshold: 0.5，然后 pred_bin = (sigmoid(out) > threshold)

2) 注意力机制（CBAM）
- 已在模型中集成，可在 train.yml model.use_cbam: true/false 控制
- CBAM 能帮助模型自动关注重要通道和空间区域（尤其在纹理/斑点背景中有明显效果）

3) 增加模型深度
- 修改 train.yml 中 model.base_channels、rdb_layers、n_rdb_per_scale、n_scales（默认已调到更大值适合 22GB）
- 若遇到 OOM，请减小 batch_size 或 base_channels

Total Variation (TV) 介绍与如何使用
- TV（全称 Total Variation）常用于鼓励图像平滑，定义（离散近似）：
  TV(x) = mean(|x_{i+1,j} - x_{i,j}|) + mean(|x_{i,j+1} - x_{i,j}|)
  即对水平/竖直相邻像素梯度的绝对值求平均或求和。
- 在本项目的用途：对模型预测的低频残差（res_low）施加 TV 惩罚会鼓励低频分量更加平滑，从而使下半区域或背景更“平整”。
- Train.yml 中的参数与建议值（实验起点）：
  - losses.lambda_lowfreq: 1.0 （把预测与 GT 的低通部分对齐）
  - losses.lambda_tv: 0.01 （鼓励低频分量平滑；若效果不足可增到 0.02~0.05）
  - extra.lowfreq_blur_sigma: 2.0 （计算低频对齐时使用的 Gaussian sigma）
  - extra.region_weight: 3~8 （如果只想把下半区域强制平整，把该值设大）
- 使用建议：
  - 先用 binary L1-only（lambda_fft=0, lambda_perc=0, lambda_ssim=0）跑 10-20 epoch 做 baseline；
  - 再打开低频监督（lambda_lowfreq=1.0）与小 TV（lambda_tv=0.01），观察是否改善；
  - 若下半仍不平整，逐步把 lambda_tv 增至 0.02 或 region_weight 提高到 5；
  - 注意 TV 过大会导致过度平滑、丢失边缘细节；建议逐步调参并查看可视化结果。

建议的实验组合（针对 22GB 显卡）
- Baseline (fast test):
  - train.yml: base_channels=64, rdb_layers=4, n_scales=4, batch_size=8, lambda_perc=0, lambda_fft=0
  - extra.binary_output=true
- Lowfreq + TV:
  - losses.lambda_lowfreq=1.0, losses.lambda_tv=0.01, extra.region_weight=5.0
  - keep lambda_perc=0, lambda_fft small (0.05) or 0
- Capacity increase for stronger modeling:
  - base_channels=80, rdb_layers=4, n_rdb_per_scale=3 (watch OOM; reduce batch_size if needed)

如何查看模型中间量（gate / res_low / res_high）
- train.py 默认会把 noisy | pred | clean 的合成可视化保存在 checkpoints/vis；
- 我建议把 train.py 做扩展 (我可以替你改) 以单独保存 gate/res_low/res_high（灰度图或伪彩色）便于调参；如需我可以在下一步提交该改动。

运行示例
- 训练（使用 run_from_config）
  python run_from_config.py --mode train --config train.yml
- 推理
  python run_from_config.py --mode test --config test.yml

调参流程（一步到位）
1. 确保 clean_dir 为二值 GT 或使用 force_gray/binarize_threshold；
2. 先跑 Baseline (L1-only + binary) 若效果不错，再加入 lowfreq + TV；
3. 调整 region_weight 观察下半区域效果；若整体欠拟合则增加模型容量和 training epochs；
4. 使用 profiler/monitoring（nvidia-smi -l 1）检查 GPU 利用率，并在需要时增加 num_workers、pin_memory、persistent_workers 或增大 batch size。

需要我替你继续做的事（选项）
- A）我把 train.py 扩展为把 gate/res_low/res_high 单独保存为图像（便于调参）；
- B）我把 run_from_config 的 build_train_cmd / train.yml 示例同步更新（若你希望通过 YAML 控制更多新 dataset 参数）；
- C）帮你生成一组预设的 train.yml 文件（baseline / lowfreq-tv / large-model）供你直接拷贝运行；
- D）其他（请说明）

请选择 A/B/C/D 或“都做”，我会按你选择继续修改并提交相应文件。祝你训练顺利 — 如果你把最近一次训练的可视化图（checkpoints/vis 里）贴上来，我可以直接根据图建议具体 lambda 值与 region_weight。
