<h2 align="center">
  实时目标检测遇见DINOv3
</h2>

<p align="center">
    <a href="https://github.com/Intellindust-AI-Lab/DEIMv2/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://arxiv.org/abs/2509.20787">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2509.20787-red">
    </a>
   <a href="https://intellindust-ai-lab.github.io/projects/DEIMv2/">
        <img alt="project webpage" src="https://img.shields.io/badge/Webpage-DEIMv2-purple">
    </a>
    <a href="https://github.com/Intellindust-AI-Lab/DEIMv2/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/Intellindust-AI-Lab/DEIMv2">
    </a>
    <a href="https://github.com/Intellindust-AI-Lab/DEIMv2/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/Intellindust-AI-Lab/DEIMv2?color=olive">
    </a>
    <a href="https://github.com/Intellindust-AI-Lab/DEIMv2">
        <img alt="stars" src="https://img.shields.io/github/stars/Intellindust-AI-Lab/DEIMv2">
    </a>
    <a href="mailto:shenxi@intellindust.com">
        <img alt="Contact Us" src="https://img.shields.io/badge/Contact-Email-yellow">
    </a>
</p>

<p align="center">
    DEIMv2是DEIM框架的演进版本，同时利用了DINOv3的丰富特征。我们的方法设计了多种模型尺寸，从超轻量版本到S、M、L和X，以适应广泛的应用场景。在这些变体中，DEIMv2实现了最先进的性能，其中S尺寸模型在具有挑战性的COCO基准测试中显著超过了50 AP。
</p>

---


<div align="center">
  <a href="http://www.shihuahuang.cn">黄世华</a><sup>1*</sup>,&nbsp;&nbsp;
  侯永杰<sup>1,2*</sup>,&nbsp;&nbsp;
  刘龙飞<sup>1*</sup>,&nbsp;&nbsp;
  <a href="https://xuanlong-yu.github.io/">余轩龙</a><sup>1</sup>,&nbsp;&nbsp;
  <a href="https://xishen0220.github.io">沈曦</a><sup>1†</sup>&nbsp;&nbsp;
</div>

  
<p align="center">
<i>
1. <a href="https://intellindust-ai-lab.github.io"> 英特工业AI实验室</a> &nbsp;&nbsp; 2. 厦门大学 &nbsp; <br> 
* 共同第一作者 &nbsp;&nbsp; † 通讯作者
</i>
</p>


<p align="center">
<strong>如果您喜欢我们的工作，请给我们一个⭐！</strong>
</p>


<p align="center">
  <img src="./figures/deimv2_coco_AP_vs_Params.png" alt="Image 1" width="49%">
  <img src="./figures/deimv2_coco_AP_vs_GFLOPs.png" alt="Image 2" width="49%">
</p>

</details>

 
  
## 🚀 更新日志
- [x] **\[2025.10.2\]** [DEIMv2已集成到X-AnyLabeling中！](https://github.com/Intellindust-AI-Lab/DEIMv2/issues/25#issue-3473960491) 非常感谢X-AnyLabeling维护者使这成为可能。
- [x] **\[2025.9.26\]** 发布DEIMv2系列。

## 🧭 目录
* [1. 🤖 模型库](#1-模型库)
* [2. ⚡ 快速开始](#2-快速开始)
* [3. 🛠️ 使用方法](#3-使用方法)
* [4. 🧰 工具](#4-工具)
* [5. 📜 引用](#5-引用)
* [6. 🙏 致谢](#6-致谢)
* [7. ⭐ Star历史](#7-star历史)
  
  
## 1. 模型库

| 模型 | 数据集 | AP | 参数量 | GFLOPs | 延迟(ms) | 配置 | 检查点 | 日志 |
| :---: | :---: | :---: | :---: | :---: |:------------:| :---: | :---: | :---: |
| **Atto** | COCO | **23.8** | 0.5M | 0.8 |     1.10     | [yml](./configs/deimv2/deimv2_hgnetv2_atto_coco.yml) | [ckpt](https://drive.google.com/file/d/18sRJXX3FBUigmGJ1y5Oo_DPC5C3JCgYc/view?usp=sharing) | [log](https://drive.google.com/file/d/1M7FLN8EeVHG02kegPN-Wxf_9BlkghZfj/view?usp=sharing) |
| **Femto** | COCO | **31.0** | 1.0M | 1.7 |     1.45     | [yml](./configs/deimv2/deimv2_hgnetv2_femto_coco.yml) | [ckpt](https://drive.google.com/file/d/16hh6l9Oln9TJng4V0_HNf_Z7uYb7feds/view?usp=sharing) | [log](https://drive.google.com/file/d/1_KWVfOr3bB5TMHTNOmDIAO-tZJmKB9-b/view?usp=sharing) |
| **Pico** | COCO | **38.5** | 1.5M | 5.2 |     2.13     | [yml](./configs/deimv2/deimv2_hgnetv2_pico_coco.yml) | [ckpt](https://drive.google.com/file/d/1PXpUxYSnQO-zJHtzrCPqQZ3KKatZwzFT/view?usp=sharing) | [log](https://drive.google.com/file/d/1GwyWotYSKmFQdVN9k2MM6atogpbh0lo1/view?usp=sharing) |
| **N** | COCO | **43.0** | 3.6M | 6.8 |     2.32     | [yml](./configs/deimv2/deimv2_hgnetv2_n_coco.yml) | [ckpt](https://drive.google.com/file/d/1G_Q80EVO4T7LZVPfHwZ3sT65FX5egp9K/view?usp=sharing) | [log](https://drive.google.com/file/d/1QhYfRrUy8HrihD3OwOMJLC-ATr97GInV/view?usp=sharing) |
| **S** | COCO | **50.9** | 9.7M | 25.6 |     5.78     | [yml](./configs/deimv2/deimv2_dinov3_s_coco.yml) | [ckpt](https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL/view?usp=sharing) | [log](https://drive.google.com/file/d/1ydA4lWiTYusV1s3WHq5jSxIq39oxy-Nf/view?usp=sharing) |
| **M** | COCO | **53.0** | 18.1M | 52.2 |     8.80     | [yml](./configs/deimv2/deimv2_dinov3_m_coco.yml) | [ckpt](https://drive.google.com/file/d/1nPKDHrotusQ748O1cQXJfi5wdShq6bKp/view?usp=sharing) | [log](https://drive.google.com/file/d/1i05Q1-O9UH-2Vb52FpFJ4mBG523GUqJU/view?usp=sharing) |
| **L** | COCO | **56.0** | 32.2M | 96.7 |    10.47     | [yml](./configs/deimv2/deimv2_dinov3_l_coco.yml) | [ckpt](https://drive.google.com/file/d/1dRJfVHr9HtpdvaHlnQP460yPVHynMray/view?usp=sharing) | [log](https://drive.google.com/file/d/13mrQxyrf1kJ45Yd692UQwdb7lpGoqsiS/view?usp=sharing) |
| **X** | COCO | **57.8** | 50.3M | 151.6 |    13.75     | [yml](./configs/deimv2/deimv2_dinov3_x_coco.yml) | [ckpt](https://drive.google.com/file/d/1pTiQaBGt8hwtO0mbYlJ8nE-HGztGafS7/view?usp=sharing) | [log](https://drive.google.com/file/d/13QV0SwJw1wHl0xHWflZj1KstBUAovSsV/view?usp=drive_link) |




## 2. 快速开始

### 环境配置

```shell
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt
```


### 数据准备

<details>
<summary> COCO2017数据集 </summary>

1. 从[OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017)或[COCO](https://cocodataset.org/#download)下载COCO2017。
1. 修改[coco_detection.yml](./configs/dataset/coco_detection.yml)中的路径

    ```yaml
    train_dataloader:
        img_folder: /data/COCO2017/train2017/
        ann_file: /data/COCO2017/annotations/instances_train2017.json
    val_dataloader:
        img_folder: /data/COCO2017/val2017/
        ann_file: /data/COCO2017/annotations/instances_val2017.json
    ```

</details>

<details>
<summary>自定义数据集</summary>

要在您的自定义数据集上训练，您需要将其组织为COCO格式。按照以下步骤准备您的数据集：

1. **设置`remap_mscoco_category`为`False`：**

    这可以防止自动重新映射类别ID以匹配MSCOCO类别。

    ```yaml
    remap_mscoco_category: False
    ```

2. **组织图像：**

    按以下结构组织您的数据集目录：

    ```shell
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── ...
    ```

    - **`images/train/`**: 包含所有训练图像。
    - **`images/val/`**: 包含所有验证图像。
    - **`annotations/`**: 包含COCO格式的注释文件。

3. **将注释转换为COCO格式：**

    如果您的注释还不是COCO格式，您需要转换它们。您可以使用以下Python脚本作为参考或利用现有工具：

    ```python
    import json

    def convert_to_coco(input_annotations, output_annotations):
        # 在此处实现转换逻辑
        pass

    if __name__ == "__main__":
        convert_to_coco('path/to/your_annotations.json', 'dataset/annotations/instances_train.json')
    ```

4. **更新配置文件：**

    修改您的[custom_detection.yml](./configs/dataset/custom_detection.yml)。

    ```yaml
    task: detection

    evaluator:
      type: CocoEvaluator
      iou_types: ['bbox', ]

    num_classes: 777 # 您的数据集类别数
    remap_mscoco_category: False

    train_dataloader:
      type: DataLoader
      dataset:
        type: CocoDetection
        img_folder: /data/yourdataset/train
        ann_file: /data/yourdataset/train/train.json
        return_masks: False
        transforms:
          type: Compose
          ops: ~
      shuffle: True
      num_workers: 4
      drop_last: True
      collate_fn:
        type: BatchImageCollateFunction

    val_dataloader:
      type: DataLoader
      dataset:
        type: CocoDetection
        img_folder: /data/yourdataset/val
        ann_file: /data/yourdataset/val/ann.json
        return_masks: False
        transforms:
          type: Compose
          ops: ~
      shuffle: False
      num_workers: 4
      drop_last: False
      collate_fn:
        type: BatchImageCollateFunction
    ```

</details>

### 骨干网络检查点

对于DINOv3 S和S+，请按照https://github.com/facebookresearch/dinov3中的指南下载

对于我们蒸馏的ViT-Tiny和ViT-Tiny+，您可以从[ViT-Tiny](https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing)和[ViT-Tiny+](https://drive.google.com/file/d/1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt/view?usp=sharing)下载。

然后将它们放入./ckpts目录：

```shell
ckpts/
├── dinov3_vits16.pth
├── vitt_distill.pt
├── vittplus_distill.pt
└── ...
```


## 3. 使用方法
<details open>
<summary> COCO2017 </summary>

1. 训练
```shell
# 对于基于ViT的变体
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml --use-amp --seed=0

# 对于基于HGNetv2的变体
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_hgnetv2_${model}_coco.yml --use-amp --seed=0
```

<!-- <summary>2. 测试 </summary> -->
2. 测试
```shell
# 对于基于ViT的变体
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml --test-only -r model.pth

# 对于基于HGNetv2的变体
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_hgnetv2_${model}_coco.yml --test-only -r model.pth
```

<!-- <summary>3. 调优 </summary> -->
3. 调优
```shell
# 对于基于ViT的变体
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml --use-amp --seed=0 -t model.pth

# 对于基于HGNetv2的变体
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_hgnetv2_${model}_coco.yml --use-amp --seed=0 -t model.pth
```
</details>

<details>
<summary> 自定义批次大小 </summary>

例如，如果您想在COCO2017上训练**DEIMv2-S**并将总批次大小加倍到64，请按照以下步骤：

1. **修改您的[deimv2_dinov3_s_coco.yml](./configs/deimv2/deimv2_dinov3_s_coco.yml)**以增加`total_batch_size`：

    ```yaml
    train_dataloader:
      total_batch_size: 64 
      dataset: 
        transforms:
          ops:
            ...
    
      collate_fn:
        ...
    ```

2. **修改您的[deimv2_dinov3_s_coco.yml](./configs/deimv2/deimv2_dinov3_s_coco.yml)**。以下是关键参数的调整方式：

    ```yaml
    optimizer:
      type: AdamW
    
      params: 
        -
          # 除了self.dinov3中的norm/bn/bias
          params: '^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$'  
          lr: 0.00005  # 加倍，线性缩放法则
        -
          # 包括self.dinov3中的所有norm/bn/bias
          params: '^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$'    
          lr: 0.00005   # 加倍，线性缩放法则
          weight_decay: 0.
        - 
          # 包括除self.dinov3之外的所有norm/bn/bias
          params: '^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
          weight_decay: 0.
    
      lr: 0.0005   # 如需要，线性缩放法则
      betas: [0.9, 0.999]
      weight_decay: 0.0001
   
    ema:  # 添加EMA设置
      decay: 0.9998  # 通过1 - (1 - decay) * 2调整
      warmups: 500  # 减半
   
    lr_warmup_scheduler:
      warmup_duration: 250  # 减半
    ```

</details>


<details>
<summary> 自定义输入尺寸 </summary>

如果您想在COCO2017上以320x320的输入尺寸训练**DEIMv2-S**，请按照以下步骤：

1. **修改您的[deimv2_dinov3_s_coco.yml](./configs/deimv2/deimv2_dinov3_s_coco.yml)**：

    ```yaml
    eval_spatial_size: [320, 320]
   
    train_dataloader:
      # 这里我们将total_batch_size设置为64作为示例。
      total_batch_size: 64 
      dataset: 
        transforms:
          ops:
            # 特别是对于Mosaic增强，建议output_size = input_size / 2。
            - {type: Mosaic, output_size: 160, rotation_range: 10, translation_range: [0.1, 0.1], scaling_range: [0.5, 1.5],
               probability: 1.0, fill_value: 0, use_cache: True, max_cached_images: 50, random_pop: True}
            ...
            - {type: Resize, size: [320, 320], }
            ...
        collate_fn:
          base_size: 320
          ...

    val_dataloader:
      dataset:
        transforms:
          ops:
            - {type: Resize, size: [320, 320], }
            ...
    ```
   
</details>

<details>
<summary> 自定义训练轮数 </summary>

如果您想对**DEIMv2-S**进行**20**个轮数的微调，请按照以下步骤（仅供参考；请根据您的需要自由调整）：

```yml
epoches: 32 # 总轮数：20个训练轮数 + EMA 4n = 12。n指匹配配置中的模型尺寸。

flat_epoch: 14    # 4 + 20 // 2
no_aug_epoch: 12  # 4n

train_dataloader:
  dataset: 
    transforms:
      ops:
        ...
      policy:
        epoch: [4, 14, 20]   # [start_epoch, flat_epoch, epoches - no_aug_epoch]

  collate_fn:
    ...
    mixup_epochs: [4, 14]  # [start_epoch, flat_epoch]
    stop_epoch: 20  # epoches - no_aug_epoch
    copyblend_epochs: [4, 20]  # [start_epoch, epoches - no_aug_epoch]
  
DEIMCriterion:
  matcher:
    ...
    matcher_change_epoch: 18  # ~90% of (epoches - no_aug_epoch)

```

</details>

## 4. 工具
<details>
<summary> 部署 </summary>

<!-- <summary>4. 导出onnx </summary> -->
1. 环境配置
```shell
pip install onnx onnxsim
```

2. 导出onnx
```shell
python tools/deployment/export_onnx.py --check -c configs/deimv2/deimv2_dinov3_${model}_coco.yml -r model.pth
```

3. 导出[tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
```shell
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

</details>

<details>
<summary> 推理（可视化） </summary>


1. 环境配置
```shell
pip install -r tools/inference/requirements.txt
```


<!-- <summary>5. 推理 </summary> -->
2. 推理（onnxruntime / tensorrt / torch）

现在支持对图像和视频进行推理。
```shell
python tools/inference/onnx_inf.py --onnx model.onnx --input image.jpg  # video.mp4
python tools/inference/trt_inf.py --trt model.engine --input image.jpg
python tools/inference/torch_inf.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml -r model.pth --input image.jpg --device cuda:0
```
</details>

<details>
<summary> 基准测试 </summary>

1. 环境配置
```shell
pip install -r tools/benchmark/requirements.txt
```

<!-- <summary>6. 基准测试 </summary> -->
2. 模型FLOPs、MACs和参数量
```shell
python tools/benchmark/get_info.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml
```

2. TensorRT延迟
```shell
python tools/benchmark/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
```
</details>

<details>
<summary> Fiftyone可视化  </summary>

1. 环境配置
```shell
pip install fiftyone
```
4. Voxel51 Fiftyone可视化([fiftyone](https://github.com/voxel51/fiftyone))
```shell
python tools/visualization/fiftyone_vis.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml -r model.pth
```
</details>

<details>
<summary> 其他 </summary>

1. 自动恢复训练
```shell
bash reference/safe_training.sh
```

2. 转换模型权重
```shell
python reference/convert_weight.py model.pth
```
</details>


## 5. 引用
如果您在您的工作中使用`DEIMv2`或其方法，请引用以下BibTeX条目：
<details open>
<summary> bibtex </summary>

```latex
@article{huang2025deimv2,
  title={Real-Time Object Detection Meets DINOv3},
  author={Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  journal={arXiv},
  year={2025}
}
  
```
</details>

## 6. 致谢
我们的工作建立在[D-FINE](https://github.com/Peterande/D-FINE)、[RT-DETR](https://github.com/lyuwenyu/RT-DETR)、[DEIM](https://github.com/ShihuaHuang95/DEIM)和[DINOv3](https://github.com/facebookresearch/dinov3)的基础上。感谢他们的出色工作！

✨ 欢迎贡献，如有任何问题请随时联系我们！ ✨

## 7. Star历史

[![Star History Chart](https://api.star-history.com/svg?repos=Intellindust-AI-Lab/DEIMv2&type=Date)](https://www.star-history.com/#Intellindust-AI-Lab/DEIMv2&Date)
