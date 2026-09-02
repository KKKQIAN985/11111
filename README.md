# 农业病虫害识别系统 (Agricultural Pest and Disease Identification System)

## 项目概述
基于深度学习的农业病虫害自动识别系统，使用卷积神经网络(CNN)对常见农作物病虫害进行分类识别。

## 数据集
- **Plant Village Dataset**: https://github.com/spMohanty/PlantVillage-Dataset
  - 包含38000+高质量植物图像
  - 涵盖14种植物、26种疾病 + 健康植物
  - 已预处理和标注

- **Kaggle Plant Diseases**: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
  - 70,295张图像
  - 38个不同的类别

## 项目结构
```
agricultural-pest-identification/
├── data/                      # 数据集目录
│   ├── raw/                  # 原始数据
│   └── processed/            # 处理后的数据
├── src/                      # 源代码
│   ├── data_preprocessing.py # 数据预处理
│   ├── model.py             # 模型定义
│   ├── train.py             # 训练脚本
│   └── predict.py           # 预测脚本
├── notebooks/               # Jupyter笔记本
│   └── exploratory_analysis.ipynb
├── models/                  # 保存的模型
├── results/                 # 结果和可视化
├── requirements.txt         # 依赖包
└── README.md               # 项目说明
```

## 技术栈
- **深度学习框架**: TensorFlow/Keras, PyTorch
- **数据处理**: NumPy, Pandas, OpenCV
- **可视化**: Matplotlib, Seaborn
- **模型架构**: ResNet50, EfficientNet, VGG16

## 快速开始

### 1. 环境配置
```bash
git clone https://github.com/KKKQIAN985/11111.git
cd 11111
pip install -r requirements.txt
```

### 2. 下载数据集
```bash
python src/download_dataset.py
```

### 3. 数据预处理
```bash
python src/data_preprocessing.py
```

### 4. 模型训练
```bash
python src/train.py --model resnet50 --epochs 50 --batch_size 32
```

### 5. 预测测试
```bash
python src/predict.py --image_path path/to/image.jpg --model_path models/best_model.h5
```

## 模型性能指标
- 训练精度 (Training Accuracy)
- 验证精度 (Validation Accuracy)
- 测试精度 (Test Accuracy)
- 混淆矩阵 (Confusion Matrix)
- 精确率/召回率/F1分数 (Precision/Recall/F1-Score)

## 关键研究内容
1. **数据增强策略** - 旋转、翻转、亮度调整等
2. **迁移学习** - 使用预训练模型 (ResNet, EfficientNet)
3. **模型优化** - 超参数调优、正则化
4. **可解释性分析** - Grad-CAM 可视化、特征重要性
5. **性能对比** - 多个模型架构的对比分析

## 论文相关内容
- 方法论: 数据预处理、模型架构、训练策略
- 实验: 数据集描述、模型性能对比、消融研究
- 结果: 准确率、混淆矩阵、错误分析
- 讨论: 模型优势、局限性、未来改进方向

## 依赖要求
- Python 3.8+
- TensorFlow 2.10+
- PyTorch 1.9+ (可选)
- CUDA 11.0+ (用于GPU加速，可选)

## 许可证
MIT License

## 引用
如果您在研究中使用本项目，请引用：
```
@misc{agricultural_pest_identification,
  author = {Your Name},
  title = {Agricultural Pest and Disease Identification System},
  year = {2024},
  url = {https://github.com/KKKQIAN985/11111}
}
```

## 联系方式
欢迎提交Issue和Pull Request！
