"""
配置文件 (Configuration File)
功能: 存储项目配置参数
"""

import os
from pathlib import Path

# ============ 项目路径配置 ============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'
RESULTS_DIR = PROJECT_ROOT / 'results'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# 创建必要目录
for dir_path in [MODELS_DIR, LOGS_DIR, RESULTS_DIR, PROCESSED_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============ 数据配置 ============
IMAGE_SIZE = (224, 224)  # 输入图像大小
NUM_CHANNELS = 3  # RGB 通道数

# 数据集分割比例
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# 数据增强参数
DATA_AUGMENTATION = {
    'rotation_range': 40,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# ============ 模型配置 ============
# 可用模型: 'custom_cnn', 'resnet50', 'efficientnet', 'vgg16', 'mobilenetv2'
MODEL_NAME = 'resnet50'

# 模型超参数
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# 优化器参数
OPTIMIZER = {
    'name': 'Adam',
    'learning_rate': LEARNING_RATE,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-7
}

# ============ 训练配置 ============
RANDOM_SEED = 42
GPU_MEMORY_FRACTION = None  # None = 使用全部可用GPU内存
USE_MIXED_PRECISION = False  # 混合精度训练

# ============ 评估配置 ============
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'auc',
    'top_k_accuracy'
]

# Top-K 准确率设置
TOP_K = 3

# ============ 日志配置 ============
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
SAVE_LOGS = True
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============ 输出配置 ============
SAVE_PLOTS = True
PLOT_DPI = 300
PLOT_FORMAT = 'png'

# ============ 类别配置 ============
# 26种常见植物病虫害类别 (PlantVillage Dataset)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two_spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

NUM_CLASSES = len(CLASS_NAMES)

# ============ 回调配置 ============
CALLBACKS_CONFIG = {
    'early_stopping': {
        'monitor': 'val_accuracy',
        'patience': EARLY_STOPPING_PATIENCE,
        'restore_best_weights': True,
        'verbose': 1
    },
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': REDUCE_LR_FACTOR,
        'patience': REDUCE_LR_PATIENCE,
        'min_lr': 1e-7,
        'verbose': 1
    },
    'model_checkpoint': {
        'monitor': 'val_accuracy',
        'save_best_only': True,
        'verbose': 1
    },
    'tensorboard': {
        'histogram_freq': 1,
        'write_graph': True,
        'write_images': False
    }
}

# ============ 论文相关配置 ============
PAPER_CONFIG = {
    'title': '基于深度学习的农业病虫害自动识别系统',
    'title_en': 'Agricultural Pest and Disease Identification System Based on Deep Learning',
    'authors': ['Your Name'],
    'keywords': ['深度学习', '病虫害识别', 'CNN', '转移学习'],
    'keywords_en': ['Deep Learning', 'Pest Identification', 'CNN', 'Transfer Learning'],
    'dataset': 'Plant Village Dataset',
    'models_used': ['ResNet50', 'EfficientNet', 'VGG16', 'MobileNetV2'],
    'publication_target': 'SCI Q4'
}

# ============ 导出配置到字典 ============
CONFIG = {
    'paths': {
        'root': str(PROJECT_ROOT),
        'data': str(DATA_DIR),
        'raw_data': str(RAW_DATA_DIR),
        'processed_data': str(PROCESSED_DATA_DIR),
        'models': str(MODELS_DIR),
        'logs': str(LOGS_DIR),
        'results': str(RESULTS_DIR)
    },
    'data': {
        'image_size': IMAGE_SIZE,
        'num_channels': NUM_CHANNELS,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES
    },
    'model': {
        'name': MODEL_NAME,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE
    },
    'paper': PAPER_CONFIG
}


def print_config():
    """打印配置信息"""
    import json
    print("=" * 60)
    print("项目配置信息")
    print("=" * 60)
    print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
    print("=" * 60)


if __name__ == "__main__":
    print_config()
