"""
数据预处理模块 (Data Preprocessing Module)
功能: 加载、清理、增强农业病虫害图像数据
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import shutil
from tqdm import tqdm


class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, image_size=(224, 224)):
        """
        初始化预处理器
        
        Args:
            image_size: 输入图像大小 (height, width)
        """
        self.image_size = image_size
        self.train_data_generator = None
        self.val_data_generator = None
        
    def load_images_from_directory(self, data_dir, label_map=None):
        """
        从目录加载图像和标签
        
        Args:
            data_dir: 数据目录路径
            label_map: 标签映射字典
            
        Returns:
            images: 图像数组
            labels: 标签数组
            class_names: 类别名称列表
        """
        images = []
        labels = []
        class_names = sorted(os.listdir(data_dir))
        
        if label_map is None:
            label_map = {name: idx for idx, name in enumerate(class_names)}
        
        print(f"发现 {len(class_names)} 个病虫害类别: {class_names}")
        
        for class_name in class_names:
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            class_label = label_map[class_name]
            image_files = os.listdir(class_path)
            
            print(f"加载类别 '{class_name}' ({len(image_files)} 张图像)...")
            
            for img_file in tqdm(image_files, desc=class_name):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = load_img(img_path, target_size=self.image_size)
                    img_array = img_to_array(img) / 255.0  # 归一化到 [0, 1]
                    images.append(img_array)
                    labels.append(class_label)
                except Exception as e:
                    print(f"错误加载图像 {img_path}: {e}")
        
        return np.array(images), np.array(labels), class_names, label_map
    
    def augment_images(self, images, labels, save_dir=None):
        """
        数据增强 (旋转、翻转、亮度调整等)
        
        Args:
            images: 图像数组
            labels: 标签数组
            save_dir: 保存增强后图像的目录
            
        Returns:
            augmented_images: 增强后的图像
            augmented_labels: 对应的标签
        """
        augmenter = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        augmented_images = []
        augmented_labels = []
        
        print("执行数据增强...")
        for img, label in tqdm(zip(images, labels), total=len(images)):
            # 原始图像
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # 增强图像 (每张增强3次)
            img_expanded = np.expand_dims(img, 0)
            aug_iter = augmenter.flow(img_expanded, batch_size=1)
            
            for _ in range(3):
                aug_img = next(aug_iter)[0]
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        
        print(f"增强后数据集大小: {augmented_images.shape[0]} 张图像")
        
        return augmented_images, augmented_labels
    
    def split_data(self, images, labels, test_size=0.2, val_size=0.1, random_state=42):
        """
        分割数据集为训练、验证、测试集
        
        Args:
            images: 图像数组
            labels: 标签数组
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
            
        Returns:
            训练、验证、测试集的 (images, labels) 元组
        """
        # 先分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # 再从剩余数据中分出验证集
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        print(f"数据集划分:")
        print(f"  训练集: {X_train.shape[0]} 张图像")
        print(f"  验证集: {X_val.shape[0]} 张图像")
        print(f"  测试集: {X_test.shape[0]} 张图像")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_data_generators(self, train_data, val_data, batch_size=32):
        """
        创建训练和验证数据生成器
        
        Args:
            train_data: (X_train, y_train) 元组
            val_data: (X_val, y_val) 元组
            batch_size: 批大小
            
        Returns:
            train_generator, val_generator
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_augmenter = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        val_augmenter = ImageDataGenerator()  # 验证集不增强
        
        train_generator = train_augmenter.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_generator = val_augmenter.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
        
        return train_generator, val_generator
    
    @staticmethod
    def display_sample_images(images, labels, class_names, num_samples=9):
        """
        显示样本图像
        
        Args:
            images: 图像数组
            labels: 标签数组
            class_names: 类别名称
            num_samples: 显示的样本数
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_samples):
            axes[i].imshow(images[i])
            axes[i].set_title(f"Class: {class_names[labels[i]]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
        print("样本图像已保存到 'sample_images.png'")
        plt.close()


if __name__ == "__main__":
    # 示例使用
    data_dir = "data/processed/train"
    
    preprocessor = DataPreprocessor(image_size=(224, 224))
    
    if os.path.exists(data_dir):
        images, labels, class_names, label_map = preprocessor.load_images_from_directory(data_dir)
        print(f"加载完成: {images.shape[0]} 张图像, {len(class_names)} 个类别")
        
        # 显示样本
        preprocessor.display_sample_images(images, labels, class_names)
    else:
        print(f"数据目录不存在: {data_dir}")
