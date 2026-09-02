"""
训练脚本 (Training Script)
功能: 训练病虫害识别模型
"""

import os
import argparse
import numpy as np
import json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    TensorBoard, CSVLogger
)
from tensorflow.keras.utils import to_categorical

from model import PestIdentificationModel
from data_preprocessing import DataPreprocessor


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model_name='resnet50', batch_size=32, epochs=50, 
                 learning_rate=1e-3, input_shape=(224, 224, 3)):
        """
        初始化训练器
        
        Args:
            model_name: 模型名称
            batch_size: 批大小
            epochs: 训练轮数
            learning_rate: 学习率
            input_shape: 输入形状
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.history = None
        self.model = None
        
        # 创建输出目录
        self.checkpoint_dir = 'models'
        self.log_dir = 'logs'
        self.results_dir = 'results'
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def build_model(self, num_classes):
        """构建模型"""
        print(f"\n构建 {self.model_name} 模型...")
        self.model = PestIdentificationModel.create_model(
            model_name=self.model_name,
            input_shape=self.input_shape,
            num_classes=num_classes
        )
        
        self.model = PestIdentificationModel.compile_model(
            self.model,
            learning_rate=self.learning_rate
        )
        
        print(self.model.summary())
        return self.model
    
    def get_callbacks(self, model_name):
        """获取回调函数"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # 早停
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # 学习率调整
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # 模型检查点
            ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, f'{model_name}_{timestamp}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard
            TensorBoard(
                log_dir=os.path.join(self.log_dir, timestamp),
                histogram_freq=1,
                write_graph=True
            ),
            
            # CSV日志
            CSVLogger(
                os.path.join(self.log_dir, f'{model_name}_{timestamp}.csv')
            )
        ]
        
        return callbacks
    
    def train(self, train_generator, val_generator, steps_per_epoch=None, 
              validation_steps=None):
        """
        训练模型
        
        Args:
            train_generator: 训练数据生成器
            val_generator: 验证数据生成器
            steps_per_epoch: 每轮步数
            validation_steps: 验证步数
            
        Returns:
            history: 训练历史
        """
        callbacks = self.get_callbacks(self.model_name)
        
        print(f"\n开始训练 {self.model_name}...")
        print(f"批大小: {self.batch_size}, 训练轮数: {self.epochs}")
        
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_generator):
        """
        评估模型
        
        Args:
            test_generator: 测试数据生成器
            
        Returns:
            test_loss, test_accuracy
        """
        print("\n评估测试集...")
        results = self.model.evaluate(test_generator, verbose=1)
        return results
    
    def save_model(self, model_name=None):
        """保存模型"""
        if model_name is None:
            model_name = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = os.path.join(self.checkpoint_dir, f"{model_name}_final.h5")
        self.model.save(model_path)
        print(f"模型已保存到: {model_path}")
        return model_path
    
    def save_history(self, history_dict=None):
        """保存训练历史"""
        if history_dict is None:
            history_dict = self.history.history
        
        history_path = os.path.join(self.results_dir, 
                                   f"{self.model_name}_history.json")
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"训练历史已保存到: {history_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练病虫害识别模型")
    
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='数据目录')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['custom_cnn', 'resnet50', 'efficientnet', 'vgg16', 'mobilenetv2'],
                       help='模型名称')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--image_size', type=int, default=224,
                       help='输入图像大小')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        print("请先运行: python src/data_preprocessing.py")
        return
    
    # 预处理数据
    print("\n加载数据...")
    preprocessor = DataPreprocessor(image_size=(args.image_size, args.image_size))
    
    train_dir = os.path.join(args.data_dir, 'train')
    images, labels, class_names, label_map = preprocessor.load_images_from_directory(train_dir)
    
    num_classes = len(class_names)
    print(f"类别数: {num_classes}")
    
    # 转换标签
    labels = to_categorical(labels, num_classes)
    
    # 分割数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(
        images, labels, test_size=0.15, val_size=0.15
    )
    
    # 创建数据生成器
    train_gen, val_gen = preprocessor.get_data_generators(
        (X_train, y_train), 
        (X_val, y_val),
        batch_size=args.batch_size
    )
    
    # 训练模型
    trainer = ModelTrainer(
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        input_shape=(args.image_size, args.image_size, 3)
    )
    
    trainer.build_model(num_classes)
    trainer.train(train_gen, val_gen)
    
    # 评估
    test_gen = preprocessor.get_data_generators(
        (X_test, y_test),
        (X_test, y_test),
        batch_size=args.batch_size
    )[1]
    
    test_results = trainer.evaluate(test_gen)
    print(f"\n测试集 - 损失: {test_results[0]:.4f}, 精度: {test_results[1]:.4f}")
    
    # 保存
    trainer.save_model()
    trainer.save_history()
    
    print("\n训练完成!")


if __name__ == "__main__":
    main()
