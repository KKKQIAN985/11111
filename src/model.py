"""
模型定义模块 (Model Definition Module)
功能: 定义多种深度学习模型架构
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, VGG16, MobileNetV2, EfficientNetB0
)
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
)


class PestIdentificationModel:
    """病虫害识别模型工厂类"""
    
    @staticmethod
    def build_custom_cnn(input_shape=(224, 224, 3), num_classes=26):
        """
        构建自定义 CNN 模型
        
        Args:
            input_shape: 输入图像形状
            num_classes: 输出类别数
            
        Returns:
            model: Keras 模型
        """
        model = models.Sequential([
            # 块1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=input_shape, name='conv1_1'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25),
            
            # 块2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25),
            
            # 块3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25),
            
            # 块4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool4'),
            layers.Dropout(0.25),
            
            # 全连接层
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu', name='fc1'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu', name='fc2'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        return model
    
    @staticmethod
    def build_resnet50(input_shape=(224, 224, 3), num_classes=26, trainable_layers=50):
        """
        构建 ResNet50 转移学习模型
        
        Args:
            input_shape: 输入图像形状
            num_classes: 输出类别数
            trainable_layers: 可训练层数
            
        Returns:
            model: Keras 模型
        """
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # 冻结早期层
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        
        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', name='fc1'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu', name='fc2'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax', name='output')
        ])
        
        return model
    
    @staticmethod
    def build_efficientnet(input_shape=(224, 224, 3), num_classes=26):
        """
        构建 EfficientNetB0 转移学习模型
        
        Args:
            input_shape: 输入图像形状
            num_classes: 输出类别数
            
        Returns:
            model: Keras 模型
        """
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # 冻结基础模型
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu', name='fc1'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(num_classes, activation='softmax', name='output')
        ])
        
        return model
    
    @staticmethod
    def build_vgg16(input_shape=(224, 224, 3), num_classes=26):
        """
        构建 VGG16 转移学习模型
        
        Args:
            input_shape: 输入图像形状
            num_classes: 输出类别数
            
        Returns:
            model: Keras 模型
        """
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # 冻结早期层，只训练后3个卷积块
        for layer in base_model.layers[:-9]:
            layer.trainable = False
        
        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', name='fc1'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu', name='fc2'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax', name='output')
        ])
        
        return model
    
    @staticmethod
    def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=26):
        """
        构建 MobileNetV2 转移学习模型 (轻量级)
        
        Args:
            input_shape: 输入图像形状
            num_classes: 输出类别数
            
        Returns:
            model: Keras 模型
        """
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # 冻除了最后20层
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu', name='fc1'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(num_classes, activation='softmax', name='output')
        ])
        
        return model
    
    @staticmethod
    def create_model(model_name='resnet50', input_shape=(224, 224, 3), num_classes=26):
        """
        模型工厂方法
        
        Args:
            model_name: 模型名称 ('custom_cnn', 'resnet50', 'efficientnet', 'vgg16', 'mobilenetv2')
            input_shape: 输入图像形状
            num_classes: 输出类别数
            
        Returns:
            model: Keras 模型
        """
        models_dict = {
            'custom_cnn': PestIdentificationModel.build_custom_cnn,
            'resnet50': PestIdentificationModel.build_resnet50,
            'efficientnet': PestIdentificationModel.build_efficientnet,
            'vgg16': PestIdentificationModel.build_vgg16,
            'mobilenetv2': PestIdentificationModel.build_mobilenetv2
        }
        
        if model_name not in models_dict:
            raise ValueError(f"未知模型: {model_name}. 可用模型: {list(models_dict.keys())}")
        
        return models_dict[model_name](input_shape, num_classes)
    
    @staticmethod
    def compile_model(model, learning_rate=1e-3):
        """
        编译模型
        
        Args:
            model: Keras 模型
            learning_rate: 学习率
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        return model


if __name__ == "__main__":
    # 示例: 创建并打印模型
    print("创建 ResNet50 模型...")
    model = PestIdentificationModel.create_model('resnet50', num_classes=26)
    model = PestIdentificationModel.compile_model(model)
    model.summary()
