"""
模型评估和可视化模块 (Evaluation and Visualization Module)
功能: 绘制训练曲线、混淆矩阵、特征可视化等
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from itertools import cycle


class ModelEvaluator:
    """模型评估类"""
    
    def __init__(self, model_path=None, class_names=None):
        """
        初始化评估器
        
        Args:
            model_path: 模型文件路径
            class_names: 类别名称列表
        """
        self.model = None
        self.class_names = class_names or []
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"加载模型: {model_path}")
    
    def plot_training_history(self, history_file, output_file='training_history.png'):
        """
        绘制训练历史曲线
        
        Args:
            history_file: 训练历史JSON文件
            output_file: 输出图像文件
        """
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 准确率
        if 'accuracy' in history:
            axes[0, 0].plot(history['accuracy'], label='训练准确率', linewidth=2)
            axes[0, 0].plot(history['val_accuracy'], label='验证准确率', linewidth=2)
            axes[0, 0].set_title('模型准确率', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('轮次')
            axes[0, 0].set_ylabel('准确率')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 损失
        if 'loss' in history:
            axes[0, 1].plot(history['loss'], label='训练损失', linewidth=2)
            axes[0, 1].plot(history['val_loss'], label='验证损失', linewidth=2)
            axes[0, 1].set_title('模型损失', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('轮次')
            axes[0, 1].set_ylabel('损失值')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Top-3 准确率
        if 'top_3_accuracy' in history:
            axes[1, 0].plot(history['top_3_accuracy'], label='训练 Top-3', linewidth=2)
            axes[1, 0].plot(history['val_top_3_accuracy'], label='验证 Top-3', linewidth=2)
            axes[1, 0].set_title('Top-3 准确率', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('轮次')
            axes[1, 0].set_ylabel('准确率')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 学习率变化
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, '训练统计', 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"训练历史已保存到: {output_file}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, output_file='confusion_matrix.png'):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            output_file: 输出图像文件
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': '样本数'})
        
        plt.title('混淆矩阵 - 病虫害分类', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {output_file}")
        plt.close()
    
    def plot_classification_report(self, y_true, y_pred, output_file='classification_report.png'):
        """
        绘制分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            output_file: 输出图像文件
        """
        report = classification_report(y_true, y_pred, 
                                      target_names=self.class_names,
                                      output_dict=True)
        
        # 提取指标
        metrics_names = ['Precision', 'Recall', 'F1-Score']
        class_metrics = {}
        
        for class_name in self.class_names:
            if class_name in report:
                class_metrics[class_name] = [
                    report[class_name]['precision'],
                    report[class_name]['recall'],
                    report[class_name]['f1-score']
                ]
        
        # 绘制
        fig, ax = plt.subplots(figsize=(14, 10))
        
        x_pos = np.arange(len(self.class_names))
        width = 0.25
        
        for i, metric_name in enumerate(metrics_names):
            values = [class_metrics[cn][i] if cn in class_metrics else 0 
                     for cn in self.class_names]
            ax.bar(x_pos + i*width, values, width, label=metric_name)
        
        ax.set_xlabel('病虫害类别', fontsize=12)
        ax.set_ylabel('得分', fontsize=12)
        ax.set_title('分类性能指标', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"分类报告已保存到: {output_file}")
        plt.close()
    
    def plot_roc_curves(self, y_true, y_pred_proba, output_file='roc_curves.png'):
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签 (one-hot编码)
            y_pred_proba: 预测概率
            output_file: 输出图像文件
        """
        n_classes = y_true.shape[1]
        
        plt.figure(figsize=(12, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机分类器')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
        plt.title('ROC 曲线 - 多分类问题', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存到: {output_file}")
        plt.close()
    
    def plot_class_distribution(self, y_true, output_file='class_distribution.png'):
        """
        绘制类别分布
        
        Args:
            y_true: 真实标签
            output_file: 输出图像文件
        """
        unique, counts = np.unique(y_true, return_counts=True)
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(self.class_names, counts, color='skyblue', edgecolor='navy')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('病虫害类别', fontsize=12)
        plt.ylabel('样本数', fontsize=12)
        plt.title('数据集类别分布', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"类别分布已保存到: {output_file}")
        plt.close()


class FeatureVisualizer:
    """特征可视化类 (GradCAM等)"""
    
    def __init__(self, model):
        """
        初始化可视化器
        
        Args:
            model: Keras 模型
        """
        self.model = model
    
    def compute_grad_cam(self, img_array, layer_name, pred_index=None):
        """
        计算 Grad-CAM
        
        Args:
            img_array: 输入图像 (1, H, W, 3)
            layer_name: 目标层名称
            pred_index: 预测类别索引
            
        Returns:
            heatmap: 热力图
        """
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def plot_grad_cam(self, image_path, layer_name, output_file='grad_cam.png',
                      class_names=None):
        """
        绘制 Grad-CAM 可视化
        
        Args:
            image_path: 图像路径
            layer_name: 目标层名称
            output_file: 输出图像文件
            class_names: 类别名称
        """
        # 加载并预处理图像
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 获取预测
        predictions = self.model.predict(img_array, verbose=0)
        pred_class = np.argmax(predictions[0])
        
        # 计算 Grad-CAM
        heatmap = self.compute_grad_cam(img_array, layer_name, pred_class)
        
        # 绘制
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(img)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 热力图
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM 热力图')
        axes[1].axis('off')
        
        # 叠加图像
        heatmap_resized = np.uint8(255 * heatmap)
        heatmap_3ch = np.stack([heatmap_resized]*3, axis=2)
        overlay = np.uint8(np.array(img) * 0.7 + heatmap_3ch * 0.3)
        axes[2].imshow(overlay)
        axes[2].set_title('热力图叠加')
        axes[2].axis('off')
        
        pred_label = class_names[pred_class] if class_names else f'Class {pred_class}'
        confidence = predictions[0][pred_class]
        fig.suptitle(f'预测: {pred_label} (置信度: {confidence:.2%})',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM 可视化已保存到: {output_file}")
        plt.close()


if __name__ == "__main__":
    # 示例用法
    print("评估模块已准备好使用")
    print("使用方式:")
    print("  from evaluation import ModelEvaluator")
    print("  evaluator = ModelEvaluator(model_path='models/best_model.h5')")
    print("  evaluator.plot_training_history('logs/history.json')")
