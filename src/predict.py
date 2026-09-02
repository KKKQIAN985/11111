"""
预测脚本 (Prediction Script)
功能: 对新图像进行病虫害预测
"""

import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import json


class PestPredictor:
    """病虫害预测器"""
    
    def __init__(self, model_path, class_names=None, image_size=(224, 224)):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
            class_names: 类别名称列表
            image_size: 输入图像大小
        """
        self.model_path = model_path
        self.image_size = image_size
        self.class_names = class_names
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = load_model(model_path)
        print("模型加载完成")
    
    def preprocess_image(self, image_path):
        """
        预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            preprocessed_image, original_image
        """
        # 加载原始图像
        original_img = Image.open(image_path)
        
        # 调整大小并转换
        img = load_img(image_path, target_size=self.image_size)
        img_array = img_to_array(img) / 255.0  # 归一化
        img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
        
        return img_array, original_img
    
    def predict(self, image_path, top_k=3):
        """
        预测图像
        
        Args:
            image_path: 图像路径
            top_k: 返回前k个预测
            
        Returns:
            predictions: 预测结果字典
        """
        # 预处理图像
        img_array, original_img = self.preprocess_image(image_path)
        
        # 进行预测
        predictions = self.model.predict(img_array, verbose=0)
        pred_probs = predictions[0]
        
        # 获取前k个预测
        top_indices = np.argsort(pred_probs)[::-1][:top_k]
        
        results = {
            'image_path': image_path,
            'predictions': []
        }
        
        for idx in top_indices:
            class_name = self.class_names[idx] if self.class_names else f"Class {idx}"
            probability = float(pred_probs[idx])
            
            results['predictions'].append({
                'class': class_name,
                'confidence': probability,
                'confidence_percent': f"{probability*100:.2f}%"
            })
        
        return results, original_img
    
    def predict_batch(self, image_dir, output_file='predictions.json'):
        """
        批量预测
        
        Args:
            image_dir: 包含图像的目录
            output_file: 输出JSON文件
            
        Returns:
            all_predictions: 所有预测结果
        """
        all_predictions = []
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(supported_formats)]
        
        print(f"找到 {len(image_files)} 张图像")
        
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            
            try:
                results, _ = self.predict(img_path)
                all_predictions.append(results)
                print(f"✓ {img_file}: {results['predictions'][0]['class']} "
                      f"({results['predictions'][0]['confidence_percent']})")
            except Exception as e:
                print(f"✗ {img_file}: {e}")
        
        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        
        print(f"\n预测结果已保存到: {output_file}")
        
        return all_predictions
    
    def visualize_prediction(self, image_path, output_path='prediction_result.png'):
        """
        可视化预测结果
        
        Args:
            image_path: 图像路径
            output_path: 输出图像路径
        """
        results, original_img = self.predict(image_path, top_k=5)
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 显示图像
        axes[0].imshow(original_img)
        axes[0].set_title("输入图像")
        axes[0].axis('off')
        
        # 显示预测结果
        classes = [p['class'] for p in results['predictions']]
        confidences = [p['confidence'] for p in results['predictions']]
        
        axes[1].barh(classes, confidences, color='skyblue')
        axes[1].set_xlabel('置信度')
        axes[1].set_title('预测结果 (Top 5)')
        axes[1].set_xlim([0, 1])
        
        # 在条形上显示数值
        for i, (c, conf) in enumerate(zip(classes, confidences)):
            axes[1].text(conf + 0.02, i, f'{conf:.2%}', va='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {output_path}")
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="预测病虫害")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径 (.h5)')
    parser.add_argument('--image_path', type=str, default=None,
                       help='单张图像路径')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='图像目录（批量预测）')
    parser.add_argument('--class_names_file', type=str, default=None,
                       help='类别名称JSON文件')
    parser.add_argument('--image_size', type=int, default=224,
                       help='输入图像大小')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化预测结果')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 加载类别名称
    class_names = None
    if args.class_names_file and os.path.exists(args.class_names_file):
        with open(args.class_names_file, 'r') as f:
            class_names = json.load(f)
    
    # 初始化预测器
    predictor = PestPredictor(
        model_path=args.model_path,
        class_names=class_names,
        image_size=(args.image_size, args.image_size)
    )
    
    # 单张图像预测
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"错误: 图像文件不存在: {args.image_path}")
            return
        
        print(f"\n预测: {args.image_path}")
        results, _ = predictor.predict(args.image_path)
        
        print("\n预测结果:")
        for pred in results['predictions']:
            print(f"  {pred['class']}: {pred['confidence_percent']}")
        
        if args.visualize:
            predictor.visualize_prediction(args.image_path)
    
    # 批量预测
    elif args.image_dir:
        if not os.path.isdir(args.image_dir):
            print(f"错误: 目录不存在: {args.image_dir}")
            return
        
        print(f"\n批量预测: {args.image_dir}")
        predictor.predict_batch(args.image_dir)
    
    else:
        print("错误: 必须指定 --image_path 或 --image_dir")


if __name__ == "__main__":
    main()
