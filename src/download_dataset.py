"""
数据集下载脚本 (Dataset Download Script)
功能: 从Kaggle下载Plant Village或Plant Diseases数据集
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


class DatasetDownloader:
    """数据集下载器"""
    
    @staticmethod
    def download_file(url, filename, chunk_size=8192):
        """
        下载文件
        
        Args:
            url: 文件URL
            filename: 保存的文件名
            chunk_size: 块大小
        """
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    @staticmethod
    def extract_zip(zip_path, extract_to='.'):
        """
        解压ZIP文件
        
        Args:
            zip_path: ZIP文件路径
            extract_to: 解压目标目录
        """
        print(f"解压 {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("解压完成")
    
    @staticmethod
    def download_from_kaggle(dataset_name, api_key_file=None):
        """
        使用Kaggle API下载数据集
        
        Args:
            dataset_name: Kaggle数据集名称
            api_key_file: Kaggle API密钥文件位置
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # 初始化API
            api = KaggleApi()
            api.authenticate()
            
            print(f"从Kaggle下载: {dataset_name}")
            api.dataset_download_files(dataset_name, path='data/raw', unzip=True)
            print("下载完成!")
            
        except ImportError:
            print("需要安装 kaggle: pip install kaggle")
            print("然后从 https://www.kaggle.com/settings/account 下载API密钥")
        except Exception as e:
            print(f"下载失败: {e}")
    
    @staticmethod
    def setup_data_structure():
        """设置数据目录结构"""
        dirs = [
            'data/raw',
            'data/processed/train',
            'data/processed/val',
            'data/processed/test',
            'models',
            'logs',
            'results'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✓ 创建目录: {dir_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="下载病虫害识别数据集")
    parser.add_argument('--dataset', type=str, 
                       choices=['plant_village', 'plant_diseases', 'new_plant_diseases'],
                       default='plant_diseases',
                       help='数据集选择')
    parser.add_argument('--setup_only', action='store_true',
                       help='仅创建目录结构，不下载数据')
    
    args = parser.parse_args()
    
    # 创建目录结构
    print("设置目录结构...")
    DatasetDownloader.setup_data_structure()
    
    if args.setup_only:
        print("\n目录结构创建完成!")
        print("接下来可以:")
        print("1. 手动将数据集放到 data/raw/ 目录")
        print("2. 运行 python src/data_preprocessing.py 进行数据预处理")
        return
    
    # 下载数据集
    print("\n下载数据集...")
    
    if args.dataset == 'plant_village':
        dataset_name = 'spmohanty/plant-village-dataset'
    elif args.dataset == 'plant_diseases':
        dataset_name = 'vipoooool/new-plant-diseases-dataset'
    else:
        dataset_name = 'vipoooool/new-plant-diseases-dataset'
    
    DatasetDownloader.download_from_kaggle(dataset_name)
    
    print("\n数据下载完成!")
    print("接下来运行: python src/data_preprocessing.py")


if __name__ == "__main__":
    main()
