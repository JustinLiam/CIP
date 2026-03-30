import os
import pandas as pd
import numpy as np

def process_mse_files():
    # 遍历所有子目录
    for root, dirs, files in os.walk('.'):
        if 'mse.csv' in files:
            csv_path = os.path.join(root, 'mse.csv')
            print(f"Processing: {csv_path}")
            
            # 读取CSV
            df = pd.read_csv(csv_path)
            
            # 分离seed列和数据列
            data_cols = df.drop('seed', axis=1)
            
            # 计算mean和std
            means = data_cols.mean().round(3)
            stds = data_cols.std().round(3)
            
            # 创建和原始mse.csv相同格式的DataFrame
            results = pd.DataFrame({
                'seed': ['mean', 'std'],
                **{col: [f"{mean:.3f}", f"{std:.3f}"] 
                   for col, (mean, std) in zip(data_cols.columns, zip(means, stds))}
            })
            
            # 保存结果
            output_path = os.path.join(root, 'mse_stats.csv')
            results.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
            
            # 打印结果
            print("\nResults:")
            for col in data_cols.columns:
                print(f"{col}: {means[col]:.3f}±{stds[col]:.3f}")
            print("\n" + "="*50 + "\n")

        if 'out_mse.csv' in files:
            csv_path = os.path.join(root, 'out_mse.csv')
            print(f"Processing: {csv_path}")
            
            # 读取CSV
            df = pd.read_csv(csv_path)
            
            # 分离seed列和数据列
            data_cols = df.drop('seed', axis=1)
            
            # 计算mean和std
            means = data_cols.mean().round(3)
            stds = data_cols.std().round(3)
            
            # 创建和原始mse.csv相同格式的DataFrame
            results = pd.DataFrame({
                'seed': ['mean', 'std'],
                **{col: [f"{mean:.3f}", f"{std:.3f}"] 
                   for col, (mean, std) in zip(data_cols.columns, zip(means, stds))}
            })
            
            # 保存结果
            output_path = os.path.join(root, 'out_mse_stats.csv')
            results.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
            
            # 打印结果
            print("\nResults:")
            for col in data_cols.columns:
                print(f"{col}: {means[col]:.3f}±{stds[col]:.3f}")
            print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    process_mse_files()