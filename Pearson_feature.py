import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_pearson_correlation(csv_file_path):
    # 读取数据
    df = pd.read_csv(csv_file_path)

    # 只保留感兴趣的列
    # features = ['Depth', 'AC', 'DEN', 'CN', 'GR', 'LLD', 'LLS', 'CAL', 'SP','pp']
    features = ['Depth', 'AC', 'DEN', 'CN', 'GR', 'LLD','pp']
    df = df[features]

    # 处理缺失值（线性插值 + 均值填充）
    df.interpolate(method='linear', inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 计算相关系数矩阵
    corr_matrix = df.corr(method='pearson')

    # 打印与 pp 的相关系数
    print("各特征与压力 (pp) 的 Pearson 相关系数：")
    print(corr_matrix['pp'].drop('pp'))

    # 可视化热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
    plt.title('Pearson Correlation of Features')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == '__main__':
    analyze_pearson_correlation(r"E:\AAAA工作-研一\3济阳凹陷\5超压预测\2-机器学习方法\数据\multi_Eaton_index\原始数据\Resform_export.csv")
