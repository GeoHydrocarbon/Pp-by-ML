import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=["Depth", "pp"])
    y = df["pp"]
    depth = df["Depth"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    depth_train = depth.iloc[X_train.index]
    depth_test = depth.iloc[X_test.index]
    return X_train, X_test, y_train, y_test, depth_train, depth_test


def tune_model(pipe, param_distributions, X_train, y_train):
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=10,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_models(X_train, X_test, y_train, y_test, model_save_path):
    os.makedirs(model_save_path, exist_ok=True)

    models = {
        "DecisionTree": (DecisionTreeRegressor(random_state=42), {
            'model__max_depth': [5, 10, 20, 30, 40, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }),
        "RandomForest": (RandomForestRegressor(random_state=42), {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [10, 20, 30, None],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        }),
        "XGBoost": (XGBRegressor(objective='reg:squarederror', random_state=42), {
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 6, 9],
            'model__subsample': [0.8, 1.0],
            'model__n_estimators': [100, 200]
        }),
        "CatBoost": (CatBoostRegressor(verbose=0, random_state=42), {
            'model__depth': [4, 6, 8],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__iterations': [100, 200]
        }),
        "MLP": (MLPRegressor(max_iter=2000, random_state=42), {
            'model__hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__learning_rate_init': [0.001, 0.01]
        }),
        "SVR": (SVR(), {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['rbf', 'linear'],
            'model__gamma': ['scale', 'auto']
        })
    }

    results = {}
    all_predictions = {}

    for name, (model, param_grid) in models.items():
        print(f"Training and tuning {name}...")

        # 标准化管道（对所有模型都应用，非线性模型无影响）
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        try:
            best_model, best_params = tune_model(pipeline, param_grid, X_train, y_train)
        except Exception as e:
            print(f"{name} 调参失败: {e}")
            continue

        y_pred = best_model.predict(X_test)
        plot_three_panel(y_test, y_pred, best_model.named_steps['model'], X_test, name, model_save_path)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results[name] = {"MSE": mse, "R2": r2, "MAE": mae, "Best Params": best_params}
        all_predictions[name] = y_pred

        joblib.dump(best_model, os.path.join(model_save_path, f"{name}.pkl"), compress=3)

    # 保存最佳参数到CSV文件
    best_params_df = pd.DataFrame()
    for model_name, metrics in results.items():
        best_params = metrics['Best Params']
        # 将参数字典转换为DataFrame格式
        params_series = pd.Series(best_params)
        params_series.name = model_name
        best_params_df = pd.concat([best_params_df, params_series], axis=1)
    
    # 转置DataFrame以便更好地查看
    best_params_df = best_params_df.T
    best_params_df.to_csv(os.path.join(model_save_path, 'best_model_parameters.csv'))
    
    # 保存MSE、R2和MAE指标到CSV文件
    performance_df = pd.DataFrame()
    for model_name, metrics in results.items():
        performance_series = pd.Series({
            'MSE': metrics['MSE'],
            'R2': metrics['R2'],
            'MAE': metrics['MAE']
        })
        performance_series.name = model_name
        performance_df = pd.concat([performance_df, performance_series], axis=1)
    
    # 转置DataFrame以便更好地查看
    performance_df = performance_df.T
    performance_df.to_csv(os.path.join(model_save_path, 'model_performance_metrics.csv'))
    
    return results, all_predictions, y_test

def cross_val_r2_plot(models_dict, X, y, model_save_path):
    cv_scores = {}
    
    # 创建保存交叉验证图的目录
    cv_plots_dir = os.path.join(model_save_path, 'cross_validation_plots')
    os.makedirs(cv_plots_dir, exist_ok=True)
    
    for name, pipeline in models_dict.items():
        try:
            if "CatBoost" in name:
                model = pipeline.named_steps['model']
                scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
            else:
                scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)

            cv_scores[name] = scores
            print(f"{name} CV R²: {scores.mean():.4f} ± {scores.std():.4f}")
            
            # 为每个模型绘制每一折的交叉验证图
            plot_cv_folds(pipeline, X, y, name, cv_plots_dir)
            
        except Exception as e:
            print(f"{name} 交叉验证失败: {e}")

    # 保存交叉验证结果到CSV
    cv_df = pd.DataFrame(cv_scores)
    cv_df.to_csv(os.path.join(model_save_path, 'cross_validation_results.csv'), index=False)
    
    # 保存交叉验证统计信息
    cv_stats = {}
    for name, scores in cv_scores.items():
        cv_stats[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
    cv_stats_df = pd.DataFrame(cv_stats).T
    cv_stats_df.to_csv(os.path.join(model_save_path, 'cross_validation_stats.csv'))

    # 转换为 DataFrame 方便绘图
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cv_scores.items()]))
    sns.boxplot(data=df)
    plt.ylabel("Cross-Validated R² Score")
    # plt.title("模型交叉验证 R² 分布对比")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'cross_validation_boxplot.png'), dpi=300, bbox_inches='tight')
    # plt.show()


def plot_results(results, model_save_path):
    models = list(results.keys())
    mse_values = [results[model]["MSE"] for model in models]
    r2_values = [results[model]["R2"] for model in models]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(models, mse_values, color='blue', alpha=0.6, label='MSE')
    ax2.plot(models, r2_values, color='red', marker='o', linestyle='-', linewidth=2, label='R2')

    ax1.set_xlabel("Models")
    ax1.set_ylabel("MSE", color='blue')
    ax2.set_ylabel("R2", color='red')
    ax1.set_title("Model Performance Comparison")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.savefig(os.path.join(model_save_path, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_three_panel(y_true, y_pred, model, X_test, model_name, save_dir):
    feature_names = X_test.columns if hasattr(X_test, 'columns') else [f"F{i}" for i in range(X_test.shape[1])]
    plt.figure(figsize=(18, 5))

    # 1. Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r-', lw=2)
    r2 = r2_score(y_true, y_pred)
    plt.title(f"Actual vs Predicted (R²={r2:.2f})")
    plt.xlabel("Actual Pressure")
    plt.ylabel("Predicted Pressure")

    # 2. Residual distribution
    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, bins=50, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")

    # 3. Feature Importance
    plt.subplot(1, 3, 3)
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    if importances is not None and len(importances) == len(feature_names):
        indices = np.argsort(importances)[::-1]
        plt.barh(np.array(feature_names)[indices], importances[indices])
        plt.title("Feature Importances")
        plt.gca().invert_yaxis()
    else:
        plt.text(0.5, 0.5, f"{model_name} failed", ha='center', va='center')
        plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_result.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_all_predictions(y_test, all_predictions, depth_test, model_save_path):
    # 整合为 DataFrame 以便排序
    df = pd.DataFrame({
        'Depth': depth_test.values,
        'True': y_test.values
    })

    for model_name, y_pred in all_predictions.items():
        df[model_name] = y_pred

    # 按深度升序排序
    df = df.sort_values(by='Depth')

    # 为每个模型创建单独的子图，横向排列
    n_models = len(all_predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(3*n_models, 8), sharey=False)  # 改为False
    
    if n_models == 1:
        axes = [axes]

    # 实测压力点
    measured_points = [(1508.35, 14.67), (1563.45, 15.12), (1701.8, 16.5), (1886.95, 17.4), (2184.1, 22.28), (2305.55, 21.86), (2397.5, 22.88), (2802, 29.98), (2993.4, 37.14), (3013.125, 31.32), (3016.625, 38.7), (3016.75, 31.3), (3019.125, 36.75), (3025.125, 28.43), (3045.875, 37.14), (3049.75, 32.16), (3092.125, 39.57), (3103.125, 33.81), (3109.5, 44.4), (3110.875, 36.94), (3126.375, 35.58), (3127.375, 48.76), (3154.875, 46.83), (3205.875, 45.53), (3221.25, 44.6), (3249.625, 48.61), (3273.875, 48.95), (3286.875, 52.89), (3389, 59.38), (3394.875, 48.49), (3504.4, 60.74)]  # (深度, 压力)
    
    for i, (model_name, y_pred) in enumerate(all_predictions.items()):
        ax = axes[i]
        ax.plot(df['True'], df['Depth'], label='Actual Pressure', color='blue', linewidth=2)
        ax.plot(df[model_name], df['Depth'], label=f'{model_name} Predicted', color='red', linewidth=1)
        
        # 绘制实测压力点
        for depth, pressure in measured_points:
            ax.scatter(pressure, depth, color='green', s=100, marker='o', 
                      label='Measured Points' if depth == measured_points[0][0] else "", zorder=5)
        
        ax.set_xlabel('Pressure (MPa)')
        if i == 0:
            ax.set_ylabel('Depth (m)')
        ax.set_title(f'{model_name}')
        ax.legend()
        # 确保y轴反转并设置范围
        ax.invert_yaxis()
        ax.set_ylim(df['Depth'].max(), df['Depth'].min())  # 明确设置y轴范围
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'all_models_predictions_comparison.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    # 添加验证集实测压力与预测压力的交汇图
    plot_validation_scatter(y_test, all_predictions, model_save_path)


def plot_cv_folds(pipeline, X, y, model_name, save_dir):
    """为每个模型绘制每一折交叉验证的训练集和验证集真实值vs预测值图"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 创建2行3列的子图布局（5折 + 1个汇总图）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练模型
        if "CatBoost" in model_name:
            # CatBoost需要特殊处理
            model = pipeline.named_steps['model']
            scaler = pipeline.named_steps['scaler']
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            model.fit(X_train_scaled, y_train_fold)
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)
        else:
            pipeline.fit(X_train_fold, y_train_fold)
            y_train_pred = pipeline.predict(X_train_fold)
            y_val_pred = pipeline.predict(X_val_fold)
        
        # 计算R²分数
        train_r2 = r2_score(y_train_fold, y_train_pred)
        val_r2 = r2_score(y_val_fold, y_val_pred)
        fold_scores.append(val_r2)
        
        # 绘制当前折的图
        ax = axes[fold]
        
        # 训练集散点图
        ax.scatter(y_train_fold, y_train_pred, alpha=0.6, s=30, color='blue', label=f'Train (R²={train_r2:.3f})')
        
        # 验证集散点图
        ax.scatter(y_val_fold, y_val_pred, alpha=0.8, s=50, color='red', label=f'Val (R²={val_r2:.3f})')
        
        # 绘制对角线
        min_val = min(y_train_fold.min(), y_val_fold.min(), y_train_pred.min(), y_val_pred.min())
        max_val = max(y_train_fold.max(), y_val_fold.max(), y_train_pred.max(), y_val_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_name} - Fold {fold+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴范围一致
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
    
    # 绘制汇总图（第6个子图）
    ax_summary = axes[5]
    ax_summary.plot(range(1, 6), fold_scores, 'o-', linewidth=2, markersize=8)
    ax_summary.set_xlabel('Fold')
    ax_summary.set_ylabel('R² Score')
    ax_summary.set_title(f'{model_name} - CV R² Scores')
    ax_summary.grid(True, alpha=0.3)
    ax_summary.set_xticks(range(1, 6))
    
    # 添加平均分数线
    mean_score = np.mean(fold_scores)
    ax_summary.axhline(y=mean_score, color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean R² = {mean_score:.3f}')
    ax_summary.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_cv_folds.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{model_name} 交叉验证图已保存到: {save_path}")


def plot_validation_scatter(y_test, all_predictions, model_save_path):
    """绘制验证集实测压力与预测压力的交汇图"""
    n_models = len(all_predictions)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (model_name, y_pred) in enumerate(all_predictions.items()):
        if i < len(axes):
            ax = axes[i]
            
            # 绘制散点图
            ax.scatter(y_test, y_pred, alpha=0.7, s=50)
            
            # 绘制对角线（理想预测线）
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # 计算R²和MAE
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            ax.set_xlabel('Measured Pressure (MPa)')
            ax.set_ylabel('Predicted Pressure (MPa)')
            ax.set_title(f'{model_name}\nR² = {r2:.3f}, MAE = {mae:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 设置坐标轴范围一致
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
    
    # 隐藏多余的子图
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'validation_scatter_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    file_path = r"E:\AAAA工作-研一\3济阳凹陷\5超压预测\2-机器学习方法\数据\multi_Eaton_index\原始数据\Resform_export.csv"
    model_save_path = r"E:\AAAA工作-研一\3济阳凹陷\5超压预测\2-机器学习方法\数据\multi_Eaton_index\第三次训练-密度不变"

    X_train, X_test, y_train, y_test, depth_train, depth_test = load_data(file_path)
    results, all_predictions, y_test = train_models(X_train, X_test, y_train, y_test, model_save_path)

    print("Model training results:")
    for model, metrics in results.items():
        print(f"{model}: MSE={metrics['MSE']:.4f}, R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, Best Params={metrics['Best Params']}")

    plot_results(results, model_save_path)
    plot_all_predictions(y_test, all_predictions, depth_test, model_save_path)

    # 交叉验证 R² 箱线图
    # 创建与 train_models 相同模型结构（不调参，只用于评估）
    raw_models = {
        "DecisionTree": Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeRegressor(random_state=42))]),
        "RandomForest": Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=42))]),
        "XGBoost": Pipeline([("scaler", StandardScaler()), ("model", XGBRegressor(objective='reg:squarederror', random_state=42))]),
        "CatBoost": Pipeline([("scaler", StandardScaler()), ("model", CatBoostRegressor(verbose=0, random_state=42))]),
        "MLP": Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(max_iter=2000, random_state=42))]),
        "SVR": Pipeline([("scaler", StandardScaler()), ("model", SVR())])
    }

    cross_val_r2_plot(raw_models, X_train, y_train, model_save_path)