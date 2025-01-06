# Import necessary libraries
import os
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from hyper_dti.utils.metrics import ci_score
from scripts.dataAcquisition.md_dataPreparation import get_md_regressor

'''****************************初始基本设置****************************'''
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置显示非科学计数法，最多6位小数展示
np.set_printoptions(precision=6, suppress=False)
'''****************************定义回归模型****************************'''
# Define models and hyperparameters
models = {
    'Ridge': Ridge(),
    'DecisionTree': DecisionTreeRegressor(),
    # 'MLP': MLPRegressor(max_iter=1000),
    # 'GradientBoosting': GradientBoostingRegressor(),
    # 'LightGBM': LGBMRegressor(force_col_wise=True),
    # 'Bagging': BaggingRegressor(),
    # 'XGBoost': XGBRegressor(),
    # 'RandomForest': RandomForestRegressor(),
    # 'SVR': svm.SVR()
}
'''****************************数据集获取与处理****************************'''
# 获取已完成处理的数据集
data_path = '/home/cpua212/code/ylyao/MDDTI_YYL/Data/'
datafiles = ['KIBA_md.csv', 'BindingDBIC50_md.csv', 'Lenselink_md.csv']
all_drug_encoder = ['ECFP', 'RDKit', 'MolBert']
all_target_encoder = ['Glove', 'SeqVec', 'UniRep']
dataset = datafiles[2]
data_name = dataset.split(".")[0][:-3]
drug_encoder = all_drug_encoder[2]
target_encoder = all_target_encoder[1]
X_train, X_test, y_train, y_test = get_md_regressor(dataset=dataset, drug_encoder=drug_encoder, target_encoder=target_encoder)

BIOACTIVITY_THRESHOLD = {
    'KIBA': 12.1,
    'Lenselink': 6.5,
    'BindingDBIC50': 6
}

# 展示y值基本情况：中位数、平均数、最值
def y_count(y_raw):
    return np.median(y_raw), np.mean(y_raw), np.max(y_raw), np.min(y_raw)

print(y_count(y_train), y_count(y_test))

y_train -= BIOACTIVITY_THRESHOLD[dataset.split(".")[0][:-3]]
y_test -= BIOACTIVITY_THRESHOLD[dataset.split(".")[0][:-3]]

print('The details of Processed y_value :')
print(y_count(y_train), y_count(y_test))

'''****************************模型训练****************************'''
results_score = {}
METRICS = {
    'regression': {
        'MSE': metrics.mean_squared_error,
        'MAE': metrics.mean_absolute_error,
        'CI': ci_score,
        'r2_score': metrics.r2_score
    }
}
# 计算位于测试集的指标值
def evaluation_metrics(score, y_test, y_test_pred):
    score['RMSE'] = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    for metric, score_func in METRICS['regression'].items():
        test_score = score_func(y_test, y_test_pred)
        score[metric] = test_score
    return score
def distribution_plot(fig, model_name, ith, y_train, y_test, y_train_pred, y_test_pred):
    ax = fig.add_subplot(3, 3, ith)
    if ith % 3 == 1:
        ax.set_ylabel("Predicted Values", fontsize=14, fontweight="bold")
    if ith in [7, 8, 9]:
        ax.set_xlabel("Experimental Values", fontsize=14, fontweight="bold")
    # 红蓝  蓝灰 黄棕
    spot_color = ["#387db8", "#e11a1d", "#CFD4D7", "#557591", "#ffdd14", "#ac592a"]
    line_color = ["#ffdd14", "#ac592a", "#ff5c5c", "#6262ff", "#179b73", "#d48aaf"]
    ax.scatter(y_train, y_train_pred, color=spot_color[3], label="Train", s=9, alpha=0.5)  # 黄棕
    ax.scatter(y_test, y_test_pred, color=spot_color[2], label="Test", s=9, alpha=0.5)
    ax.axhline(y=0, ls=":", c=line_color[0])  # 添加水平直线 红蓝
    ax.axvline(x=0, ls=":", c=line_color[1])  # 添加垂直直线
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_title(model_name, fontsize=16, fontweight="bold")

def run_single(k):
    if k==0:
        fig = plt.figure(1024, figsize=(24, 24))
        ith_plot = 1
    print("="*20+f"The {k}th Regressor Model on {data_name} is beginning"+"="*20)
    # Iterate through models
    for model_name, model in models.items():
        print(f"\n\n")
        print(f"Model: {model_name}")
        score = {}
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        # 分别绘制分布图
        if k==0:
            distribution_plot(fig, model_name, ith_plot, y_train, y_test, y_train_pred, y_test_pred)
            ith_plot += 1
        # 计算并存储评价指标值
        score_temp = evaluation_metrics(score, y_test, y_test_pred)
        results_score[model_name] = score_temp
        print(f'The score for {model_name} is {score_temp}')

    if k == 0:
        fig.suptitle(f'All Regressor_Models on {data_name} ', fontsize=24, fontweight="bold")
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper right', fontsize=12, bbox_to_anchor=(0.9, 0.92))
        # plt.savefig( os.path.join(data_path, f'pictures/{data_name}_{drug_encoder}_{target_encoder}RegressionDistribution.png'), dpi=1024)
        plt.show()
    print("=" * 20 + f"The {k}th Regressor Model on {data_name} is ended" + "=" * 20)
    print("\n\n\n")
    return results_score
# results_single = run_single(0)
# print(results_single)
# results_df = pd.DataFrame.from_dict(results_single)
# print(results_df.T)
# results_df.to_csv(os.path.join(data_path, f'results/{data_name}_{drug_encoder}_{target_encoder}RegressionTaskScores.csv'),
#         sep=',', encoding='utf-8')


def run_multiple(k):
    print(f"The total {k} Experiments about Regression Tasks on {data_name} is beginning")
    print(f"\n\n")
    results = {}
    # 运行k次
    for i in range(k):
        model_score = run_single(i)
        for model, scores in model_score.items():
            if i == 0:
                results[model] = {index: [] for index in scores.keys()}
            else:
                for metric, values in scores.items():
                    if metric not in scores.keys():
                        results[model] = {index: [] for index in scores.keys()}
                    results[model][metric].append(values)

    score_df = {metric: [] for metric in results.keys()}
    for metric, scores in results.items():
        for split in scores.keys():
            print(f"{split}: {scores[split]}")
            mean = np.mean(scores[split])
            std = np.std(scores[split])
            score_df[metric].append(f"{mean:.4f}±{std:.3f}")
    score_df = pd.DataFrame(score_df, index=scores.keys())
    print(score_df.T.to_string())
    # score_df.T.to_csv(
    #     os.path.join(data_path, f'results/{data_name}_{drug_encoder}_{target_encoder}RegressionTaskScores.csv'),
    #     sep=',', encoding='utf-8')
if __name__ == '__main__':
    run_multiple(10)