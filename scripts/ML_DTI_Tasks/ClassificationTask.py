# Import necessary libraries
import os.path
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import array
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scripts.dataAcquisition.raw_dataPreparation import get_data
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from hyper_dti.settings import constants
import sklearn.metrics as metrics
from hyper_dti.utils.metrics import mcc_score, re05_score, re1_score, re2_score, re5_score
# from lightgbm import LGBMClassifier
from scripts.dataAcquisition.md_dataPreparation import get_md_data
from lightgbm import LGBMClassifier

'''****************************初始基本设置****************************'''
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
'''****************************定义分类模型****************************'''
# Define models and hyperparameters
models = {
    # 'Logistic': LogisticRegression(max_iter=6000, penalty='l2'),#, solver='sag'
    # 'XGBoost': XGBClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'LightGBM': LGBMClassifier(force_col_wise=True),
    # 'AdaBoost': AdaBoostClassifier(),
    # 'Histogram Gradient Boosting': HistGradientBoostingClassifier(),
    # 'Gradient Boosting': GradientBoostingClassifier(),
    # 'Random Forest': RandomForestClassifier(),
    # 'SVMC': svm.SVC(probability=True)
}
# Define hyperparameter search space
param_grids = {
    'LightGBM': {
        "clf__n_estimators": list(range(0, 110, 10)),
        "clf__num_leaves": list(range(5, 60, 10)),
        "clf__learning_rate": [0.05, 0.1, 0.2, 0.3]
        },
    "Random Forest": {
        "clf__n_estimators": list(range(100, 1050, 50)),
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [None, 5, 10, 20],
    },
    'Logistic': {
        "clf__C": [0.01, 0.1, 0.5, 1, 5, 10]
    },
    "Gradient Boosting": {
        "clf__n_estimators": [50, 100, 150, 200],
        "clf__learning_rate": [0.05, 0.1, 0.2, 0.3],
        "clf__max_depth": [3, 5, 7],
    },
    "Histogram Gradient Boosting": {
        "clf__loss": ["log_loss"],
        "clf__learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3],
        "clf__max_depth": [3, 5, 7, 9],
    },
    "AdaBoost": {
        "clf__n_estimators": list(range(100, 1050, 50)),
        "clf__learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3]
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200, 300, 400, 500],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7, 9]
    },
    "Decision Tree": {
        "clf__criterion": ["gini", "entropy"],
        "clf__splitter": ["best", "random"],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": [None, "sqrt", "log2"]
    },
    "SVMC": {
        'clf__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
        'clf__C': np.linspace(0.1, 20, 50),
        'clf__gamma': np.linspace(0.1, 20, 20)
    }
}
'''****************************数据获取与预处理****************************'''
# 获取已处理的数据集，完成最终模型训练与测试验证，
data_path = '/home/cpua212/code/ylyao/MDDTI_YYL/Data/'
datafiles = ['KIBA_md.csv', 'BindingDBIC50_md.csv', 'Lenselink_md.csv']
all_drug_encoder = ['ECFP', 'RDKit', 'MolBert']
all_target_encoder = ['Glove', 'SeqVec', 'UniRep']
dataset = datafiles[2]
data_name = dataset.split(".")[0][:-3]
drug_encoder = all_drug_encoder[2]
target_encoder = all_target_encoder[1]

X_train, X_val, X_test, y_train, y_val, y_test = get_md_data(dataset=dataset, drug_encoder=drug_encoder, target_encoder=target_encoder)

def y_count(y_raw):
    return np.median(y_raw), np.mean(y_raw), np.max(y_raw), np.min(y_raw)

print(y_count(y_val), y_count(y_test))

BIOACTIVITY_THRESHOLD = {
    'KIBA': 12.1,
    'Lenselink': 6.5,
    'BindingDBIC50': 6
}

y_train = (y_train > BIOACTIVITY_THRESHOLD[data_name])
y_val = (y_val > BIOACTIVITY_THRESHOLD[data_name])
y_test = (y_test > BIOACTIVITY_THRESHOLD[data_name])

def y_display(y_raw):
    sns.countplot(x=y_raw, alpha=0.95)
    sns.despine()
    plt.title(f"The Distribution of Binding Affinities")
    plt.ylabel('Bioactivity')
    plt.xlabel('Count of values')
    plt.show()
# y_display(y_train)
y_display(y_val)
y_display(y_test)
'''****************************参数调优****************************'''
best_models = {}
results_dict = {}
# 随机搜索最优参数
# Iterate through models
def hyper_finetuning():

    for model_name, model in models.items():
        print(f"\n\n")
        print(f"Model: {model_name}")
        print("=" * 20 + f"The hyper_finetuning of {model_name} is beginning" + "=" * 20)
        # Create the pipeline with the model
        pipeline = Pipeline([('clf', model)])

        # Perform hyperparameter tuning using RandomizedSearchCV
        param_grid = param_grids.get(model_name, {})
        search_cv = RandomizedSearchCV
        random_search = search_cv(
            pipeline,
            param_grid,
            n_iter=16,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
        )

        # Perform cross-validation on the training and validation data to assess model performance
        cv_scores = cross_val_score(random_search, X_train, y_train, cv=5, scoring='accuracy', error_score='raise')

        # Fit the model on the training and validation data
        random_search.fit(X_val, y_val)

        # Store the best model
        best_models[model_name] = random_search.best_estimator_

        # Store the results in the dictionary
        results_dict[model_name] = (random_search.best_params_, cv_scores)
        print("=" * 20 + f"The hyper_finetuning of {model_name} is ended" + "=" * 20)
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    results_df.to_csv(os.path.join(data_path, f'results/{data_name}_{drug_encoder}_{target_encoder}_BestParam.csv'),
                      sep=',', index=False)

    print(results_dict)
    return results_dict

'''
'''
# KIBA数据集最优参数展示
# results_dict = {
#     'Logistic': ({'clf__C': 0.1}, array([0.81313433, 0.8119403 , 0.82198327, 0.80943847, 0.81541219])),
#     'XGBoost': ({'clf__n_estimators': 100, 'clf__max_depth': 7, 'clf__learning_rate': 0.2}, array([0.88955224, 0.86985075, 0.890681  , 0.87933094, 0.87037037])),
#     'Decision Tree': ({'clf__splitter': 'best', 'clf__min_samples_split': 5, 'clf__min_samples_leaf': 4, 'clf__max_features': 'log2', 'clf__max_depth': 20, 'clf__criterion': 'entropy'}, array([0.83880597, 0.84776119, 0.84289128, 0.83691756, 0.84408602])),
#     'LightGBM': ({'clf__num_leaves': 15, 'clf__n_estimators': 60, 'clf__learning_rate': 0.1}, array([0.88238806, 0.87343284, 0.88530466, 0.87156511, 0.87037037])),
#     'AdaBoost': ({'clf__n_estimators': 550, 'clf__learning_rate': 0.05}, array([0.8161194 , 0.8161194 , 0.81839904, 0.80585424, 0.79868578])),
#     'Histogram Gradient Boosting': ({'clf__max_depth': 9, 'clf__loss': 'log_loss', 'clf__learning_rate': 0.3}, array([0.87522388, 0.86686567, 0.88410992, 0.86260454, 0.85304659])),
#     'Gradient Boosting': ({'clf__n_estimators': 200, 'clf__max_depth': 5, 'clf__learning_rate': 0.1}, array([0.87462687, 0.8561194 , 0.87335723, 0.86798088, 0.86260454])),
#     'Random Forest': ({'clf__n_estimators': 700, 'clf__max_depth': None, 'clf__criterion': 'gini'}, array([0.8680597 , 0.86507463, 0.87694146, 0.86260454, 0.86200717])),
#     'SVC': ({'clf__kernel': 'sigmoid', 'clf__gamma': 4.289473684210526, 'clf__C': 10.253061224489795}, array([0.86149254, 0.8519403 , 0.85304659, 0.85782557, 0.85663082]))
# }
'''****************************迭代训练模型并验证****************************'''
# 最佳模型验证结果
results_score = {}
METRICS = {
    'classification': {
        'AUC': metrics.roc_auc_score,
        'AUPR': metrics.average_precision_score,
        'MCC': mcc_score,
        'Accuracy': accuracy_score
    }
}
def evaluation_metrics(score, y_train, y_test, y_train_pred, y_test_pred):
    for metric, score_func in METRICS['classification'].items():
        if metric == 'MCC':
            train_score, mcc_threshold = score_func(y_train, y_train_pred, threshold=None)
            test_score, _ = score_func(y_test, y_test_pred, threshold=mcc_threshold)
        else:
            # train_score = score_func(y_train, y_train_pred)
            test_score = score_func(y_test, y_test_pred)
        score[metric] = test_score
    return score
def auc_curve_plot(ith, model_name, y_test, y_test_pred, y_test_proba):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba)
    auc = metrics.roc_auc_score(y_test, y_test_pred)
    ax_auc.plot(fpr, tpr, lw=2, color=color_type[ith], linestyle=line_style[ith],
                label=f'{model_name}--(AUC={round(auc,4)})')
    ax_auc.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax_auc.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax_auc.set_xlim([-0.02, 1.05])  # 横竖增加一点长度 以便更好观察图像
    ax_auc.set_ylim([-0.02, 1.05])
    if ith == 8:
        ax_auc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_auc.legend(loc='lower right', fontsize=12)
    ax_auc.set_title(f'The ROC Curve for AUC', fontsize=16, fontweight="bold")
def aupr_curve_plot(ith, model_name, y_test, y_test_pred, y_test_proba):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_test_proba)
    aupr = metrics.average_precision_score(y_test, y_test_pred)
    ax_aupr.plot(recall, precision,  lw=2, color=color_type[ith], linestyle=line_style[ith],
                 label=f'{model_name}--(AUPR={round(aupr,4)})')
    ax_aupr.set_xlabel("Recall", fontsize=14, fontweight="bold")
    ax_aupr.set_ylabel("Precision", fontsize=14, fontweight="bold")
    ax_aupr.set_xlim([-0.02, 1.05])  # 横竖增加一点长度 以便更好观察图像
    ax_aupr.set_ylim([-0.02, 1.05])
    ax_aupr.legend(fontsize=12)
    ax_aupr.set_title(f'The PR Curve for AUPR', fontsize=16, fontweight="bold")
# 绘制曲线画布基本设置
fig = plt.figure(1024, figsize=(20, 10))
# 浅色系
# color_type = ["#F5D9E6", "#DFE1E2", "#CEE2F5", "#CBE0BB", "#F7E5CA", "#5E6C82", "#81B3A9", "#B3C6BB", "#D6CDBE"]
# 深色系
color_type = ["#df7976", "#ef6547", "#d72d35", "#e1a3c6", "#9d77ac", "#c2a389", "#f0bc81", "#37999c", "#3781c2"]
#
# color_type = ["#BCC6DD", "#98A3CA", "#8092C4", "#EFD6D1", "#E6BCB0", "#C89C91", "#F7EABB", "#F2DB96", "#E9CB95"]
# #
line_style = ['-', '--', '-.', '--', ':', 'dotted', '-.', '--', ':']
ax_auc = fig.add_subplot(1, 2, 1)
ax_aupr = fig.add_subplot(1, 2, 2)

def run_single(k):
    if k==0: ith_plot = 0
    print("="*20+f"The {k}th Classification Model on {data_name} is beginning"+"="*20)
    # Iterate through models
    for model_name, (best_params, cv_scores) in results_dict.items():
        # Get the best model for this model_name
        print(f"\n\n")
        print(f"Model: {model_name}")
        score = {}
        best_model = models[model_name]
        best_model.fit(X_train, y_train)
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        if k==0:
            # 分别绘制ROC曲线和PR曲线
            # 分别绘制ROC和PR曲线
            auc_curve_plot(ith_plot, model_name, y_test, y_test_pred, y_test_proba)
            aupr_curve_plot(ith_plot, model_name, y_test, y_test_pred, y_test_proba)
            ith_plot += 1
        # 计算评估值
        score_temp = evaluation_metrics(score, y_train, y_test, y_train_pred, y_test_pred)
        f1_score_test = f1_score(y_test, y_test_pred, average='weighted')
        score_temp['f1_score'] = f1_score_test
        # Append results as a dictionary to the list
        results_score[model_name] = score_temp
        print(f'The score for {model_name} is {score_temp}')

    if k==0:
        fig.suptitle(f'All Classification_Models on {data_name}', fontsize=24, fontweight="bold")
        # plt.tight_layout()
        plt.savefig(os.path.join(data_path, f'pictures/{data_name}_{drug_encoder}_{target_encoder}_CurveScore.png'),
                    dpi=1024)
        plt.show()
    return results_score
# results_single = run_single(0)
# print(results_single)
# results_df = pd.DataFrame.from_dict(results_single)
# print(results_df.T)
# results_df.to_csv(os.path.join(data_path, f'results/{data_name}_{drug_encoder}_{target_encoder}ClassificationScores.csv'),
#         sep=',', encoding='utf-8')
def run_multiple(k):
    print(f"The total {k} Experiments about Classification Tasks on {data_name} is beginning")
    print(f"\n\n")
    results = {}
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
            mean = np.mean(scores[split])
            print(f"{split}: {scores[split]}")
            gap = np.max(scores[split]) - np.min(scores[split])
            std = np.std(scores[split])
            score_df[metric].append(f"{mean:.4f}±{std:.3f}")
    score_df = pd.DataFrame(score_df, index=scores.keys())
    print(score_df.T.to_string())
    # score_df.T.to_csv(os.path.join(data_path, f'results/{data_name}_{drug_encoder}_{target_encoder}_ClassificationTaskResults.csv'),
    #                 sep=',', encoding='utf-8')


if __name__ == '__main__':
    hyper_finetuning()
    run_multiple(10)