"""
数据读取、清洗、特征工程。

这里把预处理拆成了两步:
  fit_preprocessor() 只在训练集上算参数(中位数、众数、均值方差这些)
  transform()        拿着这套参数去处理数据,训练集、测试集、新样本都用它

为什么要这么拆:如果先拿全部数据填完缺失值再划分,测试集的信息就漏进训练集了。
题目 3.2 明确要求「预处理参数只能根据训练数据确定」,所以顺序必须是
先划分 -> 只在训练集上算参数 -> 两边套同一套参数。
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 随机种子,固定住数据划分的结果
SEED = 42

# 最后真正喂给模型的 10 个特征。顺序在这里写死,
# 这样训练和预测的时候列的顺序一定对得上。
FEATURE_COLUMNS = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
    'Fare', 'Embarked_Q', 'Embarked_S', 'Title', 'FamilySize',
]

# 称谓 -> 数字。全是男的 Mr、年轻男的 Master、女的 Miss/Mrs,剩下少见的都归到 Rare
TITLE_MAPPING = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

# 归到 Rare 的那些少见称谓
RARE_TITLES = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']


def load_raw_data(csv_path):
    """读原始 csv,返回 DataFrame"""
    return pd.read_csv(csv_path)


def _extract_title(name_series):
    """
    从姓名里把称谓抠出来。
    比如 'Braund, Mr. Owen Harris' -> 'Mr','Heikkinen, Miss. Laina' -> 'Miss'
    正则的意思:一个空格 + 一串英文字母 + 一个点
    """
    return name_series.str.extract(r' ([A-Za-z]+)\.', expand=False)


def fit_preprocessor(train_df):
    """
    只在训练集上算预处理要用到的参数,算完返回一个字典。

    注意这里传进来的只能是训练集,不能是整个数据集。
    """
    params = {}

    # 1. 缺失值填什么:年龄填中位数(不容易被极端值带偏),登船港口填众数
    params['age_median'] = float(train_df['Age'].median())
    params['embarked_mode'] = str(train_df['Embarked'].mode()[0])

    # 2. 标准化要用到的均值和标准差
    params['age_mean'] = float(train_df['Age'].mean())
    params['age_std'] = float(train_df['Age'].std())
    params['fare_mean'] = float(train_df['Fare'].mean())
    params['fare_std'] = float(train_df['Fare'].std())

    # 3. 称谓映射表(本来是写死的,这里也存一份,以后想改就改这里)
    params['title_mapping'] = TITLE_MAPPING

    # 4. 特征列的顺序
    params['feature_columns'] = FEATURE_COLUMNS

    return params


def transform(df, params):
    """
    用 fit_preprocessor() 算好的参数处理数据,返回 10 列特征的数组。
    训练集、测试集、新样本全都走这个函数,保证处理方式完全一样。
    """
    df = df.copy()

    # --- 1. 处理缺失值 ---
    df['Age'] = df['Age'].fillna(params['age_median'])
    df['Embarked'] = df['Embarked'].fillna(params['embarked_mode'])
    # 客舱号缺失太多(77%),这一列直接不要了
    if 'Cabin' in df.columns:
        df = df.drop(columns=['Cabin'])

    # --- 2. 性别:文字转成数字,模型只认数字。male=0,female=1 ---
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # --- 3. 登船港口:独热编码 ---
    # C / Q / S 三个港口,用 drop_first=True 把 C 去掉当基准,只留 Q 和 S 两列。
    # 这样做是避免三列加起来恒等于 1(共线性),也顺便少一个特征。
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    # 如果这一批数据里正好没有某个港口,get_dummies 就不会生成那一列,
    # 这里补上,免得列数对不上
    for col in ['Embarked_Q', 'Embarked_S']:
        if col not in df.columns:
            df[col] = 0
    df['Embarked_Q'] = df['Embarked_Q'].astype(int)
    df['Embarked_S'] = df['Embarked_S'].astype(int)

    # --- 4. 从姓名里提取称谓 ---
    df['Title'] = _extract_title(df['Name'])
    df['Title'] = df['Title'].replace(RARE_TITLES, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map(params['title_mapping'])
    # 万一冒出个没见过的称谓,map 完会是空的,统一填 0
    df['Title'] = df['Title'].fillna(0).astype(int)

    # --- 5. 家庭人数 = 兄弟姐妹配偶数 + 父母子女数 + 自己 ---
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # --- 6. 标准化 ---
    # 年龄是 0~80,票价是 0~512,量级差太多,直接喂进网络会让票价这个特征
    # 独占风头。这里减均值除标准差,把两个都拉到差不多的范围。
    # 用的是训练集的均值和标准差,测试集和新样本套同一个。
    df['Age'] = (df['Age'] - params['age_mean']) / params['age_std']
    df['Fare'] = (df['Fare'] - params['fare_mean']) / params['fare_std']

    # --- 7. 只留这 10 列,顺序按 FEATURE_COLUMNS 来 ---
    df = df[params['feature_columns']]

    return df.astype(np.float32).values


def load_and_split(csv_path, test_size=0.2):
    """
    一步到位:读数据 -> 划分训练集测试集 -> 只在训练集上算参数 -> 两边都套同一套参数。

    返回 X_train, y_train, X_test, y_test, params
    注意顺序:先划分再算参数,不能反过来。
    """
    df = load_raw_data(csv_path)

    # 先划分
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=SEED)

    # 参数只看训练集
    params = fit_preprocessor(train_df)

    # 训练集和测试集走同一个 transform
    X_train = transform(train_df, params)
    y_train = train_df['Survived'].values
    X_test = transform(test_df, params)
    y_test = test_df['Survived'].values

    return X_train, y_train, X_test, y_test, params


def save_params(params, path):
    """把预处理参数存成 json,预测的时候再读回来"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def load_params(path):
    """读回预处理参数"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
