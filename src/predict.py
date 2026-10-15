"""
新样本预测:输入一名乘客的个人信息,输出 0(遇难)或 1(幸存)。

这里最关键的一点是:预测时用的预处理,必须和训练时完全一样。
所以不是把特征工程再手抄一遍,而是直接从 save/preprocess_params.json
把训练时算好的参数读回来,调用同一个 transform()。
"""

from pathlib import Path

import pandas as pd
import torch

from data_preprocessing import load_params, transform
from model import load_model

ROOT_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT_DIR / 'save'
MODEL_PATH = SAVE_DIR / 'titanic_model.pth'
PARAMS_PATH = SAVE_DIR / 'preprocess_params.json'


def guess_title(sex, age):
    """
    没填姓名的时候,按性别和年龄猜一个称谓,补进姓名里。
    因为称谓是从姓名里正则提的,没有姓名就提不出来,
    而它又是挺重要的一个特征,所以这里给个合理的默认值。
    """
    if sex == 'female':
        return 'Miss' if age < 18 else 'Mrs'
    return 'Master' if age < 15 else 'Mr'


def predict_new_passenger(passenger):
    """
    输入:一个字典,字段和 csv 里一样
          Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Name
          (Name 可以不填)
    输出:(预测结果 0 或 1, 生存概率)
    """
    params = load_params(PARAMS_PATH)

    row = dict(passenger)
    if not row.get('Name'):
        row['Name'] = f"Unknown, {guess_title(row['Sex'], row['Age'])}. Unknown"

    # 拼成一行 DataFrame,再套训练时那套预处理
    df = pd.DataFrame([row])
    X = transform(df, params)

    # 按训练时一样的结构建模型,把参数读进来
    model = load_model(MODEL_PATH, len(params['feature_columns']))
    model.eval()

    with torch.no_grad():
        output = model(torch.tensor(X, dtype=torch.float32))

    prob = output.item()
    prediction = 1 if prob > 0.5 else 0
    return prediction, prob


def run_demo():
    """自动跑一个示例乘客:22 岁的三等舱男性"""
    sample = {
        'Pclass': 3,
        'Sex': 'male',
        'Age': 22,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 7.25,
        'Embarked': 'S',
        'Name': 'Braund, Mr. Owen Harris',
    }
    pred, prob = predict_new_passenger(sample)
    print(f'\n示例乘客(22岁 三等舱 男性):')
    print(f'  预测结果 = {pred} (1=幸存, 0=遇难)   生存概率 = {prob:.2%}')
    return pred, prob


def run_interactive():
    """让用户自己输入一名乘客的信息"""
    print('\n' + '-' * 50)
    print('输入一名乘客的信息来预测:')
    print('-' * 50)
    try:
        sex = input('性别 (male/female): ').strip().lower()
        if sex not in ('male', 'female'):
            print('性别只能填 male 或 female。')
            return

        age = float(input('年龄 (比如 22): '))
        pclass = int(input('舱位等级 (1/2/3): '))
        sibsp = int(input('同行的兄弟姐妹或配偶数 SibSp (没有填 0): '))
        parch = int(input('同行的父母或子女数 Parch (没有填 0): '))
        fare = float(input('票价 Fare (比如 7.25): '))

        embarked = input('登船港口 (C/Q/S): ').strip().upper()
        if embarked not in ('C', 'Q', 'S'):
            print('港口只能填 C、Q 或 S。')
            return

        name = input('姓名(可以留空,留空就按性别年龄猜称谓): ').strip()

        user_sample = {
            'Pclass': pclass, 'Sex': sex, 'Age': age,
            'SibSp': sibsp, 'Parch': parch, 'Fare': fare,
            'Embarked': embarked, 'Name': name,
        }
        pred, prob = predict_new_passenger(user_sample)
        print(f'\n预测结果 = {pred} (1=幸存, 0=遇难)   生存概率 = {prob:.2%}')

    except (ValueError, EOFError):
        # 格式填错了,或者根本不是用键盘交互运行的
        print('输入格式不对,请重新运行,数字的地方填数字。')


if __name__ == '__main__':
    print('=' * 50)
    print('泰坦尼克号生存预测系统')
    print('=' * 50)

    try:
        run_demo()
        run_interactive()
    except FileNotFoundError:
        print('\n找不到模型文件,请先运行 python main.py 或 python train.py 训练模型。')
