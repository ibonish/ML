from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import metrics


def teach(data, *args):
    '''
    Обучение модели:
    Передаем данные и названия кофет
    '''
    train_data = data.drop([i for i in args])
    X = pd.DataFrame(train_data.drop(['winpercent', 'Y'], axis=1))
    Y = pd.DataFrame(train_data['Y'])
    return LogisticRegression(random_state=2019,
                              solver='lbfgs').fit(X, Y.values.ravel())


def probability(probs):
    '''
    Печатает вероятность отнесения к одному из классов
    '''
    print(
        pd.DataFrame({
            '0': [x[0] for x in probs],
            '1': [x[1] for x in probs]
        }, index=test_data.index)
    )


def metr(test_data, y_pred):
    '''
    Вычисляет значения матрицы ошибок
    '''
    Y_true = (test_data['Y'].to_frame().T).values.ravel()
    fpr, tpr, _ = metrics.roc_curve(Y_true, y_pred)
    Y_pred_probs_class_1 = Y_pred_probs[:, 1]
    print(f'TPR {metrics.recall_score(Y_true, y_pred)}')
    print(f'Precision {metrics.precision_score(Y_true, y_pred)}')
    print(f'AUC{metrics.roc_auc_score(Y_true, Y_pred_probs_class_1)}')


# Датасет для обучения
df = pd.read_csv('data.csv', index_col='competitorname')
logreg = teach(df, 'Mike & Ike', 'Root Beer Barrels', 'Starburst')

# Тестовый датасет
test_data = pd.read_csv('data-test.csv',
                        delimiter=',',
                        index_col='competitorname')

X_test = pd.DataFrame(test_data.drop(['Y'], axis=1))
Y_pred = logreg.predict(X_test)
Y_pred_probs = logreg.predict_proba(X_test)

# Вычисление вероятности
probability(Y_pred_probs)

# Вычисление матрицы ошибок
metr(test_data, Y_pred)
