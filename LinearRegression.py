# Линейная регрессия
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


def drop(data):
    '''
    Разделяет данные на предиктор и отклик
    '''
    x = pd.DataFrame(data.drop(['X'], axis=1))
    y = pd.DataFrame(data['Y'])
    return x, y


def regression(x, y):
    '''
    Принимает на вход предиктор - x
    И отклик - y
    '''
    test = LinearRegression().fit(x, y)
    print('Значение коэффициента тета_1:')
    print(test.coef_)

    print('Значение коэффициента тета_0:')
    print(test.intercept_)

    print('Значение коэффициента R^2:')
    print(test.score(x, y))


# if __name__ == '__main__':
#     data1 = pd.read_csv('data.csv')
#     regression(drop(data1)[0], drop(data1)[1])
#     print('Cреднее X')
#     print(data1['X'].mean())
#     print('Cреднее Y')
#     print(data1['Y'].mean())


# Линейная многомерная регрессия

df2 = pd.read_csv('candy-data.csv', index_col='competitorname')
train_data = df2.drop(['Charleston Chew', 'Dum Dums'])
X2 = pd.DataFrame(train_data.drop(['winpercent', 'Y'], axis=1))
Y2 = pd.DataFrame(train_data['winpercent'])
reg2 = LinearRegression().fit(X2, Y2)
predict = reg2.predict([[0, 0, 0, 1, 0, 1, 1, 0, 1, 0.885, 0.649]])
print(predict)

candy = df2.loc['Dum Dums', :].to_frame().T
print(reg2.predict(candy.drop(['winpercent', 'Y'], axis=1)))
