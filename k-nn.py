import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def euclidean_metric(data, value, k):
    '''
    Евклидова метрикa.
    Принимает на вход данные, координаты точки и k.
    Возвращает расстояние и айди ближайших соседей,
    а также классифицирует
    '''
    x = pd.DataFrame(data.drop(['Class'], axis=1))
    y = pd.DataFrame(data['Class']).values.ravel()

    euclidean = KNeighborsClassifier(n_neighbors=k, p=2)
    euclidean.fit(x, y)
    a = euclidean.kneighbors([value])
    b = euclidean.predict([value])
    return a, b


def manhattan(data, value, k):
    '''
    Манхеттенское расстояние.
    Принимает на вход данные, координаты точки и k.
    Возвращает расстояние и айди ближайших соседей,
    а также классифицирует
    '''
    x = pd.DataFrame(data.drop(['Class'], axis=1))
    y = pd.DataFrame(data['Class']).values.ravel()

    manhattan = KNeighborsClassifier(n_neighbors=k, p=1)
    manhattan.fit(x, y)
    a = manhattan.kneighbors([value])
    b = manhattan.predict([value])
    return a, b
