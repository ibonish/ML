# Метод главных компонент
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#

# Уменьшение размерности входных данных
# с минимальными потерями в информативности

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca(x, n_components):
    """
    Вычисление координат относительно главных компонент
    1 аргумент - ваш датасет
    2 аргумент - количество гк
    """

    pca = PCA(n_components=n_components, svd_solver='full')
    data_transformed = pca.fit(x).transform(x)
    return data_transformed


def explained_variance_ratio(y):
    """
    Ввчисление доли объясненной дисперсии
    Аргумент функции - датасет, с новыми координатами
    """

    return y.explained_variance_ratio_


def show(z):
    """Визуализация данных для 2 гк"""

    plt.scatter(z[:len(z), 0], z[:len(z), 1])
    plt.savefig('visual_pca.png')


def get_back(scores, loadings):
    """
    Востанновление данных
    1 аргумент - матрица счетов
    2 агумент матрица весов
    """

    return np.dot(scores, loadings.T)
