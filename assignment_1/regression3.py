import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin


class LinearRegression3(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.w_opt = None
        self.mu_list = np.linspace(-1, 1, 5)
        self.sigma = (self.mu_list[1] - self.mu_list[0])

    def fit(self, X, y):
        self.w_opt = self.find_optimal_weights(X, y)
        return self

    def predict(self, X):
        Phi = self.design_matrix(X)
        y_pred = Phi.dot(self.w_opt)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return -np.mean((y - y_pred) ** 2)

    def gaussian_basis_function(self, x, mu):
        return np.exp(-0.5 * ((x - mu) / self.sigma) ** 2)

    # Создаем матрицу базисных функций для разных mu
    def design_matrix(self, X):
        n_samples, n_features = X.shape
        n_basis = len(self.mu_list)
        phi = np.zeros((n_samples, n_basis * n_features))

        for i in range(n_samples):
            for j in range(n_features):
                for k, mu in enumerate(self.mu_list):
                    phi[i, j * n_basis + k] = self.gaussian_basis_function(X[i, j], mu)

        return phi

    # Определяем функцию линейной регрессии
    def find_optimal_weights(self, X, y):
        # Создаем матрицу базисных функций
        phi = self.design_matrix(X)

        # Инициализируем веса случайными значениями
        w_init = np.random.randn(phi.shape[1])

        # Определяем функцию потерь (среднеквадратичная ошибка)
        def loss(w):
            y_pred = phi.dot(w)
            return 0.5 * np.mean((y - y_pred) ** 2)  # Умножаем MSE на 0.5 для упрощения вычисления градиента.

        # Минимизируем функцию потерь
        result = minimize(loss, w_init)
        return result.x
