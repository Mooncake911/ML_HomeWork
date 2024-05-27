import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score


def find_optimal_degree(x, y, model, cv: int = 10, max_degree: int = 1):
    _degrees = np.arange(1, max_degree + 1)
    _mean_bias = []
    _mean_variance = []
    _total = {}

    for degree in _degrees:
        _polynomial_features = PolynomialFeatures(degree=degree, include_bias=True)
        _x_poly = _polynomial_features.fit_transform(x)
        _model = model
        _model.fit(_x_poly, y)

        _scores = cross_val_score(_model, _x_poly, y, cv=cv, scoring='neg_mean_squared_error')
        _mean_squared_errors = -_scores

        _mean_bias.append(_mean_squared_errors.mean())
        _mean_variance.append(_mean_squared_errors.var())
        _total[degree] = _mean_squared_errors.mean() + _mean_squared_errors.var()

    _optimal_degree = min(_total, key=_total.get)
    print(f'Optimal degree with cross-validation: {_optimal_degree}')
    print(f"Error: {_total[_optimal_degree]}")

    plt.figure(figsize=(10, 6))
    plt.plot(_degrees, _mean_bias, label="Bias^2", marker='o', linestyle='--')
    plt.plot(_degrees, _mean_variance, label="Variance", marker='o', linestyle='--')
    plt.xlabel("Degree of Polynomial")
    plt.ylabel("Error")
    plt.title("Bias-Variance Trade-off")
    plt.legend()
    plt.show()

    return _optimal_degree


def draw_scatter_plot(X, y, y_pred):
    plt.scatter(X[:, 0], y, label='Data', alpha=0.5)
    plt.scatter(X[:, 0], y_pred, label='Prediction', color='red', alpha=0.5)
    plt.legend()
    plt.show()