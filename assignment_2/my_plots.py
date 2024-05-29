import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    cross_validate, cross_val_score, validation_curve, train_test_split, cross_val_predict, KFold)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

cv = 5
random_state = 42

depths = range(1, 15)
len_depths = len(depths)


def task_plot_2(X_train, y_train, X_test=None, y_test=None, label=str):
    mean_scores = np.zeros(len_depths, dtype=float)
    std_scores = np.zeros(len_depths, dtype=float)

    mean_errors = np.zeros(len_depths, dtype=float)
    std_errors = np.zeros(len_depths, dtype=float)

    for i, depth in enumerate(depths):
        model = DecisionTreeRegressor(max_depth=depth, random_state=random_state)

        if X_test is not None and y_test is not None:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mean_errors[i] = np.mean(predictions - y_test)
            std_errors[i] = np.std(predictions - y_test)
            scores = mean_squared_error(y_test, predictions)
        else:
            predictions = cross_val_predict(model, X_train, y_train, cv=cv)
            mean_errors[i] = np.mean(predictions - y_train)
            std_errors[i] = np.std(predictions - y_train)
            scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

        mean_scores[i] = scores.mean()
        std_scores[i] = scores.std()

    optimal_score = np.min(mean_scores)
    optimal_depth = np.argmin(mean_scores) + 1
    print(f"Optimal score: {optimal_score}")
    print(f"Optimal depth: {optimal_depth}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(depths, mean_scores, marker='o', zorder=1)
    axs[0].scatter(optimal_depth, optimal_score,
                   color='red', label='Optimal Depth', marker='X', zorder=2)
    axs[0].fill_between(depths,
                        mean_scores - 2 * std_scores,
                        mean_scores + 2 * std_scores,
                        alpha=0.2, label='Mean Score +/- 2 Std. Dev.')
    axs[0].set_xlabel('Max Depth of Tree')
    axs[0].set_ylabel('Cross-Validated MSE')
    axs[0].set_title(f'Max Depth vs Cross-Validated MSE ({label})')
    axs[0].legend()

    # Второй график - Mean Error +/- Std. Dev. на тестовой выборке
    axs[1].errorbar(depths, mean_errors, yerr=std_errors, fmt='o-', label='Mean Error +/- Std. Dev.',
                    zorder=1)
    axs[1].scatter(optimal_depth, mean_errors[optimal_depth - 1],
                   color='red', label='Optimal Depth', marker='X', zorder=2)
    axs[1].set_xlabel('Max Depth of Tree')
    axs[1].set_ylabel('Mean Error')
    axs[1].set_title(f'Max Depth vs Mean Error ({label})')
    axs[1].legend()

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return optimal_score, optimal_depth


def task_plot_3(X_train, y_train, X_test, y_test, depth):
    tree = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    mse = mean_squared_error(y_pred, y_test)
    print(f"MSE: {mse}")

    plt.plot(y_test, 'bo')
    plt.plot(y_pred - mse, 'ro', alpha=0.2)
    plt.plot(y_pred + mse, 'ro', alpha=0.2)
    plt.title('DecisionTree Predictions')
    plt.grid(True)
    plt.show()

    return mse


def task_plot_4(X, y, n_estimators=3, bagging: bool = False):

    r2_scores = np.zeros((len_depths, cv), dtype=float)

    for i, depth in enumerate(depths):
        model = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
        if bagging:
            model = BaggingRegressor(model, n_estimators=n_estimators, random_state=random_state)
        score = cross_val_score(model, X, y, cv=cv, scoring='r2')
        r2_scores[i] = score

    for i, r2 in enumerate(r2_scores):
        print(f"Tree {i + 1}: Mean R^2 = {np.mean(r2):.4f}, Variance R^2 = {np.var(r2):.4f}")
    print(f"Average R^2 across all trees: {r2_scores.mean():.4f}")

    plt.boxplot(r2_scores.T)
    plt.xlabel('Tree depth')
    plt.ylabel('R^2 Score')
    plt.title('Distribution of R^2 Scores for Bagging Trees')
    plt.grid(True)
    plt.show()


def task_plot_5(X, y, depth):
    n_estimators = range(1, 501, 50)
    len_n_estimators = len(n_estimators)
    biases = np.zeros(len_n_estimators, dtype=float)
    variances = np.zeros(len_n_estimators, dtype=float)

    # kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    for i, n in enumerate(n_estimators):
        tree = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
        bagging_reg = BaggingRegressor(tree, n_estimators=n, random_state=random_state, n_jobs=-1)
        y_pred = cross_val_predict(bagging_reg, X, y, cv=cv, n_jobs=-1)
        biases[i] = np.mean(np.mean((y - y_pred) ** 2))
        variances[i] = np.mean(np.var(y_pred))

    print(f"Minimum Variance: {np.min(variances)}")
    print(f"Number of Trees: {n_estimators[np.argmin(variances)]}")
    print(f"Minimum Bias: {np.min(biases)}")
    print(f"Number of Trees: {n_estimators[np.argmin(biases)]}")

    # Построение графиков
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(n_estimators, biases, label='Bias', color='blue')
    plt.xlabel('Number of Trees')
    plt.ylabel('Bias')
    plt.title('Bias vs Number of Trees')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(n_estimators, variances, label='Variance', color='red')
    plt.xlabel('Number of Trees')
    plt.ylabel('Variance')
    plt.title('Variance vs Number of Trees')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
