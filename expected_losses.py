import sys
from typing import Iterable, NoReturn

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor


sys.path.append(".")
from loss_prediction.target import create_next_day_price
from probability_prediction.target import create_event_happened_day
from utils.dataset import create_arma_table


def print_scores(y_true: Iterable, y_predict: Iterable, prefix: str = "") -> NoReturn:
    mae = mean_absolute_error(y_true, y_predict)
    mse = mean_squared_error(y_true, y_predict)
    r2 = r2_score(y_true, y_predict)
    f1 = f1_score((y_true < 0), (y_predict < 0))
    acc = accuracy_score((y_true < 0), (y_predict < 0))
    print(f"{prefix} MAE: {mae:4f} MSE: {mse:.4f}, R2-score: {r2:.4f}, F1-score: {f1:.4f}, Accuracy: {acc:.4f}")


def main() -> NoReturn:
    table = pd.read_csv("./data/tesla_stock.csv", index_col=0)
    table.index = pd.to_datetime(table.index)
    variable = table["Close"]

    x = create_arma_table(variable=variable, p=50, q=50, ma_window=10)
    x["next_day_price"] = create_next_day_price(variable=variable)
    x = x.dropna()
    x = x.loc["2019":]

    x_train, x_test = x.loc[:"2019-12"], x.loc["2020-01"]
    x_train, y_train = x_train.iloc[:, :-1], x_train.iloc[:, -1]
    x_test, y_test = x_test.iloc[:, :-1], x_test.iloc[:, -1]
    true_day_losses = y_test - x_test["ar(1)"]

    x = create_arma_table(variable=variable, p=50, q=50, ma_window=10)
    x["days_to_fall_relative_to_current_day"] = create_event_happened_day(variable=variable, percent=0.02)
    x = x.dropna()
    x = x.loc["2019":]

    x_train, x_test = x.loc[:"2019-12"], x.loc["2020-01"]
    x_train, y_train = x_train.iloc[:, :-1], x_train.iloc[:, -1]
    x_test, probability = x_test.iloc[:, :-1], x_test.iloc[:, -1]

    best_day_loss = np.loadtxt("./data/best_day_loss.txt")
    best_day_prob = np.loadtxt("./data/best_day_prob.txt")
    best_day_loss = np.where(best_day_loss < 0, best_day_loss, 0)
    print(best_day_loss, best_day_prob)
    # print(best_day_loss, best_day_prob)
    print(np.mean(best_day_loss * best_day_prob))

    true_day_losses = np.where(true_day_losses < 0, true_day_losses, 0)
    print(probability, true_day_losses)
    print(np.mean(true_day_losses * probability.values))


if __name__ == "__main__":
    main()
