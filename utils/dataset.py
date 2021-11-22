import numpy as np
import pandas as pd


def create_arma_table(variable: pd.Series, p: int = 1, q: int = 0, ma_window: int = 5) -> pd.DataFrame:
    columns = ["free_coef"]
    index = variable.index
    values = np.zeros((len(variable), 1 + p + q))

    # free_coef
    values[:, 0] = 1

    # autoregression
    for ar in range(1, p + 1):
        columns.append(f"ar({ar})")

        values[: (ar - 1), ar] = np.nan
        values[(ar - 1) :, ar] = variable[: (len(variable) - ar + 1)]

    if q > 0:
        moving_average = variable.rolling(window=ma_window).mean()
        for ma in range(1, q + 1):
            columns.append(f"ma({ma})")

            values[: (ma - 1), p + ma] = np.nan
            values[(ma - 1) :, p + ma] = moving_average[: (len(moving_average) - ma + 1)]

    return pd.DataFrame(values, columns=columns, index=index)
