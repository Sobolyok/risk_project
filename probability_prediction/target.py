from typing import List

import numpy as np
import pandas as pd


def create_event_happened_day(variable: pd.Series, percent: float = 0.03) -> pd.Series:
    target: List[int] = []
    for index in range(len(variable) - 1):
        target.append(int((variable[index] - variable[index + 1]) / variable[index] > percent))
    target.append(np.nan)
    return pd.Series(target, index=variable.index)


def create_event_happened_month(variable: pd.Series, percent: float = 0.03) -> pd.Series:
    target: List[int] = []
    month_mean = variable.groupby([variable.index.year, variable.index.month]).mean()
    for index in range(len(variable) - 1):
        next_day = variable[variable.index[index + 1]]
        prev_month_date = variable.index[index] - pd.Timedelta(31, "D")
        try:
            prev_month_average = month_mean.loc[(prev_month_date.year, prev_month_date.month)]
            target.append(int((prev_month_average - next_day) / prev_month_average > percent))
        except KeyError:
            target.append(np.nan)
    target.append(np.nan)
    return pd.Series(target, index=variable.index)
