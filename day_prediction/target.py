from typing import List

import numpy as np
import pandas as pd


def create_days_to_fall_relative_to_current_day(variable: pd.Series, percent: float = 0.02) -> pd.Series:
    target: List[int] = []
    for index in range(len(variable) - 1):
        fall_happened = False
        for add_index in range(index + 1, len(variable)):
            if (variable[index] - variable[add_index]) > percent:
                target.append(add_index - index)
                fall_happened = True
                break
        if not fall_happened:
            target.append(np.nan)
    target.append(np.nan)
    return pd.Series(target, index=variable.index)


def create_days_to_fall_relative_to_previous_month(variable: pd.Series, percent: float = 0.02) -> pd.Series:
    month_mean = variable.groupby([variable.index.year, variable.index.month]).mean()
    target: List[int] = []
    for index in range(len(variable) - 1):
        fall_happened = False
        prev_month_date = variable.index[index] - pd.Timedelta(31, "D")
        try:
            prev_month_average = month_mean.loc[(prev_month_date.year, prev_month_date.month)]
        except KeyError:
            target.append(np.nan)
            continue

        for add_index in range(index + 1, len(variable)):
            if (prev_month_average - variable[add_index]) > percent:
                target.append(add_index - index)
                fall_happened = True
                break

        if not fall_happened:
            target.append(np.nan)
    target.append(np.nan)
    return pd.Series(target, index=variable.index)
