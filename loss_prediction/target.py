import numpy as np
import pandas as pd


def create_next_day_price(variable: pd.Series) -> pd.Series:
    variable[:-1] = variable[1:]
    variable[-1] = np.nan
    return variable
