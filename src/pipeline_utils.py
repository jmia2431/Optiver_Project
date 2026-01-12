# Import Modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns

# Features Generation Functions
def calculate_wap(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['WAP'] = (df_copy['bid_price1']*df_copy['ask_size1'] + df_copy['ask_price1']*df_copy['bid_size1']) /\
    (df_copy['bid_size1'] + df_copy['ask_size1'])
    return df_copy

def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    if 'WAP' not in df_copy.columns:
        df_copy = calculate_wap(df_copy)
    
    df_copy['log_return'] = np.log(df_copy['WAP'] / df_copy['WAP'].shift(1))
    
    return df_copy

def calculate_volatility(data: np.array):
    volatility = np.sqrt(np.sum(data**2))
    return volatility

# Interpolation for unchanged price
def interpolute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    full_seconds = pd.DataFrame({'seconds_in_bucket': range(600)})
    
    df_interpolated = pd.merge(full_seconds, df, on='seconds_in_bucket', how='left')
    df_interpolated['log_return'] = df_interpolated['log_return'].fillna(0)
    return df_interpolated


# Pridiction Models Training: return -> model

# GARCH Model:
class GARCH_model():
    def __init__(self, df: pd.DataFrame, total_length: int) -> None:
        self.df = df.copy()
        self.total = total_length
        self.y = None
        self.model = None

    def predict(self, predict_length: int, simulations: int):
        self.df = interpolute_log_returns(self.df)
        train_X = self.df.loc[self.df['seconds_in_bucket']<self.total-predict_length, 'log_return'].dropna()
        self.y = self.df.loc[self.df['seconds_in_bucket']>self.total-predict_length, 'log_return']
        self.model = arch_model(train_X, mean='Zero', vol='ARCH', p=1, q=1, dist='normal').fit(disp='off', show_warning=False)

        return self.model.forecast(horizon=predict_length, method='simulation', simulations=simulations)

    def get_actual(self):
        return self.y

    def get_confidence_mean_result(self, result):
        simulation_values = result.simulations.values
        lower_bounds = np.percentile(simulation_values, 2.5, axis=1)
        upper_bounds = np.percentile(simulation_values, 47.5, axis=1)
        confidence_interval_means = (lower_bounds + upper_bounds) / 2

        return confidence_interval_means

# Regression Model:
class Regression_model():
    pass
