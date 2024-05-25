import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from multiprocessing import Pool

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

def interpolute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    full_seconds = pd.DataFrame({'seconds_in_bucket': range(600)})
    
    df_interpolated = pd.merge(full_seconds, df, on='seconds_in_bucket', how='left')
    df_interpolated['log_return'] = df_interpolated['log_return'].fillna(0)
    return df_interpolated

def predict_future(training_data: pd.DataFrame, step):
    predictions = []
    model = ARIMA(training_data, order=(4,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(step)
    predictions.append(forecast)
    predictions = np.array(predictions).reshape(-1, 1)
    training_data = np.append(training_data[step:], forecast)

    return np.array(predictions)

def process_stock_id(stock_id):
    try:
        df = pd.read_csv(f'./individual_book_train/stock_{stock_id}.csv')
        df['WAP'] = (df['bid_price1']*df['ask_size1'] + df['ask_price1']*df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])

        df_by_time_id = df.groupby('time_id')

        result = []
        for group_name, group_df in df_by_time_id:
            group_df = calculate_log_returns(group_df)
            group_df = interpolute_log_returns(group_df)

            volatility = []

            for i in range(40):
                data = np.array(group_df['log_return'].loc[i*15:(i+1)*15])
                v = calculate_volatility(data)
                volatility.append({'volatility': v})
            
            volatility_array = np.array([entry['volatility'] for entry in volatility])

            train = volatility_array[:36]
            test = volatility_array[36:]
            test=np.sqrt(np.sum(np.power(test,2)))
            prediction = predict_future(train, 4)
            prediction=np.sqrt(np.sum(np.power(prediction,2)))
            numerator = np.abs(prediction - test)
            denominator = (np.abs(test) + np.abs(prediction)) / 2
            
            smape = 100 * np.mean(numerator / denominator)

            result.append({'stock_id': stock_id, 'time_id': group_name, 'smape': smape, 'prediction': prediction})
        
        return pd.DataFrame(result)
        
    except FileNotFoundError:
        return None