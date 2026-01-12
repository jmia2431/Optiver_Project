import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ARIMA_model:
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_wap(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy['WAP'] = (df_copy['bid_price1']*df_copy['ask_size1'] + df_copy['ask_price1']*df_copy['bid_size1']) /\
                         (df_copy['bid_size1'] + df_copy['ask_size1'])
        return df_copy
    
    @staticmethod
    def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if 'WAP' not in df_copy.columns:
            df_copy = ARIMA_model.calculate_wap(df_copy)

        df_copy['log_return'] = np.log(df_copy['WAP'] / df_copy['WAP'].shift(1))

        return df_copy
    
    @staticmethod
    def calculate_volatility(data: np.array):
        volatility = np.sqrt(np.sum(data**2))
        return volatility

    @staticmethod
    def interpolute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
        full_seconds = pd.DataFrame({'seconds_in_bucket': range(600)})
        df_interpolated = pd.merge(full_seconds, df, on='seconds_in_bucket', how='left')
        df_interpolated['log_return'] = df_interpolated['log_return'].fillna(0)
        return df_interpolated

    @staticmethod
    def predict_future(training_data: pd.DataFrame, step):
        predictions = []
        model = ARIMA(training_data, order=(4,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(step)
        predictions.append(forecast)
        predictions = np.array(predictions).reshape(-1, 1)
        training_data = np.append(training_data[step:], forecast)

        return np.array(predictions)
    
    @staticmethod
    def process_stock_id(stock_id):
        try:
            df = pd.read_csv(f'./individual_book_train/stock_{stock_id}.csv')
            df['WAP'] = (df['bid_price1']*df['ask_size1'] + df['ask_price1']*df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])

            df_by_time_id = df.groupby('time_id')

            result = []
            for group_name, group_df in df_by_time_id:
                group_df = ARIMA_model.calculate_log_returns(group_df)
                group_df = ARIMA_model.interpolute_log_returns(group_df)

                volatility = []

                for i in range(40):
                    data = np.array(group_df['log_return'].loc[i*15:(i+1)*15])
                    v = ARIMA_model.calculate_volatility(data)
                    volatility.append({'volatility': v})

                volatility_array = np.array([entry['volatility'] for entry in volatility])

                train = volatility_array[:36]
                test = volatility_array[36:]

                prediction = ARIMA_model.predict_future(train, 4)

                numerator = np.abs(prediction - test)
                denominator = (np.abs(test) + np.abs(prediction)) / 2

                smape = 100 * np.mean(numerator / denominator)

                result.append({'stock_id': stock_id, 'time_id': group_name, 'smape': smape})

            return pd.DataFrame(result)

        except FileNotFoundError:
            return None
    
    @staticmethod
    def volatility_list(data):
        value = []
        for time_id in data['time_id'].unique():
            df_filtered = data[data['time_id'] == time_id]  # all time id
            df_filtered = ARIMA_model.calculate_log_returns(df_filtered)
            df_filtered = ARIMA_model.interpolute_log_returns(df_filtered)

            volatility = []

            for i in range(10):
                d = np.array(df_filtered['log_return'].loc[i*60:(i+1)*60])
                v = ARIMA_model.calculate_volatility(d)
                volatility.append(v)

            value.append({'time_id': time_id, 'volatility': volatility})

        return value
    
    @staticmethod
    def mean(volatility_list):
        mean = []
        for i in range(3830):
            m = sum(volatility_list[i]['volatility'])/len(volatility_list[i]['volatility'])
            mean.append({'time_id':volatility_list[i]['time_id'], 'volatility': m})

        return pd.DataFrame(mean)
    
    @staticmethod
    def calculate_max_min_volatility(data):
        max_min_volatility = []
        for time_id in data['time_id'].unique():
            df_filtered = data[data['time_id'] == time_id]  # Filter data for each time_id
            df_filtered = ARIMA_model.calculate_log_returns(df_filtered)
            df_filtered = ARIMA_model.interpolute_log_returns(df_filtered)

            volatilities = []

            for i in range(10):
                d = np.array(df_filtered['log_return'].loc[i * 60:(i + 1) * 60])
                volatility = ARIMA_model.calculate_volatility(d)
                volatilities.append(volatility)

            train = volatilities[:9]

            max_volatility = max(train)
            min_volatility = min(train)
            max_min_volatility.append({'time_id': time_id, 'max_volatility': max_volatility, 'min_volatility': min_volatility})

        return max_min_volatility
