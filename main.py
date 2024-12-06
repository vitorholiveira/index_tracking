from indextracking import IndexTracking
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json

# Organizig data
index_ticker = '^OEX'

data_values = pd.read_csv('stock_values_^OEX.csv', index_col=0)  # Inclui o índice
data_variance = pd.read_csv('stock_variance_^OEX.csv', index_col=0)

index_values = data_values[index_ticker]
stocks_values = data_values.drop(columns=[index_ticker])

index_variance = data_variance[index_ticker]
stock_variance = data_variance.drop(columns=[index_ticker])

# Defining dates
start_date_dataset = "2023-01-01"
end_date_dataset = "2024-01-01"
train_start = start_date_dataset
train_end = "2023-06-01"
test_start = "2023-06-02"
test_end = end_date_dataset

portfolio_size=10
max_iterations=1000000

model = IndexTracking(stock_data=stock_variance, index_data=index_variance)
model.split_data(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
model.mult_start_optimization(num_starts=1, portfolio_size=portfolio_size, max_iterations=1000000)


def save_result(result, portfolio_size, max_iterations, index_ticker):
    data = {
        'Error': result['error'],
        'Portfolio Weights': json.dumps(result['portfolio']),
        'Train Tracking Error': result['performance']['train_performance']['tracking_error'],
        'Train RMSE': result['performance']['train_performance']['root_mean_squared_error'],
        'Train Correlation': result['performance']['train_performance']['correlation'],
        'Test Tracking Error': result['performance']['test_performance']['tracking_error'],
        'Test RMSE': result['performance']['test_performance']['root_mean_squared_error'],
        'Test Correlation': result['performance']['test_performance']['correlation'],
        'Optimization Time': result['optimization_time']
    }
    
    df = pd.DataFrame([data])
    df.to_csv(f'result_{index_ticker}_{portfolio_size}stocks_{max_iterations}iterations.csv', index=False)

save_result(model.best_result['regular'], portfolio_size=portfolio_size, max_iterations=max_iterations, index_ticker=index_ticker)