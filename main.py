from indextracking import IndexTracking
import pandas as pd
import json


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

def treinacao(index_ticker, portfolio_size=10, max_iterations=1000000):
    if(index_ticker=='^BVSP'):
        data = pd.read_csv('stock_variance_^BVSP.csv', index_col=0)
        index_data = data[index_ticker]
        stock_data = data.drop(columns=[index_ticker])
    elif(index_ticker=='^OEX'):
        data = pd.read_csv('stock_variance_^OEX.csv', index_col=0)
        index_data = data[index_ticker]
        stock_data = data.drop(columns=[index_ticker])
    else:
        print('ERROR')
        return
    
    start_date_dataset = "2023-01-01"
    end_date_dataset = "2024-01-01"
    train_start = start_date_dataset
    train_end = "2023-06-01"
    test_start = "2023-06-02"
    test_end = end_date_dataset

    model = IndexTracking(stock_data=stock_data, index_data=index_data)
    model.split_data(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
    model.mult_start_optimization(num_starts=1, portfolio_size=portfolio_size, max_iterations=max_iterations)

    save_result(model.best_result['regular'], portfolio_size=portfolio_size, max_iterations=max_iterations, index_ticker=index_ticker)

# FLORES
bvsp_index_ticker = '^BVSP'
sp100_index_ticker = '^OEX'

treinacao(bvsp_index_ticker, portfolio_size=10, max_iterations=10**(7))
treinacao(sp100_index_ticker, portfolio_size=10, max_iterations=10**(7))

treinacao(bvsp_index_ticker, portfolio_size=20, max_iterations=10**(7))
treinacao(sp100_index_ticker, portfolio_size=20, max_iterations=10**(7))