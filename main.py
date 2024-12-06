from indextracking import IndexTracking
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

model = IndexTracking(stock_data=stock_variance, index_data=index_variance)
model.split_data(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
model.mult_start_optimization(num_starts=3, portfolio_size=10)



































def plot_portfolio(portfolio, stocks_data, index_data, start_date, end_date, index_name):
    # Seleciona apenas as ações presentes no portfólio
    a = [key for key in portfolio if key in stocks_data.columns]
    portfolio_data = stocks_data[a]
    weights = pd.Series(portfolio).reindex(a).values
    
    # Filtra os dados para o intervalo de datas especificado
    portfolio_data_filtered = (portfolio_data * weights).sum(axis=1).loc[start_date:end_date]
    index_data_filtered = index_data.loc[start_date:end_date]
    
    # Combina as duas séries antes da normalização
    combined_data = pd.concat([portfolio_data_filtered, index_data_filtered], axis=1)
    
    # Usa fit_transform uma única vez
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(combined_data.values)
    
    # Separa os dados normalizados
    portfolio_data_normalized = normalized_data[:, 0]
    index_data_normalized = normalized_data[:, 1]
    
    # Converte datas para string
    start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    # Cria o plot
    fig = plt.figure(figsize=(15, 8), facecolor='black')
    eixo = fig.add_axes([0, 0, 1, 1], facecolor='black')
    
    eixo.plot(index_data_filtered.index, index_data_normalized, 
              color='red', linestyle='-', linewidth=3, label=index_name)
    eixo.plot(portfolio_data_filtered.index, portfolio_data_normalized, 
              color='gold', linestyle='--', linewidth=3, label="Portfolio")
    
    eixo.set_title(f"Portfolio 15 stocks {index_name}: {start_date_str} to {end_date_str}", 
                   fontsize=25, pad=20, color='white')
    
    eixo.legend(title="Indicador", loc='upper left', fontsize=15, 
                facecolor='black', edgecolor='white', 
                title_fontsize=15, labelcolor='white')
    
    eixo.set_ylabel('Normalized Value', fontsize=20, color='white')
    eixo.set_xlabel('Data', fontsize=20, color='white')
    
    eixo.tick_params(colors='white')
    eixo.grid(color='gray', linestyle='--')
    
    plt.show()

plot_portfolio(model.best_result['portfolio'], stocks_values, index_values, start_date_dataset, end_date_dataset, index_ticker)