from indextracking import create_portfolio
from data import build_dataset
import datetime as dt
import numpy as np
import pandas as pd

# Define index and stock tickers
bvsp_index_ticker = '^BVSP'

sp100_index_ticker = '^OEX'

# Build dataset from choosen tickers
data_sp100 = pd.read_csv('returns_^BVSP_and_01-01-2023_02-01-2023_years.csv')
data_bvsp = pd.read_csv('returns_^OEX_and_01-01-2023_02-01-2023_years.csv')

data_sp100_index = data_sp100[sp100_index_ticker]
data_sp100_stocks = data_sp100.drop(columns=[sp100_index_ticker])

data_bvsp_index = data_bvsp[bvsp_index_ticker]
data_bvsp_stocks = data_bvsp.drop(columns=[bvsp_index_ticker])

portfolio_weights = create_portfolio(data_sp100_stocks, data_sp100_index, 10)