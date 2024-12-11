from dataset import build_dataset, get_tickers

bvsp_index_ticker = '^BVSP'
bvsp_stock_tickers = get_tickers('tickers_^BVSP.csv')

sp100_index_ticker = '^OEX'
sp100_stock_tickers = get_tickers('tickers_^OEX.csv')

start_date_dataset = "2023-01-01"
end_date_dataset = "2024-01-01"

build_dataset(sp100_index_ticker, sp100_stock_tickers, start_date_dataset, end_date_dataset)
build_dataset(bvsp_index_ticker, bvsp_stock_tickers, start_date_dataset, end_date_dataset)
