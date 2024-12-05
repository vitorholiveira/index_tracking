from data import build_dataset

bvsp_index_ticker = '^BVSP'
bvsp_stock_tickers = [
  "ALOS3.SA", "ALPA4.SA", "ABEV3.SA", "ASAI3.SA", "AURE3.SA", "AZUL4.SA", "AZZA3.SA",
  "B3SA3.SA", "BBSE3.SA", "BBDC3.SA", "BBDC4.SA", "BRAP4.SA", "BBAS3.SA", "BRKM5.SA",
  "BRAV3.SA", "BRFS3.SA", "BPAC11.SA", "CXSE3.SA", "CRFB3.SA", "CCRO3.SA", "CMIG4.SA",
  "COGN3.SA", "CPLE6.SA", "CSAN3.SA", "CPFE3.SA", "CMIN3.SA", "CVCB3.SA", "CYRE3.SA",
  "ELET3.SA", "ELET6.SA", "EMBR3.SA", "ENGI11.SA", "ENEV3.SA", "EGIE3.SA", "EQTL3.SA",
  "EZTC3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA", "NTCO3.SA", "HAPV3.SA", "HYPE3.SA",
  "IGTI11.SA", "IRBR3.SA", "ITSA4.SA", "ITUB4.SA", "JBSS3.SA", "KLBN11.SA", "RENT3.SA",
  "LREN3.SA", "LWSA3.SA", "MGLU3.SA", "MRFG3.SA", "BEEF3.SA", "MRVE3.SA", "MULT3.SA",
  "PCAR3.SA", "PETR3.SA", "PETR4.SA", "RECV3.SA", "PRIO3.SA", "PETZ3.SA", "RADL3.SA",
  "RAIZ4.SA", "RDOR3.SA", "RAIL3.SA", "SBSP3.SA", "SANB11.SA", "STBP3.SA", "SMTO3.SA",
  "CSNA3.SA", "SLCE3.SA", "SUZB3.SA", "TAEE11.SA", "VIVT3.SA", "TIMS3.SA", "TOTS3.SA",
  "TRPL4.SA", "UGPA3.SA", "USIM5.SA", "VALE3.SA", "VAMO3.SA", "VBBR3.SA", "VIVA3.SA",
  "WEGE3.SA", "YDUQ3.SA"
]

sp100_index_ticker = '^OEX'
sp100_stock_tickers = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN',
    'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'C', 'CAT',
    'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX',
    'DHR', 'DIS', 'DOW', 'DUK', 'EMR', 'EXC', 'F', 'FDX', 'GD', 'GE',
    'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ',
    'JPM', 'KHC', 'KO', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDLZ', 'MDT',
    'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE', 'NFLX', 'NKE',
    'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM', 'RTX', 'SBUX',
    'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TMUS', 'TSLA', 'TXN', 'UNH',
    'UNP', 'UPS', 'USB', 'V', 'VZ', 'WBA', 'WFC', 'WMT'
]

start_date_dataset = "2023-01-01"
end_date_dataset = "2024-01-01"

build_dataset(sp100_index_ticker, sp100_stock_tickers, start_date_dataset, end_date_dataset)
build_dataset(bvsp_index_ticker, bvsp_stock_tickers, start_date_dataset, end_date_dataset)