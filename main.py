from test import run_tests

from utility import download_ticker_data
from generate_reports import generate_reports

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":

    run_tests()

    stocks = ['MU', 'UPS', 'IQV', 'HST', 'PPG', 'TMUS', 'FE', 'MOS', 'CL', 'FTV']

    download_ticker_data(tickers = stocks, period="max", data_dir="./snp_stocks/")
    download_ticker_data(tickers = '^VIX', period="max", data_dir="./stocks/")

    clf1 = MLPClassifier(hidden_layer_sizes=(150, 150, 200), activation= 'relu',max_iter=5000, random_state=42, learning_rate_init=0.015, warm_start=True)
    clf2 = LogisticRegression(solver='liblinear',penalty='l1', C=5.0, random_state=42, max_iter = 2000, warm_start=True)

    models = [clf1, clf2]

    performance = generate_reports(stocks = stocks, models = models, training_start_date = '2005-01-01', training_end_date = '2024-11-23')
