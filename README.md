# CryptoHiddenMarkovModel

Overview

This is a Multi-Variable Gaussian Hidden Markov Model (HMM) to historical financial price data, infers hidden market trends (e.g., Bull/Bear, High/Low volatility, Sideways), and provides analysis, forward-filtered state probabilities, short-term regime forecasts, and visualizations. Purposed for crypto assets, so model may not converge well on indicies such as SPY or NASDAQ. Model does not performed well on heavily skewed data such as years of low/no activity (e.g data pre-dating 2020 on BTC charts)

Features:

Downloads historical price data from Yahoo Finance.

Engineers features: log returns, log average true range (volatility), and a rolling area under curve feature.

Trains a multivariate Gaussian HMM (via hmmlearn) to infer hidden states.

Automatically assigns semantic regime labels using transition based analysis.

Computes numerically stable forward probabilities and propagates future state probabilities with the learned transition matrix.

Prints summary statistics (regime frequency, returns by regime, stationary distribution) and creates plots (price colored by regime, volatility, forward probabilities, transition heatmap, forecast bar chart).
