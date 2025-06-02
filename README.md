# :rocket: Bitcoin Trading System

## Description
I made this C++ trading system building off of the krakenapi repository in order to authenticate private api endpoints to execute algorithmic trades. My addition to this is I built an algorithmic trading system to trade Bitcoin by utilizing some machine learning.

## Running :bullettrain_front:
![alt](https://github.com/MoQuant/BitcoinTrader/blob/master/bitcoin_trader/images/trading.gif)

### Trading Algorithm Steps :hourglass:
1. Subscribe to Kraken, Coinbase, and Kraken Trader WebSocket feeds using PocoNet
2. Create several orderbook depth slices ranging from 10 to 100 depth
3. Normalize the price and volume data using (price - mid)/mid where price could be either bid or ask and mid is the mid-market price, and normalize the volume data by dividing by the total volume
4. Feed the normalized orderbook vector into an Autoencoder created with libtorch and has a LongShortTermMemory network which remembers previous orderbook snapshots
5. Extract the latent features from the Autoencoder and use a KMeans Clustering algorithm to generate three centroids whose weights all add up to 1
6. Group together all of the centroids from each orderbook depth slice and use them as inputs to a Support Vector Machine (using dlib)
7. Collect the rolling cumulative rate of returns (cror) of Bitcoin's price and match each input row with a -1 if cror < 0 or 1 if cror > 0, where -1 = Buy and 1 = Sell
8. Using this signal, along with an orderbook weighted price signal, execute a long order when both signals are activated, and exit the long order once the SVM predicts a 1 (sell)

### Trading Architecture :satellite:
This system is fully built in C++ and contains an authenticated WebSocket in order to place or edit algorithmic trades. Two datafeeds, one from Kraken and the other from Coinbase are streamed in this system to provide level2 orderbook data and high bid/low ask data. An authenticated token is generate in order to establish a connection between this program and Kraken's socket server. 

#### Libraries Used :key:
1. PocoNet   (WebSocket Streaming)
2. Libtorch  (PyTorch for C++ to generate Neural Network)
3. MLPack    (KMeans Clustering)
4. Armadillo (Arma Matrix Operations)
5. Dlib      (Support Vector Machine)
6. Boost     (Data Parsing in JSON)
7. CURL      (REST Requests)

### Disclaimer :red_circle: :bangbang:
This trading system is not profitable as shown in the Trade Log Graph, it is purely for educational purposes, do not use this for any trading!

## Trade Log Graph :blue_book:
![alt](https://github.com/MoQuant/BitcoinTrader/blob/master/bitcoin_trader/images/btc.png)

### Original Below :arrow_down:

krakenapi
=========

A C++ library for interfacing with the Kraken REST API (kraken.com).

Other programs
==============

krt
---

Source file of this program is krt.cpp.

### What is krt?

krt is a program to download Recent Trades from Kraken market data through API.  

### How trades are displayed? 
 
Recent trades are printed out to standard output in CSV format. The order of fields is "Time", "Order", "Price" and "Volume".

### Command line arguments

usage: krt \<pair\> \[interval\] \[since\]

krt can get the following command line arguments:

  \<pair\>   
  Asset pair to get trade data for.

  \[since\]  
  (Optional) the program returns trade data since given id. By default [since] is equal 
  to "0" to indicate the oldest possible trade data.

  \[interval\] 
  (Optional) how many seconds the program has to download new trade data. 
  By default the program doesn't use this parameter and it exits immidiatly after 
  download trade data. If [interval] is equal to 0 the program will not 
  use this parameter.
