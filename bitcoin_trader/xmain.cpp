#include <dlib/svm.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <thread>
#include <curl/curl.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <map>
#include <math.h>
#include <cmath>
#include <string>
#include <sstream>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <armadillo>
#include <mlpack/core.hpp>
#include <time.h>
#include <chrono>
#include "kapi.hpp"

#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/WebSocket.h>
#include <Poco/Net/Context.h>
#include <Poco/StreamCopier.h>
#include <Poco/Net/SSLManager.h>
#include <Poco/Net/AcceptCertificateHandler.h>
#include <Poco/SharedPtr.h>


constexpr char api_key[] = "";
constexpr char api_private_key[] = "";

using namespace boost::property_tree;
using namespace Poco::Net;
using namespace Poco;
using namespace dlib;

using sample_type = matrix<double, 30, 1>;

using KMeansType = mlpack::KMeans<>;

int stamp(){
    // Get the current time since the Unix epoch
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();

    // Convert to seconds as a floating-point number
    int timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0;

    // Print the timestamp
    return timestamp;
}

// Normalize orderbook snapshot to feed into autoencoder
std::vector<double> Normalizer(std::map<std::string, std::vector<double>> data){
    std::vector<double> result;
    double hbid = data["bidPrice"][0];
    double lask = data["askPrice"][0];
    double mid = (hbid + lask) / 2.0;
    int n = data["bidPrice"].size();
    double maxbid = 0, maxask = 0;
    for(int i = 0; i < n; ++i){
        data["bidPrice"][i] = (data["bidPrice"][i] - mid) / mid;
        data["askPrice"][i] = (data["askPrice"][i] - mid) / mid;
        maxbid += data["bidVolume"][i];
        maxask += data["askVolume"][i];
    }
    double imbalance = (maxbid - maxask)/(maxbid + maxask);
    for(int i = 0; i < n; ++i){
        data["bidVolume"][i] /= maxbid;
        data["askVolume"][i] /= maxask;
    }
    for(auto & choices : {"bidPrice","bidVolume","askPrice","askVolume"}){
        result.insert(result.end(), data[choices].begin(), data[choices].end());
    }
    //result.push_back(imbalance);

    return result;
}

// Convertes a torch tensor to a 2d vector
std::vector<std::vector<float>> tensor_to_vector2d(torch::Tensor t) {
    std::vector<std::vector<float>> result;
    auto t_contig = t.contiguous(); // Ensure tensor is contiguous
    auto dims = t_contig.dim();  // Get tensor dimensions
    
    if (dims == 1) {
        // Handle 1D tensor (single vector)
        auto accessor = t_contig.accessor<float, 1>(); // 1D accessor
        std::vector<float> row;
        for (int i = 0; i < t.size(0); ++i) {
            row.push_back(accessor[i]);
        }
        result.push_back(row);
    } else if (dims == 2) {
        // Handle 2D tensor (matrix)
        auto accessor = t_contig.accessor<float, 2>(); // 2D accessor
        for (int i = 0; i < t.size(0); ++i) {
            std::vector<float> row;
            for (int j = 0; j < t.size(1); ++j) {
                row.push_back(accessor[i][j]);
            }
            result.push_back(row);
        }
    } else {
        // Handle higher-dimensional tensors (optional)
        throw std::invalid_argument("Unsupported tensor dimensionality");
    }

    return result;
}

// Neural Network: Builds an Autoencoder with an LSTM Neural Network to learn and generate new orderbook patterns
struct OBNetImpl : torch::nn::Module {
    torch::nn::Sequential encoder, decoder;

    torch::nn::LSTM lstm1{nullptr}, lstm2{nullptr};
    // Fully connected layers
    torch::nn::Linear fc_enc{nullptr}, fc_dec1{nullptr}, fc_dec2{nullptr}, fc_out{nullptr};

    OBNetImpl(int64_t input_dim) {
        // Encoder: 2 LSTM layers + 1 FC
        lstm1 = register_module("lstm1", torch::nn::LSTM(torch::nn::LSTMOptions(input_dim, 32).batch_first(true)));
        lstm2 = register_module("lstm2", torch::nn::LSTM(torch::nn::LSTMOptions(32, 16).batch_first(true)));
        fc_enc = register_module("fc_enc", torch::nn::Linear(16, 8));

        // Decoder: 3 FC layers
        fc_dec1 = register_module("fc_dec1", torch::nn::Linear(8, 16));
        fc_dec2 = register_module("fc_dec2", torch::nn::Linear(16, 32));
        fc_out = register_module("fc_out", torch::nn::Linear(32, input_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto lstm_out1 = std::get<0>(lstm1->forward(x));
        lstm_out1 = torch::relu(lstm_out1);
        auto lstm_out2 = std::get<0>(lstm2->forward(lstm_out1));
        lstm_out2 = torch::relu(lstm_out2);

        // Get last time step from LSTM output
        torch::Tensor last_hidden = lstm_out2.select(1, lstm_out2.size(1) - 1);
        torch::Tensor latent = fc_enc->forward(last_hidden);

        // Decoder forward pass
        auto out = torch::relu(fc_dec1->forward(latent));
        out = torch::relu(fc_dec2->forward(out));
        out = fc_out->forward(out);
        return out;
    }

    torch::Tensor encoding(torch::Tensor x) {
        auto lstm_out1 = std::get<0>(lstm1->forward(x));
        lstm_out1 = torch::relu(lstm_out1);
        auto lstm_out2 = std::get<0>(lstm2->forward(lstm_out1));
        lstm_out2 = torch::relu(lstm_out2);
        torch::Tensor last_hidden = lstm_out2.select(1, lstm_out2.size(1) - 1);
        torch::Tensor latent = fc_enc->forward(last_hidden);
        return latent;
    }

};

// Convert 2D vector to ARMA format for matrix operations
arma::mat convert_to_arma(const std::vector<std::vector<float>>& data) {
    size_t rows = data.size();
    size_t cols = data[0].size();
    arma::mat mat(cols, rows); // mlpack expects col-major format (features in rows)

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(j, i) = data[i][j]; // transpose for mlpack (col-major)

    return mat;
}

// Generates a series of weights for three classes [Buy, Hold, Sell] and exports them to be fit into a support vector machine
std::vector<double> kmeans_classify(const std::vector<std::vector<float>>& features) {
    std::vector<double> classes;
    arma::mat data = convert_to_arma(features);
    KMeansType kmeans;

    size_t clusters = 3;
    arma::Row<size_t> assignments;
    arma::mat centroids;

    // Perform KMeans with centroids output
    kmeans.Cluster(data, clusters, assignments, centroids);

    // Analyze centroids to determine meaning
    std::map<size_t, std::string> clusterMeaning;

    std::vector<std::pair<size_t, double>> centroid_scores;
    for (size_t i = 0; i < centroids.n_cols; ++i) {
        double sum = arma::accu(centroids.col(i)); // sum as a proxy signal strength
        centroid_scores.push_back({i, sum});
    }

    // Sort centroids by score (lowest = SELL, middle = HOLD, highest = BUY)
    std::sort(centroid_scores.begin(), centroid_scores.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    clusterMeaning[centroid_scores[0].first] = "SELL";
    clusterMeaning[centroid_scores[1].first] = "HOLD";
    clusterMeaning[centroid_scores[2].first] = "BUY";

    std::string final_result;
    // Print results
    for (size_t i = 0; i < assignments.n_elem; ++i) {
        //std::cout << "Pattern: " << clusterMeaning[assignments[i]] << " signal\n";
        final_result = clusterMeaning[assignments[i]];
    }
    if(final_result == "BUY"){
        classes.push_back(0);
    }
    if(final_result == "SELL"){
        classes.push_back(1);
    }
    
    //std::cout << "\n📊 Centroid Signals:\n";
    for (const auto& [idx, score] : centroid_scores) {
        //std::cout << "Cluster " << idx << " score: " << score
        //          << " → " << clusterMeaning[idx] << "\n";
        classes.push_back(score);
    }
    

    return classes;
}

// Cuts each orderbook slice with a given depth
std::map<std::string, std::vector<double>> QFinBook(std::map<double, double> bids, std::map<double, double> asks, int depth){
    std::map<std::string, std::vector<double>> result;
    int count = 0;
    for (auto it = bids.rbegin(); it != bids.rend() && count < depth; ++it, ++count) {
        result["bidPrice"].push_back(it->first);
        result["bidVolume"].push_back(it->second);
    }
    count = 0;
    for(auto it = asks.begin(); it != asks.end() && count < depth; ++it, ++count){
        result["askPrice"].push_back(it->first);
        result["askVolume"].push_back(it->second);
    }
    return result;
}

// Calculates the cumulative rate of return for Bitcoin in a set window
double xbox(std::vector<double> u){
    double result = 1.0;
    for(int i = 1; i < u.size(); ++i){
        result *= u[i]/u[i-1];
    }
    return result - 1.0;
}


int main(){

    curl_global_init(CURL_GLOBAL_ALL);
    KAPI kapi(api_key, api_private_key);

    // Connect to trading websocket
    SharedPtr<InvalidCertificateHandler> pCert = new AcceptCertificateHandler(false);
    Context::Ptr context = new Context(Context::CLIENT_USE, "", "", "", Context::VERIFY_RELAXED);
    SSLManager::instance().initializeClient(0, pCert, context);

    // Start importing and updating data feeds
    std::thread kraken(kapi.KrakenFeed, &kapi);
    std::thread coinbase(kapi.CoinbaseFeed, &kapi);

    // Get a token to authenticate with the trading socket api
    std::string TOKEN = kapi.get_token();
    
    std::map<std::string, std::vector<double>> book;

    std::string side = "neutral";
    double volume = 0.00005;
    double entrybal = 0;

    // Define trading log writer
    std::ofstream writer("TradeLog.csv");

    int t0 = stamp();
    std::vector<double> snapshot, quantitative, bprices, kprices;
    
    bool trading = false;
    std::vector<sample_type> samples;
    std::vector<double> labels;

    // Declare support vector machine
    svm_c_trainer<linear_kernel<sample_type>> trainer;
    trainer.set_c(10); // Set the regularization parameter
    decision_function<linear_kernel<sample_type>> df;

    // Price Limit is the number of Bitcoin prices stored to get the cumulative rate of return
    int price_limit = 10;

    // Input limit is how much rows the Support Vector Machine needs to store for predictions
    int input_limit = 40;

    // Slices of the limit orderbook to be analyzed in the Autoencoder
    std::vector<int> ldepth = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    // Declare an Autoencoder for each depth
    std::map<int, std::shared_ptr<OBNetImpl>> AI;

    for (auto& depth : ldepth) {
        int64_t input_dim = depth * 4;
        AI[depth] = std::make_shared<OBNetImpl>(input_dim);
    }

    // Connect to Kraken's Trading WebSocket
    HTTPSClientSession session("ws-auth.kraken.com", 443);

    // Prepare the HTTP upgrade request for WebSocket
    HTTPRequest request(HTTPRequest::HTTP_GET, "/v2", HTTPMessage::HTTP_1_1);
    request.set("Upgrade", "websocket");
    request.set("Connection", "Upgrade");
    request.set("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==");  // use any base64-encoded string
    request.set("Sec-WebSocket-Version", "13");

    HTTPResponse response;

    // Create the WebSocket and connect
    WebSocket wsock(session, request, response);
    std::vector<char> buffer(2048);
    int flags;

    int n = wsock.receiveFrame(buffer.data(), static_cast<int>(buffer.size()), flags);
    std::string incoming(buffer.data(), n);

    std::cout << "Connected to socket trader: " << incoming << std::endl;

    std::string opening = kapi.SOpenOrders(TOKEN);
    
    // Send the token to the WebSocket for authentication
    wsock.sendFrame(opening.data(), static_cast<int>(opening.length()), WebSocket::FRAME_TEXT);
    
    std::string transaction_id = "", order_fill = "";

    double cc = 0.1; // Incremental change in order price if limit order does not get filled

    int T0 = stamp();

    while(true){
        // Waits until the datafeed from the websocket threads has fully synced
        if(kapi.SocketSync() == true){

            // Feeds each slice of the level2 orderbook into the autoencoder to detect patterns
            quantitative.clear();
            for(auto & depth : ldepth){

                // Pull and normalize orderbook
                book = QFinBook(kapi.bids, kapi.asks, depth);
                std::vector<double> inputs = Normalizer(book);
                
                // Convert inputs to a torch tensor and reshape
                torch::Tensor lob_batch = torch::tensor(inputs);
                lob_batch = lob_batch.view({1, 1, static_cast<int64_t>(inputs.size())});

                // Extract latent features to be sent to the KMeans clustering algorithm
                torch::Tensor latents = AI[depth]->encoding(lob_batch);

                // Convert tensor to a 2D vector
                std::vector<std::vector<float>> features = tensor_to_vector2d(latents);
                
                // Extract Buy and Sell signals with the clustering algorithm
                snapshot = kmeans_classify(features);
                std::string position;
                if(snapshot[0] == 0){
                    position = "BUY";
                }
                if(snapshot[0] == 1){
                    position = "SELL";
                }
                snapshot.erase(snapshot.begin());

                // Push all data to quantitative vector which is fed into the support vector machine
                quantitative.insert(quantitative.end(), snapshot.begin(), snapshot.end());
                
            }

            // Get the most current websocket stream showing trading orders being executed
            std::vector<char> buyer(2048);
            int flagz;
            int o = wsock.receiveFrame(buyer.data(), static_cast<int>(buyer.size()), flagz);
            std::string current_message(buyer.data(), o);

            std::cout << ">. " << current_message << ") " << side << std::endl;

            // Waits until limit order is filled
            if(side == "longwait"){
                ptree nft;
                std::stringstream ss(current_message);
                read_json(ss, nft);
                for(ptree::const_iterator it = nft.begin(); it != nft.end(); ++it){
                    // Edits the order until it has been filled
                    if(it->first == "type"){
                        if(it->second.get_value<std::string>() == "update"){
                            for (const auto& item : nft.get_child("data")) {
                                order_fill = item.second.get<std::string>("order_status");
                                std::cout << "has bid been filled? " << order_fill << " | " << cc << std::endl;
                                // If order has been filled it transfers to a long position
                                if(order_fill == "filled"){
                                    t0 = stamp();
                                    side = "long";
                                    break;
                                }
                                // Increases bid price until order has been filled
                                std::string editor = kapi.AmendOrder("BTC/USD", kapi.khigh_bid+cc, volume, "ClassOf2014", TOKEN);
                                wsock.sendFrame(editor.data(), static_cast<int>(editor.length()), WebSocket::FRAME_TEXT);
                                cc += 0.1;
                            }
                            
                        }
                    }
                }
                
            }
            
            // Waits until exit order has been filled
            if(side == "exitwait"){
                ptree nft;
                std::stringstream ss(current_message);
                read_json(ss, nft);
                for(ptree::const_iterator it = nft.begin(); it != nft.end(); ++it){
                    // Edits order until it has been filled
                    if(it->first == "type"){
                        if(it->second.get_value<std::string>() == "update"){
                            for (const auto& item : nft.get_child("data")) {
                                order_fill = item.second.get<std::string>("order_status");
                                std::cout << "has ask been filled? " << order_fill << " | " << cc << std::endl;
                                // Exits trade if sell order has been filled
                                // Writes the rate of return of the trade to a csv file along with the current timestamp
                                if(order_fill == "filled"){
                                    double retrate = kapi.AccountBalance() / entrybal - 1.0;
                                    std::cout << "Rate of Return: " << retrate << std::endl;
                                    writer << std::to_string(stamp()) << "," << std::to_string(retrate) << "\n";
                                    writer.flush();
                                    side = "neutral";
                                    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
                                    t0 = stamp();
                                    break;
                                }
                                // Changes the ask price until sell order has been filled
                                std::string editor = kapi.AmendOrder("BTC/USD", kapi.klow_ask-cc, volume, "ClassOf2019", TOKEN);
                                wsock.sendFrame(editor.data(), static_cast<int>(editor.length()), WebSocket::FRAME_TEXT);
                                cc += 0.1;
                            }
                        }
                    }
                }
                
            }
            
            // Places trades once all machine learning data has been loaded
            if(trading == true){
                // Builds the latest data vector to be passed into the predictor
                sample_type test_sample(quantitative.size());
                for(int i = 0; i < quantitative.size(); ++i){
                    test_sample(i) = quantitative[i];
                }

                // Predicts the next price movement
                double prediction = df(test_sample);
                std::cout << stamp() << ") Position: " << side << " | Signal: " << prediction << " | Bid: " << kapi.khigh_bid << " Ask: " << kapi.klow_ask << std::endl;
                
                // Does not process signal if it is infinite
                if(std::isinf(prediction)){
                    prediction = 0;
                }

                // Places a sell order to exit the long position
                if((stamp() - t0 > 60 || prediction >= 1) && side == "long"){
                    side = "exitwait";
                    transaction_id = "00";
                    cc = 0.1;
                    std::cout << "Exit long position" << std::endl;
                    // Communicates with socket to sell out of the position
                    std::string message = kapi.SLimitSell("BTC/USD", kapi.klow_ask-cc, volume, "ClassOf2019", TOKEN);
                    wsock.sendFrame(message.data(), static_cast<int>(message.length()), WebSocket::FRAME_TEXT);
                }

                // Places a buy order to enter the long position
                if(stamp() - t0 > 20 && kapi.kprice < kapi.cbwap && prediction <= -1 && side == "neutral"){
                    entrybal = kapi.AccountBalance();
                    cc = 0.1;
                    side = "longwait";
                    transaction_id = "00";
                    std::cout << "Enter long position" << std::endl;
                    // Communicates with socket to buy into a position in Bitcoin
                    std::string message = kapi.SLimitBuy("BTC/USD", kapi.khigh_bid+cc, volume, "ClassOf2014", TOKEN);
                    wsock.sendFrame(message.data(), static_cast<int>(message.length()), WebSocket::FRAME_TEXT);
                    t0 = stamp();
                }
            }

            // Stores orderbook weighted price and last price
            bprices.push_back(kapi.cbwap);
            kprices.push_back(kapi.kprice);

            // Starts building the Support Vector Machine inputs
            if(bprices.size() > price_limit){
                sample_type a(quantitative.size());
                for(int i = 0; i < quantitative.size(); ++i){
                    a(i) = quantitative[i];
                }
                samples.push_back(a);
                
                // Sets SVM output signal to 1 if the cumulative rate of return of Bitcoins price
                // is greater than zero, else it sets to -1
                if(0 < xbox(kprices)){
                    labels.push_back(+1);
                } else {
                    labels.push_back(-1);
                }
                bprices.erase(bprices.begin());
                kprices.erase(kprices.begin());
            } else {
                std::cout << "Still need to collect money: " << price_limit - bprices.size() << std::endl;
            }

            // Trains the support vector machine once the input frame has reached its limit
            if(samples.size() > input_limit){
                trading = true;
                df = trainer.train(samples, labels);
                samples.erase(samples.begin());
                labels.erase(labels.begin());
            } else {
                std::cout << "Still need to load dataset: " << input_limit - samples.size() << std::endl;
            }

        }
        // Pauses each iteration for 0.333 seconds to prevent overload
        std::this_thread::sleep_for(std::chrono::milliseconds(333));        
    }

    
    kraken.join();
    coinbase.join();
    
    return 0;
}



