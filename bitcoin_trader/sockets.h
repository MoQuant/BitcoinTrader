#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <sstream>
#include <vector>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <thread>
#include <time.h>
#include <chrono>

#include <Poco/Net/SSLManager.h>
#include <Poco/Net/Context.h>
#include <Poco/Net/WebSocket.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/StreamCopier.h>

using namespace boost::property_tree;
using namespace Poco::Net;
using namespace Poco;

inline int postage(){
    // Get the current time since the Unix epoch
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();

    // Convert to seconds as a floating-point number
    int timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0;

    // Print the timestamp
    return timestamp;
}

class sockets {

    private:
        std::string kxurl = "ws.kraken.com";
        std::string cburl = "ws-feed.exchange.coinbase.com";

        std::string debugJSON(const boost::property_tree::ptree& pt) {
            std::ostringstream oss;
            boost::property_tree::write_json(oss, pt);
            return oss.str();
        }

        double getFirst(const boost::property_tree::ptree& pt) {
            // Iterate through the ptree
            for (const auto& kv : pt) {
                // Return the value of the first key-value pair as a double
                return std::stod(kv.second.get_value<std::string>());
            }
            throw std::runtime_error("The ptree is empty or does not contain valid values.");
        }

        // Calculates price volatility of the close prices
        double volatility(std::vector<double> close){
            double total = 0;
            for(auto & i : close){
                total += i;
            }
            double mu = total/((double) close.size());
            total = 0;
            for(auto & i : close){
                total += pow(i - mu, 2);
            }
            total /= ((double) close.size() - 1.0);
            return pow(total, 0.5)*6.0;
        }

        // Calculates the RSI of the close prices
        double relative_strength_index(std::vector<double> close){
            double up = 0;
            double down = 0;
            for(int i = 1; i < close.size(); ++i){
                double val = close[i] - close[i-1];
                if(val >= 0){
                    up += val;
                } else {
                    down += abs(val);
                }
            }
            down = fmax(down, 1);
            return 100 - 100 / (1 + up/down);
        }

        // Parses the Coinbase level2 orderbook
        void ParseCBBook(ptree df){
            bool snapshot = false;
            bool l2update = false;
            for(ptree::const_iterator it = df.begin(); it != df.end(); ++it){
                // Updates the level2 orderbook
                if(l2update == true && it->first == "changes"){
                    for(ptree::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt){
                        std::vector<std::string> hold;
                        for(ptree::const_iterator kt = jt->second.begin(); kt != jt->second.end(); ++kt){
                            hold.push_back(kt->second.get_value<std::string>().c_str());
                        }

                        double price = atof(hold[1].c_str());
                        double volume = atof(hold[2].c_str());
                        
                        // Adds and removes orders
                        if(hold[0] == "buy"){
                            if(volume == 0){
                                bids.erase(price);
                            } else {
                                bids[price] = volume;
                            }
                        } else {
                            if(volume == 0){
                                asks.erase(price);
                            } else {
                                asks[price] = volume;
                            }
                        }
                    }
                }
                // Parses the initial bid book snapshot
                if(snapshot == true && it->first == "bids"){
                    for(ptree::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt){
                        std::vector<double> hold;
                        for(ptree::const_iterator kt = jt->second.begin(); kt != jt->second.end(); ++kt){
                            hold.push_back(atof(kt->second.get_value<std::string>().c_str()));
                        }
                        bids[hold[0]] = hold[1];
                    }
                }

                // Parses the initial ask book snapshot
                if(snapshot == true && it->first == "asks"){
                    for(ptree::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt){
                        std::vector<double> hold;
                        for(ptree::const_iterator kt = jt->second.begin(); kt != jt->second.end(); ++kt){
                            hold.push_back(atof(kt->second.get_value<std::string>().c_str()));
                        }
                        asks[hold[0]] = hold[1];
                    }
                }

                // Checks to see whether the type is a snapshot or update
                if(it->first == "type"){
                    if(it->second.get_value<std::string>() == "l2update"){
                        l2update = true;
                    }
                    if(it->second.get_value<std::string>() == "snapshot"){
                        snapshot = true;
                    }
                }
            }
            // Fetches the orderbook imbalance metric
            obarb = orderbook_imbalancer(20);
        }

        // Parses the datafeed from Kraken for Bitcoins highest bid, lowest ask, and last price to be used in trading
        void CycloneKraken(ptree dataset){
            for(ptree::const_iterator it = dataset.begin(); it != dataset.end(); ++it){
                if(it->first == "a"){
                    klow_ask = getFirst(it->second);
                }
                if(it->first == "b"){
                    khigh_bid = getFirst(it->second);
                }
                if(it->first == "c"){
                    kprice = getFirst(it->second);
                }
                CycloneKraken(it->second);
                
            }
            
        }

        // Base parsing function for Kraken
        void KrakenParse(std::string message){
            std::stringstream ss(message);
            ptree dataset;
            read_json(ss, dataset);
            
            for(ptree::const_iterator it = dataset.begin(); it != dataset.end(); ++it){
                if(it->first == ""){
                    CycloneKraken(it->second);
                }
            }
        }

        // Base parsing function for Coinbase
        void CoinbaseParse(std::string message){
            std::stringstream ss(message);
            ptree dataset;
            read_json(ss, dataset);
            bool ticker = false;
            for(ptree::const_iterator it = dataset.begin(); it != dataset.end(); ++it){
                if(it->first == "type"){
                    if(it->second.get_value<std::string>() == "ticker"){
                        for(ptree::const_iterator ut = dataset.begin(); ut != dataset.end(); ++ut){
                            if(ut->first == "price"){
                                cbprice = std::stod(ut->second.get_value<std::string>());
                                cbclose.push_back(cbprice);
                                cbma = 2.0*volatility(cbclose);
                                cbrsi = relative_strength_index(cbclose);
                                if(cbclose.size() > 70){
                                    cbclose.erase(cbclose.begin());
                                }
                            }
                        }
                    } else {
                        ParseCBBook(dataset);
                    }
                }
            }
        }

        std::vector<double> cbclose;
        

    public:

        std::map<double, double> bids, asks;
        double kprice = 0;
        double cbprice = 0;
        double khigh_bid = 0;
        double klow_ask = 0;
        double cbma = 0;
        double cbrsi = 0;
        double obarb = 0;
        double cbwap = 0;

        // Calculates the imbalance in the level2 orderbook by taking
        // volume summations (bidv - askv)/(bidv + askv)
        double orderbook_imbalancer(int depth){
            //std::this_thread::sleep_for(std::chrono::milliseconds(250));
            double asksum = 0;
            double bidsum = 0;
            double bidpriceX = 0;
            double askpriceX = 0;
            int count = 0;
            for(auto it = bids.rbegin(); it != bids.rend(); ++it){
                bidsum += it->second;
                bidpriceX += it->first*it->second;
                if(count >= depth){
                    break;
                }
                count += 1;
            }
            count = 0;
            for(auto it = asks.begin(); it != asks.end(); ++it){
                asksum += it->second;
                askpriceX += it->first*it->second;
                if(count >= depth){
                    break;
                }
                count += 1;
            }
            cbwap = 0.5*((bidpriceX/bidsum) + (askpriceX/asksum));
            return (bidsum - asksum)/(bidsum + asksum);
        }

        // Checks to see if both websocket feeds for Coinbase and Kraken have synced
        bool SocketSync(){
            if(kprice == 0 || khigh_bid == 0 || klow_ask == 0 || cbclose.size() < 15 || obarb == 0){
                std::cout << "Loading: " << kprice << " | " << khigh_bid << " | " << klow_ask << " | " << obarb << std::endl;
                return false;
            }
            return true;
        }

        // Runs the websocket client to fetch Kraken data
        static void KrakenFeed(sockets * ws){
            try {
                std::cout << "Connected To Kraken............" << std::endl;
                
                // Kraken subscription message
                std::string message = R"({
                    "event": "subscribe",
                    "pair": ["XBT/USD"],
                    "subscription": {"name": "ticker"}
                })";
        
                // Setup SSL context
                Context::Ptr context = new Context(
                    Context::CLIENT_USE,
                    "",  // private key file
                    "",  // certificate file
                    "",  // CA location
                    Context::VERIFY_NONE,  // ← disables cert verification
                    9,    // verification depth
                    true // use default ciphers
                );
                SSLManager::instance().initializeClient(nullptr, nullptr, context);
        
                // Create session
                HTTPSClientSession session(ws->kxurl, 443);
                HTTPRequest request(HTTPRequest::HTTP_GET, "/");
                HTTPResponse response;
        
                // Open WebSocket connection
                WebSocket wsock(session, request, response);
        
                // Send subscription message
                wsock.sendFrame(message.data(), message.length(), WebSocket::FRAME_TEXT);
        
                int T0 = postage();
                char buffer[8192];
                int flags;
        
                while (true) {
                    // Gets the latest websocket stream from the channel
                    int n = wsock.receiveFrame(buffer, sizeof(buffer), flags);
                    std::string incoming(buffer, n);
                    
                    // Sends incoming string to be parsed into the Kraken variables
                    ws->KrakenParse(incoming);
                    
                    // Sends a ping message every 5 minutes to Kraken
                    if (postage() - T0 > 300) {
                        std::string pingMsg = R"({"event":"ping"})";
                        wsock.sendFrame(pingMsg.data(), pingMsg.size(), WebSocket::FRAME_TEXT);
                        T0 = postage();
                    }
        
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // Prevent tight loop
                }
        
            } catch (const Exception& ex) {
                std::cerr << "WebSocket Error: " << ex.displayText() << std::endl;
            }
        }

        // Runs the websocket data for the Coinbase feed
        static void CoinbaseFeed(sockets * ws){
            try {
                std::cout << "Connected To Coinbase..........." << std::endl;
        
                // Coinbase subscription message
                std::string message = R"({
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channels": ["ticker", "level2_batch"]
                })";
        
                // Create SSL context
                Context::Ptr context = new Context(
                    Context::CLIENT_USE,
                    "",  // private key file
                    "",  // certificate file
                    "",  // CA location
                    Context::VERIFY_NONE,  // ← disables cert verification
                    9,    // verification depth
                    true // use default ciphers
                );
                SSLManager::instance().initializeClient(nullptr, nullptr, context);
        
                // Setup secure WebSocket connection
                HTTPSClientSession session(ws->cburl, 443);
                HTTPRequest request(HTTPRequest::HTTP_GET, "/");
                HTTPResponse response;
        
                WebSocket wsock(session, request, response);
        
                // Send the subscription message
                wsock.sendFrame(message.data(), message.size(), WebSocket::FRAME_TEXT);
        
                std::vector<char> buffer(2 * 1024 * 1024);;
                int flags;
        
                while (true) {
                    // Gets live data from Coinbase and parses the level2 orderbook
                    int n = wsock.receiveFrame(buffer.data(), buffer.size(), flags);
                    std::string incoming(buffer.data(), n);
                    ws->CoinbaseParse(incoming);
                    
                    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // optional throttle
                }
        
                wsock.close();
        
            } catch (const Poco::Exception& ex) {
                std::cerr << "WebSocket Error: " << ex.displayText() << std::endl;
            }
        }

        int currenttime(){
            // Get the current time since the Unix epoch
            auto now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
        
            // Convert to seconds as a floating-point number
            int timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0;
        
            // Print the timestamp
            return timestamp;
        }

};


