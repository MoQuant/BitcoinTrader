#ifndef _KRAKEN_KAPI_HPP_
#define _KRAKEN_KAPI_HPP_

#include <map>
#include <string>
#include <vector>
#include <curl/curl.h>
#include <thread>
#include "sockets.h"

// Kraken trader class
class KAPI : public sockets {
public:  

   // Input map
   typedef std::map<std::string,std::string> Input;

   std::string ws_url = "wss://ws.kraken.com";
   std::string auth_url = "wss://ws-auth.kraken.com";
   
   // Initial key and secret arguments
   KAPI(const std::string& key, const std::string& secret);
   
   // Declares functions to fetch account balance, trading balance, place limit and market orders
   std::string account_balance();
   std::string trading_balance();
   std::string limit_buy(std::string pair, double price, double volume, std::string clordid);
   std::string limit_sell(std::string pair, double price, double volume, std::string clordid);
   std::string market_sell(std::string pair, double volume);
   std::string market_buy(std::string pair, double volume);
   
   // Declares functions to cancel, open, or edit orders
   std::string cancel_order(std::string pair, std::string txid);
   std::string open_orders(std::string cl_ord_id);
   std::string edit_order(std::string pair, std::string txid, double volume, double price);

   // These functions are designed to extract transaction ids
   double AccountBalance();
   std::string Filled(std::string x, std::string txid);
   std::string tID(std::string response);
   std::string eID(std::string response);

   // These are custom functions which allow for editing limit orders through REST or WebSocket
   std::string LimitBuy(std::string pair, double volume, std::string clordid);
   std::string LimitSell(std::string pair, double volume, std::string clordid);
   std::string get_token();
   std::string SLimitBuy(std::string pair, double price, double volume, std::string cl_ord_id, std::string token);
   std::string SLimitSell(std::string pair, double price, double volume, std::string cl_ord_id, std::string token);
   std::string AmendOrder(std::string pair, double price, double volume, std::string cl_ord_id, std::string token);
   std::string ordID(std::string resp);
   std::string SOpenOrders(std::string ordid);
   std::string ordFill(std::string resp);

private:

   void init();

   std::string signature(const std::string& path,
			 const std::string& nonce,
			 const std::string& postdata) const;

   static size_t write_cb(char* ptr, size_t size, 
			  size_t nmemb, void* userdata);

   std::string private_method(const std::string& method,
			      const KAPI::Input& input) const;


   std::string key_;     
   std::string secret_;  
   std::string url_;     
   std::string version_; 
   CURL*  curl_;         

   

};

#endif 