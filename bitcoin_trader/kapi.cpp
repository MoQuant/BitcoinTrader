#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <string>
#include <cstring>
#include <ctime>
#include <cerrno>
#include <curl/curl.h>
#include <map>

#include <openssl/buffer.h>
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <openssl/bio.h>

#include <thread>
#include <unistd.h>

#include "kapi.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <time.h>
#include <chrono>

#define CURL_VERBOSE 0L 

using namespace boost::property_tree;

// Builds string result from an inputted map
std::string slicer(std::map<std::string, std::string> x){
   std::string result = "{";
   for(auto & entry : x){
      if(entry.first == "order_qty" || entry.first == "limit_price" || entry.first == "snap_orders" || entry.first == "snap_trades" || entry.first == "limit_price" || entry.first == "order_qty"){
         result += "\"" + entry.first + "\":" + entry.second + ",";
      } else {
         result += "\"" + entry.first + "\":\"" + entry.second + "\",";
      }
   }
   result.pop_back();
   result += "}";
   return result;
}

// Encrypts string data in SHA256 to be passed to Kraken containing private data
static std::vector<unsigned char> sha256(const std::string& data)
{
   std::vector<unsigned char> digest(SHA256_DIGEST_LENGTH);

   SHA256_CTX ctx;
   SHA256_Init(&ctx);
   SHA256_Update(&ctx, data.c_str(), data.length());
   SHA256_Final(digest.data(), &ctx);

   return digest;
}

// Decodes a base64 string which takes part in the encryption of the secret key
static std::vector<unsigned char> b64_decode(const std::string& data) 
{
   BIO* b64 = BIO_new(BIO_f_base64());
   BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);

   BIO* bmem = BIO_new_mem_buf((void*)data.c_str(),data.length());
   bmem = BIO_push(b64, bmem);
   
   std::vector<unsigned char> output(data.length());
   int decoded_size = BIO_read(bmem, output.data(), output.size());
   BIO_free_all(bmem);

   if (decoded_size < 0)
      throw std::runtime_error("failed while decoding base64.");
   
   return output;
}

// Encodes a base64 string which takes part in the encryption of the order
static std::string b64_encode(const std::vector<unsigned char>& data) 
{
   BIO* b64 = BIO_new(BIO_f_base64());
   BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);

   BIO* bmem = BIO_new(BIO_s_mem());
   b64 = BIO_push(b64, bmem);
   
   BIO_write(b64, data.data(), data.size());
   BIO_flush(b64);

   BUF_MEM* bptr = NULL;
   BIO_get_mem_ptr(b64, &bptr);
   
   std::string output(bptr->data, bptr->length);
   BIO_free_all(b64);

   return output;
}

// Takes part in encrypting the post data sent to Kraken
static std::vector<unsigned char> 
hmac_sha512(const std::vector<unsigned char>& data, 
	    const std::vector<unsigned char>& key)
{   
   unsigned int len = EVP_MAX_MD_SIZE;
   std::vector<unsigned char> digest(len);

   HMAC_CTX *ctx = HMAC_CTX_new();
   if (ctx == NULL) {
       throw std::runtime_error("cannot create HMAC_CTX");
   }

   HMAC_Init_ex(ctx, key.data(), key.size(), EVP_sha512(), NULL);
   HMAC_Update(ctx, data.data(), data.size());
   HMAC_Final(ctx, digest.data(), &len);
   
   HMAC_CTX_free(ctx);
   
   return digest;
}

// Takes a map and converts it to url encoded data
static std::string build_query(const KAPI::Input& input)
{
   std::ostringstream oss;
   KAPI::Input::const_iterator it = input.begin();
   for (; it != input.end(); ++it) {
      if (it != input.begin()) oss << '&';  // delimiter
      oss << it->first <<'='<< it->second;
   }

   return oss.str();
}

// Generates a nonce which is important to include in all of the private api requests
static std::string create_nonce()
{
   std::ostringstream oss;

   timeval tp;
   if (gettimeofday(&tp, NULL) != 0) {
      oss << "gettimeofday() failed: " << strerror(errno); 
      throw std::runtime_error(oss.str());
   }
   else {
      // format output string 
      oss << std::setfill('0') 
	  << std::setw(10) << tp.tv_sec 
	  << std::setw(6)  << tp.tv_usec;
   }
   return oss.str();
}

// Initializes class and takes the key and secret as arguments
KAPI::KAPI(const std::string& key, const std::string& secret)
   :key_(key), secret_(secret), url_("https://api.kraken.com"), version_("0") 
{ 
   init(); 
}

// Initial instructions for API regarding CURL for REST requests
void KAPI::init()
{
   curl_ = curl_easy_init();
   CURLcode code = curl_global_init(CURL_GLOBAL_ALL);
   if (curl_) {
      curl_easy_setopt(curl_, CURLOPT_VERBOSE, CURL_VERBOSE);
      curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYPEER, 1L);
      curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYHOST, 2L);
      curl_easy_setopt(curl_, CURLOPT_USERAGENT, "Kraken C++ API Client");
      curl_easy_setopt(curl_, CURLOPT_POST, 1L);

      curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, KAPI::write_cb);
   }
   else {
      throw std::runtime_error("can't create curl handle");
   }
}

// Generates an encoded string which takes the base64 encoder/decoder and sha256/512
// encryption to pass safe messages
std::string KAPI::signature(const std::string& path, 
			    const std::string& nonce, 
			    const std::string& postdata) const
{
   std::vector<unsigned char> data(path.begin(), path.end());

   std::vector<unsigned char> nonce_postdata = sha256(nonce + postdata);

   data.insert(data.end(), nonce_postdata.begin(), nonce_postdata.end());

   return b64_encode( hmac_sha512(data, b64_decode(secret_)) );
}

// Callback function used to convert bytes to strings in the REST request function
size_t KAPI::write_cb(char* ptr, size_t size, size_t nmemb, void* userdata)
{
   std::string* response = reinterpret_cast<std::string*>(userdata);
   size_t real_size = size * nmemb;

   response->append(ptr, real_size);
   return real_size;
}

// This is the private REST request function which fetches balances and places orders
std::string KAPI::private_method(const std::string& method, 
				 const KAPI::Input& input) const
{   
   std::string path = "/" + version_ + "/private/" + method;
   std::string method_url = url_ + path;

   curl_easy_setopt(curl_, CURLOPT_URL, method_url.c_str());

   std::string nonce = create_nonce();
   std::string postdata = "nonce=" + nonce;

   if (!input.empty())
      postdata = postdata + "&" + build_query(input);
   curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, postdata.c_str());

   curl_slist* chunk = NULL;

   std::string key_header =  "API-Key: "  + key_;
   std::string sign_header = "API-Sign: " + signature(path, nonce, postdata);

   chunk = curl_slist_append(chunk, key_header.c_str());
   chunk = curl_slist_append(chunk, sign_header.c_str());
   curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, chunk);
   
   std::string response;
   curl_easy_setopt(curl_, CURLOPT_WRITEDATA, static_cast<void*>(&response));

   CURLcode result = curl_easy_perform(curl_);

   curl_slist_free_all(chunk);
  
   // check perform result
   if (result != CURLE_OK) {
      std::ostringstream oss;
      oss << "curl_easy_perform() failed: " << curl_easy_strerror(result);
      throw std::runtime_error(oss.str());
   }
   
   return response;
}

// Fetches entire account balance list
std::string KAPI::account_balance()
{
   KAPI::Input in;
   return private_method("Balance", in);
}

// Fetches account balance only for USD
double KAPI::AccountBalance(){
   std::string balance = account_balance();
   std::stringstream ss(balance);
   ptree acct;
   read_json(ss, acct);
   for(ptree::const_iterator it = acct.begin(); it != acct.end(); ++it){
      if(it->first == "result"){
         for(ptree::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt){
            if(jt->first == "ZUSD"){
               return atof(jt->second.get_value<std::string>().c_str());
            }
         }
      }
   }
   return 0;
}

// Fetches trading balance
std::string KAPI::trading_balance()
{
   KAPI::Input in;
   in.insert({"asset", "USD"});
   return private_method("TradeBalance", in);
}

// Edits the order in order to get the best fill price
std::string KAPI::edit_order(std::string pair, std::string txid, double volume, double price)
{
   KAPI::Input in;
   in.insert({"txid", txid});
   in.insert({"order_qty", std::to_string(volume)});
   in.insert({"limit_price", std::to_string(price)});
   return private_method("AmendOrder", in);
}

// Conducts a limit buy order
std::string KAPI::limit_buy(std::string pair, double price, double volume, std::string clordid)
{
   KAPI::Input in;
   in.insert({"pair", pair});
   in.insert({"type", "buy"});
   in.insert({"ordertype", "limit"});
   in.insert({"price", std::to_string(price)});
   in.insert({"volume", std::to_string(volume)});
   in.insert({"cl_ord_id", clordid});
   return private_method("AddOrder", in);
}

// Conducts a limit sell order
std::string KAPI::limit_sell(std::string pair, double price, double volume, std::string clordid)
{
   KAPI::Input in;
   in.insert({"pair", pair});
   in.insert({"type", "sell"});
   in.insert({"ordertype", "limit"});
   in.insert({"price", std::to_string(price)});
   in.insert({"volume", std::to_string(volume)});
   in.insert({"cl_ord_id", clordid});
   return private_method("AddOrder", in);
}

// Conducts a market sell order
std::string KAPI::market_sell(std::string pair, double volume){
   KAPI::Input in;
   in.insert({"pair", pair});
   in.insert({"type", "sell"});
   in.insert({"ordertype", "market"});
   in.insert({"volume", std::to_string(volume)});
   return private_method("AddOrder", in);
}

// Conducts a market buy order
std::string KAPI::market_buy(std::string pair, double volume){
   KAPI::Input in;
   in.insert({"pair", pair});
   in.insert({"type", "buy"});
   in.insert({"ordertype", "market"});
   in.insert({"volume", std::to_string(volume)});
   return private_method("AddOrder", in);
}

// Cancels an order
std::string KAPI::cancel_order(std::string pair, std::string txid){
   KAPI::Input in;
   in.insert({"pair", pair});
   in.insert({"txid", txid});
   return private_method("CancelOrder", in);
}

// Generates a list of open orders
std::string KAPI::open_orders(std::string cl_ord_id){
   KAPI::Input in;
   in.insert({"trades", "true"});
   in.insert({"cl_ord_id", cl_ord_id});
   return private_method("OpenOrders", in);
}

// Fetch authenticated token to be passed into the trading websocket
std::string KAPI::get_token(){
   KAPI::Input in;
   std::string pak = private_method("GetWebSocketsToken", in);
   std::stringstream ss(pak);
   ptree tokin;
   read_json(ss, tokin);
   for(ptree::const_iterator it = tokin.begin(); it != tokin.end(); ++it){
      if(it->first == "result"){
         for(ptree::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt){
            if(jt->first == "token"){
               return jt->second.get_value<std::string>();
            }
         }
      }
   }
   return "Failed";
}

// Retreive the transaction id of a limit buy or limit sell
std::string KAPI::tID(std::string response){
   std::stringstream ss(response);
   ptree pt;
   read_json(ss, pt);

   try {
      // Navigate to the txid array
      ptree txid_node = pt.get_child("result.txid");

      // Extract the first element of the txid array
      return txid_node.begin()->second.get_value<std::string>();
   } catch(...) {
      return "No Transaction ID";
   }
}

// Retreive amended order id
std::string KAPI::eID(std::string response){
   std::stringstream ss(response);
   ptree pt;
   read_json(ss, pt);

   try {
      // Navigate to the txid array
      ptree txid_node = pt.get_child("result.amend_id");

      // Extract the first element of the txid array
      return txid_node.begin()->second.get_value<std::string>();
   } catch(...) {
      return "No Transaction ID";
   }
}

// Checks to see if order has filled by calling open orders
std::string KAPI::Filled(std::string txid, std::string cl_ord_id){
   //std::this_thread::sleep_for(std::chrono::milliseconds(300));
   std::string x = open_orders(cl_ord_id);
   std::stringstream ss(x);
   ptree data;
   read_json(ss, data);
   for(ptree::const_iterator it = data.begin(); it != data.end(); ++it){
      if(it->first == "result"){
         for(ptree::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt){
            if(jt->first == "open"){
               for(ptree::const_iterator kt = jt->second.begin(); kt != jt->second.end(); ++kt){
                  if(kt->first == txid){
                     return "Filling";
                  }
               }
            }
         }
      }
   }
   return "Filled";
}

// Limit buy order engine which edits the orders price until it is filled
std::string KAPI::LimitBuy(std::string pair, double volume, std::string clordid){
   std::string status = "NotFilled";
   double c = 0.01;
   std::string response = limit_buy(pair, khigh_bid+c, volume, clordid);
   std::cout << "Placing buy order: " << response << std::endl;
   std::this_thread::sleep_for(std::chrono::milliseconds(300));
   std::string txid = tID(response);
   while(true){
      status = Filled(txid, clordid);
      if(status == "Filled"){
         return status;
      } else {
         std::cout << "Editing Buy Order for Price: " << khigh_bid+c << std::endl;
         response = edit_order(pair, txid, volume, khigh_bid+c);
         //txid = eID(response);
         c += 0.03;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
   }
   return status;
}

// Limit sell order engine which edits the orders price until it is filled
std::string KAPI::LimitSell(std::string pair, double volume, std::string clordid){
   std::string status = "NotFilled";
   double c = 0.01;
   std::string response = limit_sell(pair, klow_ask-c, volume, clordid);
   std::cout << "Placing sell order: " << response << std::endl;
   std::this_thread::sleep_for(std::chrono::milliseconds(300));
   std::string txid = tID(response);
   while(true){
      status = Filled(txid, clordid);
      if(status == "Filled"){
         return status;
      } else {
         std::cout << "Editing Sell Order for Price: " << klow_ask-c << std::endl;
         response = edit_order(pair, txid, volume, klow_ask-c);
         c += 0.03;
         //txid = eID(response);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
   }
   return status;
}

// Returns a limit buy string to be passed to the trading websocket
std::string KAPI::SLimitBuy(std::string pair, double price, double volume, std::string cl_ord_id, std::string token){
   KAPI::Input outer, inner;
   inner.insert({"order_type", "limit"});
   inner.insert({"side", "buy"});
   inner.insert({"limit_price", std::to_string(price)});
   inner.insert({"order_qty", std::to_string(volume)});
   inner.insert({"symbol", pair});
   inner.insert({"cl_ord_id", cl_ord_id});
   inner.insert({"token", token});

   std::string A = slicer(inner);

   std::string result = "{\"method\":\"add_order\", \"params\":" + A + "}";

   return result;
}

// Returns a limit sell string to be passed to the trading websocket
std::string KAPI::SLimitSell(std::string pair, double price, double volume, std::string cl_ord_id, std::string token){
   KAPI::Input inner;
   inner.insert({"order_type", "limit"});
   inner.insert({"side", "sell"});
   inner.insert({"limit_price", std::to_string(price)});
   inner.insert({"order_qty", std::to_string(volume)});
   inner.insert({"symbol", pair});
   inner.insert({"cl_ord_id", cl_ord_id});
   inner.insert({"token", token});

   std::string A = slicer(inner);

   std::string result = "{\"method\":\"add_order\", \"params\":" + A + "}";

   return result;
}

// Gets the orderid of a response in the trading websocket, if return = "0" this is not a limit buy/sell response
std::string KAPI::ordID(std::string resp){
   ptree df;
   std::stringstream ss(resp);
   read_json(ss, df);
   int c = 0;
   for(ptree::const_iterator it = df.begin(); it != df.end(); ++it){
      if(c == 1){
         for(ptree::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt){
            if(jt->first == "order_id"){
               return jt->second.get_value<std::string>();
            }
         }
      }
      if(it->first == "method"){
         if(it->second.get_value<std::string>() == "add_order"){
            c += 1;
         }
      }
   }
   return "0";
}

// States when an order has been filled from the trading websocket, if return = "0" this is not an order response
std::string KAPI::ordFill(std::string resp){
   ptree df;
   std::stringstream ss(resp);
   read_json(ss, df);
   int c = 0;
   for(ptree::const_iterator it = df.begin(); it != df.end(); ++it){
      if(c == 2){
         for (const auto& item : df.get_child("data")) {
            std::string order_status = item.second.get<std::string>("order_status");
            return order_status;
         }
      }
      if(it->first == "channel"){
         if(it->second.get_value<std::string>() == "executions"){
            c += 1;
         }
      }
      if(it->first == "type"){
         if(it->second.get_value<std::string>() == "update"){
            c += 1;
         }
      }
   }
   return "0";
}

// Trading websocket function to see if there are any open orders on the book
std::string KAPI::SOpenOrders(std::string token){
   KAPI::Input inner;
   inner.insert({"channel", "executions"});
   inner.insert({"token", token});
   inner.insert({"snap_orders", "true"});
   inner.insert({"snap_trades", "false"});
   std::string A = slicer(inner);
   return "{\"method\":\"subscribe\",\"params\":" + A + "}";
}

// Trading websocket function to edit the price until the order gets filled
std::string KAPI::AmendOrder(std::string pair, double price, double volume, std::string cl_ord_id, std::string token){
   KAPI::Input inner;
   inner.insert({"cl_ord_id", cl_ord_id});
   inner.insert({"limit_price", std::to_string(price)});
   inner.insert({"order_qty", std::to_string(volume)});
   inner.insert({"token", token});
   std::string A = slicer(inner);
   return "{\"method\":\"amend_order\",\"params\":" + A + "}";
}