#include <iostream>
#include <sstream>

#include "interface.hpp"
#include "util.hpp"

using namespace nervana;
using namespace std;
using namespace nlohmann;

void interface::config::verify_config(
        const std::string& location,
        const vector<shared_ptr<interface::config_info_interface>>& config,
        nlohmann::json js) const
{
    json::parser_callback_t cb = [&](int depth, json::parse_event_t event, json& parsed) {
//        switch(event) {
//        /// the parser read `{` and started to process a JSON object
//        case nlohmann::json::parse_event_t::object_start:
//            cout << "object_start" << endl;
//            break;
//        /// the parser read `}` and finished processing a JSON object
//        case nlohmann::json::parse_event_t::object_end:
//            cout << "object_end" << endl;
//            break;
//        /// the parser read `[` and started to process a JSON array
//        case nlohmann::json::parse_event_t::array_start:
//            cout << "array_start" << endl;
//            break;
//        /// the parser read `]` and finished processing a JSON array
//        case nlohmann::json::parse_event_t::array_end:
//            cout << "array_end" << endl;
//            break;
//        /// the parser read a key of a value in an object
//        case nlohmann::json::parse_event_t::key:
//            cout << "key" << endl;
//            break;
//        /// the parser finished reading a JSON value
//        case nlohmann::json::parse_event_t::value:
//            cout << "value" << endl;
//            break;
//        }

        if(event == json::parse_event_t::key) {
            string key = parsed;
            bool found = false;
            for(auto item : config) {
                if(item->name() == key) {
                    found = true;
                    break;
                }
            }
            if(!found) {
                int distance = numeric_limits<int>::max();
                string suggestion;
                for(auto item : config) {
                    int test = LevenshteinDistance(item->name(), key);
                    if(test < distance) {
                        distance = test;
                        suggestion = item->name();
                    }
                }
                stringstream ss;
                ss << "key '" << key << "'" << " not found, did you mean '" << suggestion << "'";
                throw invalid_argument(ss.str());
            }
        }
        return true;
    };

    string text;
    try {
        text = js.dump();
        json::parse(text, cb);
    } catch( invalid_argument err ) {
        throw;
    } catch( exception err ) {
        // This is not an error, it just means the json is empty
//        stringstream ss;
//        ss << "parse error for json '" << text << "' in " << location;
//        throw runtime_error(ss.str());
    }
}
