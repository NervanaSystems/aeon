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
    } catch( exception err ) {
        // This is not an error, it just means the json is empty
//        stringstream ss;
//        ss << "parse error for json '" << text << "' in " << location;
//        throw runtime_error(ss.str());
    }
}
