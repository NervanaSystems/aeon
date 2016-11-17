/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <fstream>
#include <sstream>

#include "json.hpp"
#include "manifest_nds.hpp"
#include "interface.hpp"

using namespace std;
using namespace nervana;

manifest_nds::manifest_nds(const std::string& filename)
{
    // parse json
    nlohmann::json j;
    try
    {
        ifstream ifs(filename);
        ifs >> j;
    }
    catch (std::exception& ex)
    {
        stringstream ss;
        ss << "Error while parsing manifest json: " << filename << " : ";
        ss << ex.what();
        throw std::runtime_error(ss.str());
    }

    // extract manifest params from parsed json
    try
    {
        interface::config::parse_value(baseurl, "url", j, interface::config::mode::REQUIRED);

        auto val = j.find("params");
        if (val != j.end())
        {
            nlohmann::json params = *val;
            interface::config::parse_value(token, "token", params, interface::config::mode::REQUIRED);
            interface::config::parse_value(collection_id, "collection_id", params, interface::config::mode::REQUIRED);
        }
        else
        {
            throw std::runtime_error("couldn't find key 'params' in nds manifest file.");
        }
    }
    catch (std::exception& ex)
    {
        stringstream ss;
        ss << "Error while pulling config out of manifest json: " << filename << " : ";
        ss << ex.what();
        throw std::runtime_error(ss.str());
    }
}

string manifest_nds::cache_id()
{
    stringstream contents;
    contents << baseurl << collection_id;
    std::size_t  h = std::hash<std::string>()(contents.str());
    stringstream ss;
    ss << std::hex << h;
    return ss.str();
}

bool manifest_nds::is_likely_json(const std::string filename)
{
    // check the first character of the file to see if it is a json
    // object.  If so, we want to parse this as an NDS Manifest
    // instead of a CSV
    ifstream f(filename);
    char     first_char;

    f.get(first_char);

    return first_char == '{';
}
