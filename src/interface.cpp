/*
 Copyright 2016 Intel(R) Nervana(TM)
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

#include <iostream>
#include <sstream>

#include "interface.hpp"
#include "util.hpp"

using namespace nervana;
using namespace std;
using namespace nlohmann;

void json_configurable::verify_config(
    const std::string&                                          location,
    const vector<shared_ptr<interface::config_info_interface>>& config,
    nlohmann::json                                              js) const
{
    vector<string> ignore_list;
    string         error_key;
    string         suggestion;
    int            distance = numeric_limits<int>::max();

    json::parser_callback_t cb = [&](int depth, json::parse_event_t event, json& parsed) {
        if (depth == 1)
        {
            switch (event)
            {
            case json::parse_event_t::key:
            {
                string key   = parsed;
                bool   found = false;
                for (auto item : config)
                {
                    if (item->name() == key)
                    {
                        found = true;
                        break;
                    }
                }
                for (const string& s : ignore_list)
                {
                    if (key == s)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    for (auto item : config)
                    {
                        int test = LevenshteinDistance(item->name(), key);
                        if (test < distance)
                        {
                            distance   = test;
                            suggestion = item->name();
                        }
                    }
                    error_key = key;
                }
                break;
            }
            case json::parse_event_t::value:
            {
                if (error_key.size() > 0)
                {
                    stringstream ss;
                    ss << "config element {" << error_key << ": " << parsed
                       << "} is not understood";
                    if (distance < error_key.size() / 2)
                    {
                        ss << ", did you mean '" << suggestion << "'";
                    }
                    throw invalid_argument(ss.str());
                }
                break;
            }
            default: break;
            }
        }
        return true;
    };

    // type is required only for the top-level config
    // auto obj = js.find("type");
    // if (obj != js.end())
    // {
    //     string type = obj.value();
    //     ignore_list = split(type, ',');
    // }

    string text;
    try
    {
        text = js.dump();
        json::parse(text, cb);
    }
    catch (invalid_argument err)
    {
        throw;
    }
    catch (exception err)
    {
        // This is not an error, it just means the json is empty
        //        stringstream ss;
        //        ss << "parse error for json '" << text << "' in " << location;
        //        throw runtime_error(ss.str());
    }
}

std::string nervana::dump_default(const std::string& s)
{
    stringstream ss;
    ss << '"' << s << '"';
    return ss.str();
}

std::string nervana::dump_default(int v)
{
    return std::to_string(v);
}
std::string nervana::dump_default(uint32_t v)
{
    return std::to_string(v);
}
std::string nervana::dump_default(size_t v)
{
    return std::to_string(v);
}
std::string nervana::dump_default(float v)
{
    return std::to_string(v);
}
std::string nervana::dump_default(const std::vector<float>& v)
{
    return "[" + join(v, ",") + "]";
}
std::string nervana::dump_default(const std::vector<std::string>& v)
{
    return "[" + join(v, ",") + "]";
}
std::string nervana::dump_default(const std::uniform_real_distribution<float>& v)
{
    stringstream ss;
    ss << "{" << v.a() << "," << v.b() << "}";
    return ss.str();
}

std::string nervana::dump_default(const std::uniform_int_distribution<int>& v)
{
    stringstream ss;
    ss << "{" << v.a() << "," << v.b() << "}";
    return ss.str();
}

std::string nervana::dump_default(const std::normal_distribution<float>& v)
{
    stringstream ss;
    ss << "{" << v.mean() << "," << v.stddev() << "}";
    return ss.str();
}

std::string nervana::dump_default(const std::bernoulli_distribution& v)
{
    stringstream ss;
    ss << "{" << v.p() << "}";
    return ss.str();
}

std::string nervana::dump_default(nlohmann::json v)
{
    return v.dump();
}

std::string nervana::dump_default(std::vector<nlohmann::json> v)
{
    return "unimplemented";
}
