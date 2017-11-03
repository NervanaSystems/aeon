/*
 Copyright 2017 Intel(R) Nervana(TM)
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

#pragma once

#include <string>
#include <map>

namespace nervana
{
    namespace http
    {
        const int status_ok       = 200;
        const int status_created  = 201;
        const int status_accepted = 202;
        const int status_no_data  = 204;

        inline std::string merge_http_paths(const std::string& first, const std::string& second)
        {
            auto first_size  = first.size();
            auto second_size = second.size();
            if (second_size == 0)
            {
                return first;
            }
            std::string result = "";
            if (first_size > 0 && first[first_size - 1] == '/')
            {
                result = first;
            }
            else
            {
                result = first + "/";
            }
            if (second_size > 0 && second[0] == '/')
            {
                result += second.substr(1, second_size - 1);
            }
            else
            {
                result += second;
            }
            return result;
        }
    }

    struct http_response
    {
        http_response(int _code, const std::string& _data)
            : code(_code)
            , data(_data)
        {
        }
        http_response() = delete;

        int         code;
        std::string data;
    };

    using http_query_t = std::map<std::string, std::string>;

    class http_connector
    {
    public:
        virtual ~http_connector() {}
        virtual http_response get(const std::string&  endpoint,
                                  const http_query_t& query = http_query_t()) = 0;
        virtual http_response post(const std::string& endpoint, const std::string& body = "") = 0;
        virtual http_response post(const std::string& endpoint, const http_query_t& query)    = 0;

        virtual http_response del(const std::string&  endpoint,
                                  const http_query_t& query = http_query_t()) = 0;
    };
}
