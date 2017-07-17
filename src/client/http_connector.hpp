/*
 Copyright 2017 Nervana Systems Inc.
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
        const int status_ok = 200;
        const int status_created = 201;
        const int status_accepted = 202;
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
        virtual http_response get(const std::string& url, const http_query_t& query = http_query_t()) = 0;
        virtual http_response post(const std::string& url, const std::string& body = "")  = 0;
        virtual http_response post(const std::string& endpoint, const http_query_t& query) = 0;
    };
}
