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
    class http_response
    {
    public:
        http_response(int status_code, const std::string& response)
            : m_status_code(status_code)
        , m_response(response)
        {
        }

    private:
        int         m_status_code;
        std::string m_response;
    };

    using http_query_t = std::map<std::string, std::string>;

    class http_connector
    {
    public:
        virtual ~http_connector() {}
        virtual http_response get(const std::string& url, const http_query_t& query) = 0;
        virtual http_response post(const std::string& url, const std::string& body)  = 0;
    };
}
