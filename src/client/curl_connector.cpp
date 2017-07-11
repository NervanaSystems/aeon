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

#include "curl_connector.hpp"

using std::string;

namespace nervana
{
    curl_connector::curl_connector(const string& ip, unsigned int port)
        : m_ip(ip), m_port(port)
    {
    }

    http_response curl_connector::get(const std::string& url, const http_query_t& query)
    {
        return http_response();
    }

    http_response curl_connector::post(const std::string& url, const std::string& body)
    {
        return http_response();
    }
}
