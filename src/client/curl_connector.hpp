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

#include "http_connector.hpp"
#include <curl/curl.h>
#include <curl/easy.h>

namespace nervana
{
    class curl_connector : public http_connector
    {
    public:
        explicit curl_connector(const std::string& address, unsigned int port);
        curl_connector() = delete;
        ~curl_connector();

        http_response get(const std::string&  endpoint,
                          const http_query_t& query = http_query_t()) override;
        http_response post(const std::string& endpoint, const std::string& body = "") override;
        http_response post(const std::string& endpoint, const http_query_t& query) override;

        http_response del(const std::string&  endpoint,
                          const http_query_t& query = http_query_t()) override;

    private:
        // used for retrieving response body
        static size_t write_callback(void* ptr, size_t size, size_t nmemb, void* stream);
        // used for sending body
        static size_t read_callback(void* ptr, size_t size, size_t nmemb, void* stream);

        void check_response(CURLcode response, const std::string& call);
        long get_http_code(CURL* curl_handle);
        std::string url_with_query(const std::string& url, const nervana::http_query_t& query);
        std::string query_to_string(const http_query_t& query);
        std::string escape(const std::string& value);

        std::string m_address;
    };
}
