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

#include <memory>

#include "http_connector.hpp"
#include "../rdma/ofi.hpp"

namespace nervana
{
    class ofi_connector final : public http_connector
    {
    public:
        explicit ofi_connector(const std::string&              address,
                               unsigned int                    port,
                               std::shared_ptr<http_connector> base_connector);
        ofi_connector() = delete;
        ~ofi_connector();

        http_response get(const std::string&  endpoint,
                          const http_query_t& query = http_query_t()) override;
        http_response post(const std::string& endpoint, const std::string& body = "") override
        {
            return m_base_connector->post(endpoint, body);
        }
        http_response post(const std::string& endpoint, const http_query_t& query) override
        {
            return m_base_connector->post(endpoint, query);
        }

        http_response del(const std::string&  endpoint,
                          const http_query_t& query = http_query_t()) override;

    private:
        void connect(const std::string& address, unsigned int port);
        void disconnect();

        void register_rdma_memory(size_t size);
        void          unregister_rdma_memory();
        bool          is_rdma_registered() { return m_rdma_memory.is_registered(); }
        http_response receive_data(const std::string& endpoint, const http_query_t& query);
        http_response receive_message_data(const std::string& endpoint, const http_query_t& query);
        http_response receive_rdma_data(const std::string& endpoint, const http_query_t& query);

        std::shared_ptr<http_connector> m_base_connector;
        std::string                     m_connection_id;
        ofi::ofi                        m_ofi;
        ofi::rdma_memory                m_rdma_memory;
    };
}
