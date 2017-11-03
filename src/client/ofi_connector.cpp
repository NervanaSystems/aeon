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

#include "ofi_connector.hpp"
#include "../log.hpp"
#include "../json.hpp"

using std::string;

nervana::ofi_connector::ofi_connector(const string&                   address,
                                      unsigned int                    port,
                                      std::shared_ptr<http_connector> connector)
    : m_base_connector(connector)
{
    connect(address, port);
}

nervana::ofi_connector::~ofi_connector()
{
    unregister_rdma_memory();
    disconnect();
}

nervana::http_response nervana::ofi_connector::get(const string&       endpoint,
                                                   const http_query_t& query)
{
    if (endpoint.substr(endpoint.size() - 4) == "next" && !m_connection_id.empty())
    {
        return receive_data(endpoint, query);
    }
    else
    {
        return m_base_connector->get(endpoint, query);
    }
}

nervana::http_response nervana::ofi_connector::del(const string&       endpoint,
                                                   const http_query_t& query)
{
    if (!m_connection_id.empty())
    {
        http_query_t modified_query{query};
        modified_query["connection_id"] = m_connection_id;
        return m_base_connector->del(endpoint, modified_query);
    }
    else
    {
        return m_base_connector->del(endpoint, query);
    }
}

void nervana::ofi_connector::connect(const string& address, unsigned int port)
{
    ofi::connection_info conn_info(address, std::to_string(port));
    try
    {
        m_ofi.connect(conn_info);
        ofi::message connection_id;
        m_ofi.receive(connection_id);
        m_connection_id = string(connection_id.buffer(), connection_id.size());
        INFO << "connection id " << m_connection_id << " has been established on " << address << ":"
             << port;
    }
    catch (std::exception& ex)
    {
        ERR << "cannot establish RDMA connection with " << address << ":" << port << ": "
            << ex.what() << ". HTTP transfer will be used.";
    }
}

void nervana::ofi_connector::disconnect()
{
    m_ofi.disconnect();
}

void nervana::ofi_connector::register_rdma_memory(size_t size)
{
    if (m_rdma_memory.is_allocated())
    {
        if (m_rdma_memory.get_buffer_size() < size)
            throw std::runtime_error("allocated RDMA memory size is too small!");
    }
    else
    {
        m_rdma_memory.allocate(size);
    }
    if (!m_rdma_memory.is_registered())
    {
        m_ofi.register_memory(m_rdma_memory);
    }
}

void nervana::ofi_connector::unregister_rdma_memory()
{
    if (m_rdma_memory.is_registered())
        m_ofi.unregister_memory(m_rdma_memory);
    m_rdma_memory.deallocate();
}

nervana::http_response nervana::ofi_connector::receive_data(const string&       endpoint,
                                                            const http_query_t& query)
{
    http_query_t modified_query     = query;
    modified_query["connection_id"] = m_connection_id;

    auto response = m_base_connector->get(endpoint, modified_query);

    if (response.code != http::status_ok)
        return response;

    auto     json_response = nlohmann::json::parse(response.data);
    int      size;
    uint64_t key, remote_address;
    try
    {
        auto data      = json_response.at("data");
        size           = data.at("size");
        remote_address = data.at("address");
        key            = data.at("key");
    }
    catch (const nlohmann::detail::out_of_range& ex)
    {
        throw std::runtime_error(string("wrong remote response: ") + ex.what());
    }

    if (!m_rdma_memory.is_registered())
    {
        register_rdma_memory(size);
    }
    m_ofi.read_from_remote_host(m_rdma_memory, remote_address, size, key);

    response.data = string(reinterpret_cast<char*>(m_rdma_memory.get_buffer()), size);
    return response;
}
