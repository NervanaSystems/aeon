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

#include "loader_remote.hpp"

using nlohmann::json;
using std::exception;
using std::runtime_error;
using std::map;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

using nervana::fixed_buffer_map;
using nervana::shape_type;
using nervana::shape_t;

nervana::loader_remote::loader_remote(shared_ptr<service> client, const string& config)
    : m_service(client)
    , m_current_iter(*this, false)
    , m_end_iter(*this, true)
{
    m_config = json::parse(config);
    initialize();
}

nervana::loader_remote::loader_remote(shared_ptr<service> client, const nlohmann::json& config)
    : m_config(config)
    , m_service(client)
    , m_current_iter(*this, false)
    , m_end_iter(*this, true)
{
    initialize();
}

void nervana::loader_remote::initialize()
{
    try
    {
        m_session_id = m_config["server"]["session_id"];
    }
    catch (const std::exception&)
    {
    }
    if (m_session_id.empty())
    {
        create_session();
    }

    retrieve_names_and_shapes();
    retrieve_record_count();
    retrieve_batch_size();
    retrieve_batch_count();
}

void nervana::loader_remote::create_session()
{
    try
    {
        service_response<string> response = m_service->create_session(m_config.dump());
        response.status.assert_success();
        m_session_id = response.data;
    }
    catch (const exception& ex)
    {
        throw runtime_error(string("cannot create service session: ") + ex.what());
    }
}

vector<string> nervana::loader_remote::get_buffer_names() const
{
    vector<string> names;
    for (const auto& item : m_names_and_shapes)
    {
        names.push_back(item.first);
    }
    return names;
}

shape_t nervana::loader_remote::get_shape(const string& name) const
{
    auto it = m_names_and_shapes.find(name);
    if (it == m_names_and_shapes.end())
    {
        std::stringstream ss;
        ss << "key '" << name << "' not found";
        throw std::runtime_error(ss.str());
    }
    return it->second.get_shape();
}

void nervana::loader_remote::retrieve_record_count()
{
    auto response = m_service->record_count(m_session_id);
    if (!response.success())
    {
        handle_response_failure(response.status);
        return;
    }
    m_record_count = response.data;
}

void nervana::loader_remote::retrieve_names_and_shapes()
{
    auto response = m_service->get_names_and_shapes(m_session_id);
    if (!response.success())
    {
        handle_response_failure(response.status);
        m_record_count = -1;
    }
    m_names_and_shapes = response.data;
}

void nervana::loader_remote::retrieve_batch_size()
{
    auto response = m_service->batch_size(m_session_id);
    if (!response.success())
    {
        handle_response_failure(response.status);
        m_batch_size = -1;
    }
    m_batch_size = response.data;
}

void nervana::loader_remote::retrieve_batch_count()
{
    auto response = m_service->batch_count(m_session_id);
    if (!response.success())
    {
        handle_response_failure(response.status);
        m_batch_count = -1;
    }
    m_batch_count = response.data;
}

void nervana::loader_remote::retrieve_next_batch()
{
    auto response = m_service->next(m_session_id);
    if (response.status.type != service_status_type::SUCCESS)
    {
        if (response.status.type == service_status_type::END_OF_DATASET)
        {
            m_position = m_batch_count;
        }
        else
        {
            handle_response_failure(response.status);
        }
    }
    const next_response& next = response.data;
    m_output_buffer_ptr       = next.data;
}

nervana::loader::iterator nervana::loader_remote::begin()
{
    reset();
    return m_current_iter;
}

void nervana::loader_remote::reset()
{
    auto response = m_service->reset(m_session_id);
    if (!response.status.success())
    {
        handle_response_failure(response.status);
    }
    const next_response& next = response.data;
    m_output_buffer_ptr       = next.data;
}

void nervana::loader_remote::increment_position()
{
    retrieve_next_batch();
}

void nervana::loader_remote::handle_response_failure(const service_status& status)
{
    stringstream ss;
    ss << "service response failure."
       << "status: " << to_string(status.type) << "description: " << status.description;
    throw std::runtime_error(ss.str());
}
