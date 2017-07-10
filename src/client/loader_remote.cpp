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
using std::map;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

using nervana::fixed_buffer_map;
using nervana::shape_type;
using nervana::shape_t;

nervana::loader_remote::loader_remote(shared_ptr<service_client> client, const string& config)
    : m_client(client)
    , m_current_iter(*this, false)
    , m_end_iter(*this, true)
{
    m_config = json::parse(config);
    initialize();
}

nervana::loader_remote::loader_remote(shared_ptr<service_client> client, const nlohmann::json& config)
    : m_client(client)
    , m_current_iter(*this, false)
    , m_end_iter(*this, true)
    , m_config(config)
{
    initialize();
}

void nervana::loader_remote::initialize()
{
    auto response = m_client->get_names_and_shapes();
    if(response.success())
    {
        m_names_and_shapes = response.data;
    }
    throw std::runtime_error("not implemented");
}

vector<string> nervana::loader_remote::get_buffer_names() const
{
    vector<string> names;
    for(const auto& item : m_names_and_shapes)
    {
        names.push_back(item.first);
    }
    return names;
}

map<string, shape_type> nervana::loader_remote::get_names_and_shapes() const
{
    return m_names_and_shapes;
}

shape_t nervana::loader_remote::get_shape(const string& name) const {
    auto it = m_names_and_shapes.find(name);
    if (it == m_names_and_shapes.end())
    {
        std::stringstream ss;
        ss << "key '" << name << "' not found";
        throw std::runtime_error(ss.str());
    }
    return it->second.get_shape();
}

int nervana::loader_remote::record_count()
{
    auto response = m_client->record_count();
    if(!response.success())
    {
        handle_response_failure(response.status);
        return -1;
    }
    return response.data;
}

int nervana::loader_remote::batch_size()
{
    auto response = m_client->batch_size();
    if(!response.success())
    {
        handle_response_failure(response.status);
        return -1;
    }
    return response.data;
}

int nervana::loader_remote::batch_count()
{
    auto response = m_client->batch_count();
    if(!response.success())
    {
        handle_response_failure(response.status);
        return -1;
    }
    return response.data;
}

nervana::loader::iterator nervana::loader_remote::begin()
{
    reset();
    return m_current_iter;
}

nervana::loader::iterator nervana::loader_remote::end()
{
    return m_end_iter;
}

nervana::loader::iterator& nervana::loader_remote::get_current_iter()
{
    return m_current_iter;
}

nervana::loader::iterator& nervana::loader_remote::get_end_iter()
{
    return m_end_iter;
}

const fixed_buffer_map* nervana::loader_remote::get_output_buffer() const
{
    return m_output_buffer_ptr;
}

const size_t& nervana::loader_remote::position()
{
    return -1;
}

void nervana::loader_remote::reset()
{
    auto status = m_client->reset();
    if(!status.success())
    {
        handle_response_failure(status);
    }
}

json nervana::loader_remote::get_current_config() const
{
    return m_config;
}

void nervana::loader_remote::increment_position()
{
}

void nervana::loader_remote::handle_response_failure(const service_status& status)
{
    stringstream ss;
    ss << "service response failure."
        << "status: " << to_string(status.type)
        << "description: " << status.description;
    throw std::runtime_error(ss.str());
}

