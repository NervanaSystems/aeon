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

#include "../loader.hpp"
#include "service_client.hpp"

namespace nervana
{
    class loader_remote final : public loader
    {
        loader_remote(std::shared_ptr<service_client> client, const std::string&);
        loader_remote(std::shared_ptr<service_client> client, const nlohmann::json&);

        ~loader_remote() override {}
        std::map<std::string, shape_type> get_names_and_shapes() const override;
        std::vector<std::string> get_buffer_names() const override;
        shape_t get_shape(const std::string& name) const override;

        int record_count() override;
        int batch_size() override;
        int batch_count() override;

        iterator begin() override;
        iterator end() override;

        iterator& get_current_iter() override;
        iterator& get_end_iter() override;

        const fixed_buffer_map* get_output_buffer() const override;
        const size_t&           position() override;
        void                    reset() override;
        nlohmann::json          get_current_config() const override;

    private:
        void increment_position() override;

        void initialize();
        void handle_response_failure(const service_status& status);

        nlohmann::json                  m_config;
        std::shared_ptr<service_client> m_client;
        iterator                        m_current_iter;
        iterator                        m_end_iter;
        fixed_buffer_map*               m_output_buffer_ptr{nullptr};
        names_and_shapes                m_names_and_shapes;
    };
}
