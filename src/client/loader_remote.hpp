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

#include "../loader.hpp"
#include "service.hpp"

namespace nervana
{
    class loader_remote final : public loader
    {
    public:
        loader_remote() = delete;
        ~loader_remote() override;
        explicit loader_remote(std::shared_ptr<service> client, const std::string&);
        explicit loader_remote(std::shared_ptr<service> client, const nlohmann::json&);

        const std::vector<std::pair<std::string, shape_type>>& get_names_and_shapes() const override
        {
            return m_names_and_shapes;
        }
        const std::vector<std::string>& get_buffer_names() const override;
        const shape_t& get_shape(const std::string& name) const override;

        int                     record_count() const override { return m_record_count; }
        int                     batch_size() const override { return m_batch_size; }
        int                     batch_count() const override { return m_batch_count; }
        iterator                begin() override;
        iterator                end() override { return m_end_iter; }
        iterator&               get_current_iter() override;
        iterator&               get_end_iter() override { return m_end_iter; }
        const fixed_buffer_map* get_output_buffer() const override
        {
            return m_output_buffer_ptr.get();
        }
        const size_t&  position() override { return m_position; }
        void           reset() override;
        nlohmann::json get_current_config() const override { return m_config; }
        const char*    get_session_id() const override { return m_session_id.c_str(); }
    private:
        void initialize();
        void increment_position() override;

        void create_session();
        void close_session();
        void retrieve_names_and_shapes();
        void retrieve_record_count();
        void retrieve_batch_size();
        void retrieve_batch_count();
        void retrieve_next_batch();

        nlohmann::json                    m_config;
        std::string                       m_session_id;
        std::shared_ptr<service>          m_service;
        iterator                          m_current_iter;
        iterator                          m_end_iter;
        std::shared_ptr<fixed_buffer_map> m_output_buffer_ptr;
        names_and_shapes                  m_names_and_shapes;
        mutable std::vector<std::string>  m_names;
        int                               m_record_count;
        int                               m_batch_size;
        int                               m_batch_count;
        size_t                            m_position{0};
        bool                              m_shared_session{false};
        bool                              m_batch_to_fetch{true};
        bool                              m_close_session{true};
    };
}
