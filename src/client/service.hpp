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

#include <string>

#include "../buffer_batch.hpp"
#include "../async_manager.hpp"
#include "http_connector.hpp"

namespace nervana
{
    enum class service_status_type
    {
        SUCCESS,
        FAILURE,
        END_OF_DATASET,
        UNDEFINED
    };

    std::string         to_string(service_status_type);
    service_status_type service_status_type_from_string(const std::string& status);

    class service_status
    {
    public:
        service_status(service_status_type _type, const std::string& _description)
            : type(_type)
            , description(_description)
        {
        }
        service_status(const nlohmann::json& json);
        service_status()
            : type(service_status_type::UNDEFINED)
        {
        }

        void assert_success()
        {
            if (!success())
            {
                throw std::runtime_error("service responded without success; status " +
                                         to_string());
            }
        }

        std::string to_string() const { return nervana::to_string(type) + ": " + description; }
        bool        success() { return type == service_status_type::SUCCESS; }
        bool        failure() { return type != service_status_type::SUCCESS; }
        service_status_type type;
        std::string         description;
    };

    template <typename T>
    class service_response
    {
    public:
        service_response() {}
        service_response(service_status _status, const T& _data)
            : status(_status)
            , data(_data)
        {
        }
        bool           success() { return status.success(); }
        bool           failure() { return status.failure(); }
        service_status status;
        T              data;
    };

    using names_and_shapes = std::vector<std::pair<std::string, shape_type>>;

    class next_response
    {
    public:
        next_response() {}
        next_response(std::shared_ptr<fixed_buffer_map> buffer_map)
            : data(buffer_map)
        {
        }

        bool operator==(const next_response& other)
        {
            // use serialzation to compare instances - this may not work if serialization results are not repeatable
            if (data == nullptr)
            {
                return &(other.data) == nullptr;
            }
            std::ostringstream this_serialized, other_serialized;
            data->serialize(this_serialized);
            other.data->serialize(other_serialized);
            return this_serialized.str() == other_serialized.str();
        }

        std::shared_ptr<fixed_buffer_map> data;
    };

    class service
    {
    public:
        virtual ~service() {}
        virtual service_response<std::string> create_session(const std::string& config) = 0;
        virtual service_status close_session(const std::string& id)                     = 0;
        virtual service_status reset_session(const std::string& id)                     = 0;

        virtual service_response<next_response> get_next(const std::string& id)                = 0;
        virtual service_response<names_and_shapes> get_names_and_shapes(const std::string& id) = 0;
        virtual service_response<int> get_record_count(const std::string& id)                  = 0;
        virtual service_response<int> get_batch_size(const std::string& id)                    = 0;
        virtual service_response<int> get_batch_count(const std::string& id)                   = 0;
    };

    class service_connector : public service
    {
    public:
        service_connector(std::shared_ptr<http_connector> http);
        service_connector() = delete;

        // service methods
        service_response<std::string> create_session(const std::string& config) override;
        service_status close_session(const std::string& id) override;
        service_status reset_session(const std::string& id) override;

        service_response<next_response> get_next(const std::string& id) override;
        service_response<names_and_shapes> get_names_and_shapes(const std::string& id) override;
        service_response<int> get_record_count(const std::string& id) override;
        service_response<int> get_batch_size(const std::string& id) override;
        service_response<int> get_batch_count(const std::string& id) override;

    private:
        service_response<next_response> process_data_json(const nervana::http_response& data);

        void handle_request_failure(const http_response& response);
        service_response<int> handle_single_int_response(http_response      response,
                                                         const std::string& field_name);
        void extract_status_and_json(const std::string& input,
                                     service_status&    status,
                                     nlohmann::json&    output_json);

        std::shared_ptr<http_connector> m_http;
    };

    class service_async_source : public service,
                                 public async_manager_source<service_response<next_response>>
    {
    public:
        service_async_source(std::shared_ptr<service> base_service)
            : m_base_service(base_service)
        {
        }

        // service methods
        service_response<std::string> create_session(const std::string& config) override
        {
            return m_base_service->create_session(config);
        }
        service_status close_session(const std::string& id) override
        {
            return m_base_service->close_session(id);
        }
        service_status reset_session(const std::string& id) override
        {
            return m_base_service->reset_session(id);
        }

        service_response<next_response> get_next(const std::string& id) override
        {
            return m_base_service->get_next(id);
        }
        service_response<names_and_shapes> get_names_and_shapes(const std::string& id) override
        {
            m_session_id = id;
            return m_base_service->get_names_and_shapes(id);
        }
        service_response<int> get_record_count(const std::string& id) override
        {
            return m_base_service->get_record_count(id);
        }
        service_response<int> get_batch_size(const std::string& id) override
        {
            return m_base_service->get_batch_size(id);
        }
        service_response<int> get_batch_count(const std::string& id) override
        {
            return m_base_service->get_batch_count(id);
        }

        // async_manager_source methods
        service_response<next_response>* next() override;
        size_t                           record_count() const override { return 0; }
        size_t                           elements_per_record() const override { return 0; }
        void                             reset() override {}
    private:
        std::shared_ptr<service>        m_base_service;
        service_response<next_response> m_current_next_response;
        // session_id needs to be storedp for next() which does not have id parameter
        std::string m_session_id;
    };

    class service_async
        : public service,
          public async_manager<service_response<next_response>, service_response<next_response>>
    {
    public:
        service_async(std::shared_ptr<service_async_source> base_service)
            : async_manager<service_response<next_response>,
                            service_response<next_response>>{base_service, "service_async"}
            , m_base_service(base_service)
        {
        }

        ~service_async() { finalize(); }
        // service methods
        service_response<std::string> create_session(const std::string& config) override
        {
            return m_base_service->create_session(config);
        }
        service_status close_session(const std::string& id) override
        {
            finalize();
            return m_base_service->close_session(id);
        }

        service_response<names_and_shapes> get_names_and_shapes(const std::string& id) override
        {
            return m_base_service->get_names_and_shapes(id);
        }
        service_response<int> get_record_count(const std::string& id) override
        {
            return m_base_service->get_record_count(id);
        }
        service_response<int> get_batch_size(const std::string& id) override
        {
            return m_base_service->get_batch_size(id);
        }
        service_response<int> get_batch_count(const std::string& id) override
        {
            return m_base_service->get_batch_count(id);
        }

        service_status reset_session(const std::string& id) override;
        service_response<next_response> get_next(const std::string& id) override;

        // async_manager_source methods
        size_t record_count() const override { return m_base_service->record_count(); }
        size_t elements_per_record() const override
        {
            return m_base_service->elements_per_record();
        }

        service_response<next_response>* filler() override;

    private:
        std::shared_ptr<service_async_source> m_base_service;
        service_response<next_response>*      m_current_next_response;
    };
}
