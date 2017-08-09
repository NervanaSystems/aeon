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

#include "../buffer_batch.hpp"
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
                throw std::runtime_error("service responded without success with status " +
                                         to_string());
            }
        }

        std::string         to_string() const;
        bool                success() { return type == service_status_type::SUCCESS; }
        bool                failure() { return type != service_status_type::SUCCESS; }
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

    using names_and_shapes = std::map<std::string, shape_type>;

    class next_response
    {
    public:
        next_response()
            : data(nullptr)
            , position(-1)
        {
        }
        next_response(int _position, fixed_buffer_map* buffer_map)
            : data(buffer_map)
            , position(_position)
        {
        }

        bool operator==(const next_response& other)
        {
            bool result = position == other.position;

            // use serialzation to compare instances - this may not work if serialization results are not repeatable
            if (data == nullptr)
            {
                return &(other.data) == nullptr;
            }
            std::ostringstream this_serialized, other_serialized;
            data->serialize(this_serialized);
            other.data->serialize(other_serialized);
            result &= this_serialized.str() == other_serialized.str();

            return result;
        }

        fixed_buffer_map* data;
        int               position;
    };

    class service
    {
    public:
        virtual ~service() {}
        virtual service_response<std::string> create_session(const std::string& config)        = 0;
        virtual service_status close_session(const std::string& id)                                    = 0;
        virtual service_response<names_and_shapes> get_names_and_shapes(const std::string& id) = 0;
        virtual service_response<next_response> next(const std::string& id)                    = 0;
        virtual service_status reset(const std::string& id)                                    = 0;

        virtual service_response<int> record_count(const std::string& id) = 0;
        virtual service_response<int> batch_size(const std::string& id)   = 0;
        virtual service_response<int> batch_count(const std::string& id)  = 0;
    };

    class service_connector final : public service
    {
    public:
        service_connector(std::shared_ptr<http_connector> http);
        service_connector() = delete;

        service_response<std::string> create_session(const std::string& config) override;
        service_status close_session(const std::string& id) override;
        service_response<next_response> next(const std::string& id) override;
        service_status reset(const std::string& id) override;

        service_response<names_and_shapes> get_names_and_shapes(const std::string& id) override;
        service_response<int> record_count(const std::string& id) override;
        service_response<int> batch_size(const std::string& id) override;
        service_response<int> batch_count(const std::string& id) override;

    private:
        void handle_request_failure(const http_response& response);
        service_response<int> handle_single_int_response(http_response      response,
                                                         const std::string& field_name);
        void extract_status_and_json(const std::string& input,
                                     service_status&    status,
                                     nlohmann::json&    output_json);

        std::shared_ptr<http_connector> m_http;
        //TODO add double-buffering!
        fixed_buffer_map m_buffer_map;
    };
}
