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

namespace nervana
{
    enum class service_status_type
    {
        SUCCESS,
        END_OF_DATASET
    };

    std::string to_string(service_status_type);

    class service_status
    {
    public:
        bool           success() { return type == service_status_type::SUCCESS; }
        bool           failure() { return type != service_status_type::SUCCESS; }
        service_status_type type;
        std::string         description;
    };

    template <typename T>
    class service_response
    {
    public:
        bool           success() { return status.success(); }
        bool           failure() { return status.failure(); }
        service_status status;
        T              data;
    };

    using names_and_shapes = std::map<std::string, shape_type>;

    class service_client
    {
    public:
        virtual ~service_client() {}
        virtual unsigned long                      create_session()       = 0;
        virtual service_response<names_and_shapes> get_names_and_shapes() = 0;
        virtual service_response<fixed_buffer_map> next()                 = 0;
        virtual service_status             reset()                = 0;

        virtual service_response<int> record_count() = 0;
        virtual service_response<int> batch_size()   = 0;
        virtual service_response<int> batch_count()  = 0;
    };
}
