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

#include "../interface.hpp"

namespace nervana
{
    class rdma_config;
    class remote_config;
}

class nervana::remote_config : public nervana::interface::config
{
public:
    remote_config() = default;
    remote_config(const nlohmann::json& js);

    std::string  address;
    unsigned int port{0};
    std::string  session_id;
    bool         async{true};
    bool         close_session{true};
    std::string  rdma_address;
    unsigned int rdma_port{0};

private:
    std::vector<std::shared_ptr<nervana::interface::config_info_interface>> config_list = {
        ADD_SCALAR(address, mode::REQUIRED),
        ADD_SCALAR(port, mode::REQUIRED),
        ADD_SCALAR(async, mode::OPTIONAL),
        ADD_SCALAR(rdma_address, mode::OPTIONAL),
        ADD_SCALAR(rdma_port, mode::OPTIONAL)};
    nlohmann::json rdma_json;
};
