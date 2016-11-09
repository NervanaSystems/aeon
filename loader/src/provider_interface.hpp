/*
 Copyright 2016 Nervana Systems Inc.
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
#include "util.hpp"
#include "interface.hpp"
#include "buffer_in.hpp"
#include "buffer_out.hpp"

namespace nervana
{
    class provider_interface;
}

class nervana::provider_interface
{
public:
    virtual void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) = 0;
    virtual void post_process(buffer_out_array& out_buf) {}

    virtual const std::vector<nervana::shape_type>& get_oshapes() { return oshapes; }
    uint32_t num_inputs;
protected:
    std::vector<nervana::shape_type> oshapes;
};
