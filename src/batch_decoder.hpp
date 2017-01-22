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

#include "async_manager.hpp"
#include "buffer_batch.hpp"
#include "batch_iterator.hpp"
#include "provider_interface.hpp"

namespace nervana
{
    class batch_decoder;
}

class nervana::batch_decoder : public async_manager<encoded_record_list, fixed_buffer_map>
{
public:
    batch_decoder(batch_iterator*                            b_itor,
                  size_t                                     batch_size,
                  bool                                       single_thread,
                  bool                                       pinned,
                  const std::shared_ptr<provider_interface>& prov);

    virtual ~batch_decoder();

    virtual size_t            record_count() const override { return m_batch_size; }
    virtual size_t            elements_per_record() const override { return m_number_elements_out; }
    virtual fixed_buffer_map* filler() override;

private:
    void work(int id, encoded_record_list* in_buf, fixed_buffer_map* out_buf);

    std::vector<std::shared_ptr<provider_interface>> m_providers;
    size_t                                           m_batch_size;
    size_t                                           m_number_elements_in;
    size_t                                           m_number_elements_out;
    int                                              m_items_per_thread;
    std::vector<int>                                 m_start_inds;
    std::vector<int>                                 m_end_inds;
};
