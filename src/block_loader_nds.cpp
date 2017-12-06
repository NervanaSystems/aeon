/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this nds except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <sstream>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <iostream>
#include <iterator>

#include "block_loader_nds.hpp"
#include "util.hpp"
#include "file_util.hpp"
#include "log.hpp"
#include "base64.hpp"
#include "json.hpp"
#include "interface.hpp"

using namespace std;
using namespace nervana;

block_loader_nds::block_loader_nds(manifest_nds* manifest, size_t block_size)
    : async_manager<encoded_record_list, encoded_record_list>{manifest, "block_loader_nds"}
    , m_block_size{0}
    , m_block_count{manifest->block_count()}
    , m_record_count{manifest->record_count()}
    , m_elements_per_record{manifest->elements_per_record()}
{
}

nervana::encoded_record_list* block_loader_nds::filler()
{
    m_state                 = async_state::wait_for_buffer;
    encoded_record_list* rc = get_pending_buffer();
    m_state                 = async_state::processing;

    encoded_record_list* input = nullptr;
    rc->clear();

    m_state = async_state::fetching_data;
    input   = m_source->next();
    m_state = async_state::processing;

    if (input != nullptr)
    {
        input->swap(*rc);
    }
    m_state = async_state::idle;
    return rc;
}
