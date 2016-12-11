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

#include "block_loader_nds_async.hpp"
#include "util.hpp"
#include "file_util.hpp"
#include "log.hpp"
#include "base64.hpp"

using namespace std;
using namespace nervana;

block_loader_nds_async::block_loader_nds_async(manifest_csv* manifest, size_t block_size)
    : block_loader_source_async(manifest)
    , m_block_size(block_size)
    , m_manifest(*manifest)
{
    for (int k = 0; k < 2; ++k)
    {
        for (size_t j = 0; j < element_count(); ++j)
        {
            m_containers[k].emplace_back();
        }
    }
}

nervana::variable_buffer_array* block_loader_nds_async::filler()
{
    variable_buffer_array* rc = get_pending_buffer();

    return rc;
}
