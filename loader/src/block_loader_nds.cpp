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

block_loader_nds::block_loader_nds(const std::string& baseurl, const std::string& token, size_t collection_id, size_t block_size,
                                   size_t shard_count, size_t shard_index)
    : block_loader_source(*this)
    , m_baseurl(baseurl)
    , m_token(token)
    , m_collection_id(collection_id)
    , m_shard_count(shard_count)
    , m_shard_index(shard_index)
{
}

nervana::encoded_record_list* block_loader_nds::filler()
{
    encoded_record_list* rc = get_pending_buffer();

    return rc;
}

// size_t block_loader_nds::object_count() const
// {
//     return m_object_count;
// }
