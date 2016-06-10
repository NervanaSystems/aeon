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

#include <sys/stat.h>

#include <cassert>
#include <sstream>

#include "batch_file_loader.hpp"

using namespace std;

BatchFileLoader::BatchFileLoader(Manifest* manifest)
: _manifest(manifest) {
}

void BatchFileLoader::loadBlock(BufferPair& dest, uint block_num, uint block_size) {
    // NOTE: thread safe so long as you aren't modifying the manifest
    // NOTE: dest memory must already be allocated at the correct size

    uint begin_i = block_num * block_size;
    uint end_i = (block_num + 1) * block_size;

    // ensure we stay within bounds of manifest
    assert(begin_i <= _manifest->getSize());
    assert(end_i <= _manifest->getSize());

    // TODO: move index offset logic and bounds asserts into Manifest
    // interface to more easily support things like offset/limit queries
    auto begin_it = _manifest->begin() + begin_i;
    auto end_it = _manifest->begin() + end_i;

    uint i = 0;
    for(auto it = begin_it; it != end_it; ++it, ++i) {
        // load both object and target files into respective buffers
        //
        // NOTE: if at some point in the future, loadFile is loading
        // files from a network like s3 it may make sense to use multiple
        // threads to make loads faster.  multiple threads would only
        // slow down reads from a magnetic disk.
        loadFile(dest.first, it->first);
        loadFile(dest.second, it->second);
    }
}

void BatchFileLoader::loadFile(Buffer* buff, const string& filename) {
    off_t size = getSize(filename);
    ifstream fin(filename, ios::binary);
    buff->read(fin, size);
}

off_t BatchFileLoader::getSize(const string& filename) {
    // ensure that filename exists and get its size

    struct stat stats;
    int result = stat(filename.c_str(), &stats);
    if (result == -1) {
        stringstream ss;
        ss << "Could not find " << filename;
        throw std::runtime_error(ss.str());
    }

    return stats.st_size;
}
