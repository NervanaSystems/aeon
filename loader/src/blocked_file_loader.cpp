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

#include "blocked_file_loader.hpp"

using namespace std;

BlockedFileLoader::BlockedFileLoader(Manifest* manifest, uint object_size, uint target_size)
: _manifest(manifest), _object_size(object_size), _target_size(target_size) {
}

void BlockedFileLoader::loadBlock(BufferPair& dest, uint block_num, uint block_size) {
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
        loadFile(dest.first->_data + (i * _object_size), _object_size, it->first);
        loadFile(dest.second->_data + (i * _target_size), _target_size, it->second);
    }
}

void BlockedFileLoader::loadFile(char* dest, uint size, const string& filename) {
    assert_exists_and_size(filename, size);
    ifstream fin(filename, ios::binary);
    fin.read(dest, size);
}

void BlockedFileLoader::assert_exists_and_size(const string& filename, uint size) {
    // ensure that filename exists and is of the correct size

    struct stat stats;
    int result = stat(filename.c_str(), &stats);
    if (result == -1) {
        stringstream ss;
        ss << "Could not find " << filename;
        throw std::runtime_error(ss.str());
    }

    off_t st_size = stats.st_size;
    if(size != st_size) {
        stringstream ss;
        ss << "file " << filename << " was expected to be " << size;
        ss << " but was instead " << st_size << endl;
        throw std::runtime_error(ss.str());
}
