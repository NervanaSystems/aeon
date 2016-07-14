/*
 Copyright 2015 Nervana Systems Inc.
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

#include <assert.h>
#include <random>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <fstream>

#include "buffer_in.hpp"

using namespace std;

buffer_in::buffer_in(int size) {
}

buffer_in::~buffer_in() {
}

void buffer_in::reset() {
    buffers.clear();
}

void buffer_in::shuffle(uint seed) {
    // TODO: instead of reseeding the shuffle, store these in a pair
    std::minstd_rand0 rand_items(seed);
    std::shuffle(buffers.begin(), buffers.end(), rand_items);
}

vector<char>& buffer_in::getItem(int index) {
    if (index >= (int) buffers.size()) {
        // TODO: why not raise exception here?  Is anyone actually
        // checking the return value of getItem to make sure it is
        // non-0?
        throw invalid_argument("index out-of-range");
    }
    return buffers[index];
}

void buffer_in::addItem(const std::vector<char>& buf) {
    buffers.push_back(buf);
}

int buffer_in::getItemCount() {
    return buffers.size();
}

void buffer_in::read(istream& is, int size) {
    // read `size` bytes out of `ifs` and push into buffer
    vector<char> b(size);
    is.read(b.data(), size);
    buffers.push_back(b);
}

//void buffer_in::read(const char* src, int size) {
//    // read `size` bytes out of `src` and push into buffer
////    resizeIfNeeded(size);
////    memcpy((void *) _cur, (void *) src, size);
////    pushItem(size);
//}
