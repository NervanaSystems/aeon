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

#include <random>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <fstream>

#include "buffer_in.hpp"

using namespace std;
using namespace nervana;

void buffer_in::reset()
{
    m_buffers.clear();
}

void buffer_in::shuffle(uint32_t random_seed)
{
    std::minstd_rand0 rand_items(random_seed);
    std::shuffle(m_buffers.begin(), m_buffers.end(), rand_items);
}

vector<char>& buffer_in::get_item(int index)
{
    if (index >= (int) m_buffers.size()) {
        throw invalid_argument("index out-of-range");
    }

    auto it = m_exceptions.find(index);
    if (it != m_exceptions.end()) {
        std::rethrow_exception(it->second);
    }

    return m_buffers[index];
}

void buffer_in::add_item(const std::vector<char>& buf)
{
    m_buffers.push_back(buf);
}

void buffer_in::add_item(std::vector<char>&& buf)
{
    m_buffers.push_back(move(buf));
}

void buffer_in::add_exception(std::exception_ptr e)
{
    // add an exception to m_exceptions
    m_exceptions[m_buffers.size()] = e;

    // also add an empty vector to m_buffers to that indicies line up
    std::vector<char> empty;
    m_buffers.push_back(empty);
}

int buffer_in::get_item_count() {
    return m_buffers.size();
}

void buffer_in::read(istream& is, int size)
{
    // read `size` bytes out of `ifs` and push into buffer
    vector<char> b(size);
    is.read(b.data(), size);
    m_buffers.push_back(b);
}
