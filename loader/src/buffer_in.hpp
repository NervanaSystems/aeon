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

#pragma once

#include <assert.h>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <map>

class buffer_in {
public:
    explicit buffer_in(size_t size);
    virtual ~buffer_in();

    void read(std::istream& is, int size);
    void reset();
    std::vector<char>& getItem(int index);
    void addItem(const std::vector<char>&);
    void addException(std::exception_ptr);

    void shuffle(uint seed);

    int getItemCount();
    uint getSize();

private:
    buffer_in() = delete;

    std::vector<std::vector<char>> buffers;
    std::map<int, std::exception_ptr> exceptions;
};

// buffer_in_array holds a vector of buffer_in*.  Each buffer_in* holds either
// the datums or the targets.  Each buffer_in* should have the same length.
class buffer_in_array {
public:
    buffer_in_array(const std::vector<size_t>& initial_sizes)
    {
        for (auto sz : initial_sizes) {
            data.push_back(new buffer_in(sz));
        }
    }

    ~buffer_in_array()
    {
        for (auto buf : data) {
            delete buf;
        }
    }

    buffer_in* operator[](int i) { return data[i]; }
    const buffer_in* operator[](int i) const { return data[i]; }
    size_t size() const { return data.size(); }

    std::vector<buffer_in*>::iterator begin() { return data.begin(); }
    std::vector<buffer_in*>::iterator end() { return data.end(); }

private:
    std::vector<buffer_in*>    data;
};
