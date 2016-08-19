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
#include <random>
#include "buffer_in.hpp"

/*
 * A block_loader is something which can load blocks of data into a buffer_in_array
 */

namespace nervana {
    class block_loader;
    class block_loader_alphabet;
    class block_loader_random;
}

class nervana::block_loader {
public:
    virtual void loadBlock(nervana::buffer_in_array& dest, uint block_num) = 0;
    virtual uint objectCount() = 0;

    uint blockCount();
    uint blockSize();

protected:
    block_loader(uint block_size);
    uint _block_size;
};


// mock alphabet block loader for use in tests
class nervana::block_loader_alphabet : public block_loader {
public:
    block_loader_alphabet(uint block_size);
    void loadBlock(nervana::buffer_in_array &dest, uint block_num);
    uint objectCount() { return 26 * _block_size; }
};


// Random block loader used for tests
class nervana::block_loader_random : public block_loader {
public:
    block_loader_random(uint block_size) : block_loader(block_size) {}
    void loadBlock(nervana::buffer_in_array &dest, uint block_num);
    uint objectCount() { return 10; } // Not really correct, but unused for tests

    static std::string randomString();
};
