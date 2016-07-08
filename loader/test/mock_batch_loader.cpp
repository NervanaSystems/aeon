#include <sstream>

#include "mock_batch_loader.hpp"

using namespace std;

void MockBatchLoader::loadBlock(buffer_in_array &dest, uint block_num, uint block_size) {
    // load BufferPair with strings.
    // block_num 0: 'aa', 'ab', 'ac'
    // block_num 1: 'ba', 'bb', 'bc'
    // ...
    assert(block_size == 3);
    assert(block_num < 26);

    for(uint i = 0; i < block_size; ++i) {
        stringstream ss;
        ss << (char)('a' + block_num);
        ss << (char)('a' + i);
        string s = ss.str();

        dest[0]->read(s.c_str(), s.length());
        dest[1]->read(s.c_str(), s.length());
    }
};

uint MockBatchLoader::objectCount() {
    return 26 * 3;
}
