#include <sstream>

#include "mock_batch_loader.hpp"

using namespace std;

MockBatchLoader::MockBatchLoader(uint block_size)
    : BatchLoader(block_size) {
}

void MockBatchLoader::loadBlock(buffer_in_array &dest, uint block_num) {
    // load BufferPair with strings.
    // block_num 0: 'aa', 'ab', 'ac'
    // block_num 1: 'ba', 'bb', 'bc'
    // ...
    assert(_block_size == 3);
    assert(block_num < 26);

    for(uint i = 0; i < _block_size; ++i) {
        stringstream ss;
        ss << (char)('a' + block_num);
        ss << (char)('a' + i);
        string s = ss.str();

        dest[0]->addItem(vector<char>(s.begin(), s.end()));
        dest[1]->addItem(vector<char>(s.begin(), s.end()));
    }
};

uint MockBatchLoader::objectCount() {
    return 26 * 3;
}
