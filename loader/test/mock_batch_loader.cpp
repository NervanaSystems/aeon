#include <sstream>

#include "mock_batch_loader.hpp"

using namespace std;

MockBatchLoader::MockBatchLoader(uint block_size)
    : BatchLoader(block_size) {
}

void MockBatchLoader::loadBlock(buffer_in_array &dest, uint block_num) {
    // load BufferPair with strings.
    // block_num 0: 'Aa', 'Ab', 'Ac'
    // block_num 1: 'Ba', 'Bb', 'Bc'
    // ...
    assert(_block_size < 26);
    assert(block_num < 26);

    for(uint i = 0; i < _block_size; ++i) {
        stringstream ss;
        ss << (char)('A' + block_num);
        ss << (char)('a' + i);
        string s = ss.str();

        for (auto d: dest) {
            d->addItem(vector<char>(s.begin(), s.end()));
        }
    }
};

uint MockBatchLoader::objectCount() {
    return 26 * _block_size;
}
