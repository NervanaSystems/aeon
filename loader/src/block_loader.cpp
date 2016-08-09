#include <math.h>
#include <sstream>
#include "block_loader.hpp"

using namespace std;

block_loader::block_loader(uint block_size)
: _block_size(block_size)
{}

uint block_loader::blockSize()
{
    return _block_size;
}

uint block_loader::blockCount()
{
    return ceil((float)objectCount() / (float)_block_size);
}


void block_loader_alphabet::loadBlock(buffer_in_array &dest, uint block_num)
{
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
}

void block_loader_random::loadBlock(buffer_in_array &dest, uint block_num)
{
    // load BufferPair with random bytes
    std::random_device engine;

    string object_string = randomString();
    vector<char> obj(object_string.begin(), object_string.end());
    dest[0]->addItem(obj);

    string target = randomString();
    vector<char> tgt(target.begin(), target.end());
    dest[1]->addItem(tgt);
}

string block_loader_random::randomString()
{
    stringstream s;
    std::random_device engine;
    uint x = engine();
    s << x;
    return s.str();
}
