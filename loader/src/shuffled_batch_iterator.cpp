#include <vector>
#include <algorithm>
#include <random>

#include "shuffled_batch_iterator.hpp"

ShuffledBatchIterator::ShuffledBatchIterator(shared_ptr<BatchLoader> loader, uint block_size, uint seed)
    : _loader(loader), _block_size(block_size), _rand(seed), _seed(seed) {
    _epoch = 0;

    // fill indices with integers from  0 to _count.  indices can then be
    // shuffled and used to iterate randomly through the blocks.
    uint _count = ceil((float)_loader->objectCount() / (float)_block_size);
    for(uint i = 0; i < _count; ++i) {
        _indices.push_back(i);
    }

    reset();
}

void ShuffledBatchIterator::shuffle() {
    std::shuffle(_indices.begin(), _indices.end(), _rand);
}

void ShuffledBatchIterator::read(BufferPair &dest) {
    _loader->loadBlock(dest, *_it, _block_size);

    // shuffle the objects in BufferPair dest
    // seed the shuffle with the seed passed in the constructor + the _epoch
    // to ensure that the buffer shuffles are deterministic wrt the input seed.
    // HACK: pass the same seed to both shuffles to ensure that both buffers
    // are shuffled in the same order.
    dest.first->shuffle(_seed + _epoch);
    dest.second->shuffle(_seed + _epoch);

    ++_it;

    if(_it != _indices.end()) {
        reset();
    }
}

void ShuffledBatchIterator::reset() {
    shuffle();

    _it = _indices.begin();

    ++_epoch;
}
