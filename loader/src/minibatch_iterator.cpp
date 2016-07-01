#include "minibatch_iterator.hpp"

MinibatchIterator::MinibatchIterator(std::shared_ptr<BatchIterator> macroBatchIterator, int minibatchSize)
    : _macroBatchIterator(macroBatchIterator), _minibatchSize(minibatchSize) {
    Buffer* dataBuffer = new Buffer(0);
    Buffer* targetBuffer = new Buffer(0);
    _macrobatch = std::make_pair(dataBuffer, targetBuffer);

    reset();
}

void MinibatchIterator::read(BufferPair& dest) {
    // read `_minibatchSize` items from _macrobatch into `dest`
    for(auto i = 0; i < _minibatchSize; ++i) {
        popItemFromMacrobatch(dest);
    }
}

void MinibatchIterator::reset() {
    _macrobatch.first->reset();
    _macrobatch.second->reset();

    _macroBatchIterator->reset();

    _i = 0;
}

void MinibatchIterator::transferBufferItem(Buffer* dest, Buffer* src) {
    // getItem from src and read it into dest
    int len;
    char* item = src->getItem(_i, len);
    dest->read(item, len);
}

void MinibatchIterator::popItemFromMacrobatch(BufferPair& dest) {
    // load a new macrobatch if we've already iterated through the previous one
    if(_i >= _macrobatch.first->getItemCount()) {
        _macrobatch.first->reset();
        _macrobatch.second->reset();

        _macroBatchIterator->read(_macrobatch);

        _i = 0;
    }

    // because the _macrobatch Buffers may have been shuffled, and its shuffle
    // reorders the index, we can't just read a large contiguous block of
    // memory out of the _macrobatch.  We must copy out each element one at
    // a time
    transferBufferItem(dest.first, _macrobatch.first);
    transferBufferItem(dest.second, _macrobatch.second);

    _i += 1;
}
