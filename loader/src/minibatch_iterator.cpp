#include "minibatch_iterator.hpp"

MinibatchIterator::MinibatchIterator(std::shared_ptr<BatchIterator> macroBatchIterator, int minibatchSize)
    : _macroBatchIterator(macroBatchIterator),
      _minibatchSize(minibatchSize),
      _macrobatch{std::vector<uint32_t>{0,0}}
{
    reset();
}

void MinibatchIterator::read(buffer_in_array& dest) {
    // read `_minibatchSize` items from _macrobatch into `dest`
    for(auto i = 0; i < _minibatchSize; ++i) {
        popItemFromMacrobatch(dest);
    }
}

void MinibatchIterator::reset() {
    _macrobatch[0]->reset();
    _macrobatch[1]->reset();

    _macroBatchIterator->reset();

    _i = 0;
}

void MinibatchIterator::transferBufferItem(buffer_in* dest, buffer_in* src) {
    // getItem from src and read it into dest
    int len;
    char* item = src->getItem(_i, len);
    dest->read(item, len);
}

void MinibatchIterator::popItemFromMacrobatch(buffer_in_array& dest) {
    // load a new macrobatch if we've already iterated through the previous one
    if(_i >= _macrobatch[0]->getItemCount()) {
        _macrobatch[0]->reset();
        _macrobatch[1]->reset();

        _macroBatchIterator->read(_macrobatch);

        _i = 0;
    }

    // because the _macrobatch Buffers may have been shuffled, and its shuffle
    // reorders the index, we can't just read a large contiguous block of
    // memory out of the _macrobatch.  We must copy out each element one at
    // a time
    transferBufferItem(dest[0], _macrobatch[0]);
    transferBufferItem(dest[1], _macrobatch[1]);

    _i += 1;
}
