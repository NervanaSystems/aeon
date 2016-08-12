#include "batch_iterator.hpp"

batch_iterator::batch_iterator(std::shared_ptr<block_iterator> macroBatchIterator,
                               int minibatchSize)
    : _macroBatchIterator(macroBatchIterator),
      _minibatchSize(minibatchSize),
      _i(0)
{
    _macroBatchIterator->reset();
}

void batch_iterator::read(buffer_in_array& dest)
{
    if (_macrobatch == nullptr) {
        _macrobatch = std::make_shared<buffer_in_array>(dest.size());
    }
    // if (_macrobatch.size() != dest.size()) {
    //     _macrobatch = buffer_in_array(dest.size());
    // }
    // read `_minibatchSize` items from _macrobatch into `dest`
    for(auto i = 0; i < _minibatchSize; ++i) {
        popItemFromMacrobatch(dest);
    }
}

void batch_iterator::reset()
{
    for (auto m: *_macrobatch) {
        m->reset();
    }

    _macroBatchIterator->reset();

    _i = 0;
}

void batch_iterator::transferBufferItem(buffer_in* dest, buffer_in* src)
{
    try {
        dest->addItem(src->getItem(_i));
    } catch (std::exception& e) {
        dest->addException(std::current_exception());
    }
}

void batch_iterator::popItemFromMacrobatch(buffer_in_array& dest)
{
    // load a new macrobatch if we've already iterated through the previous one
    if(_i >= (*_macrobatch)[0]->getItemCount()) {
        for (auto m: *_macrobatch) {
            m->reset();
        }

        _macroBatchIterator->read(*_macrobatch);

        _i = 0;
    }

    // because the _macrobatch Buffers may have been shuffled, and its shuffle
    // reorders the index, we can't just read a large contiguous block of
    // memory out of the _macrobatch.  We must copy out each element one at
    // a time
    for (uint idx=0; idx < _macrobatch->size(); ++idx) {
        transferBufferItem(dest[idx], (*_macrobatch)[idx]);
    }

    _i += 1;
}
