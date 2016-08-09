#include "batch_iterator_minibatch.hpp"

BatchIteratorMinibatch::BatchIteratorMinibatch(std::shared_ptr<BatchIterator> macroBatchIterator,
                                               int minibatchSize)
    : _macroBatchIterator(macroBatchIterator),
      _minibatchSize(minibatchSize),
      _macrobatch{std::vector<size_t>{0,0}},
      _i(0)
{
    _macroBatchIterator->reset();
}

void BatchIteratorMinibatch::read(buffer_in_array& dest)
{
    if (_macrobatch.size() != dest.size()) {
        _macrobatch = buffer_in_array(std::vector<size_t>(dest.size(), (size_t) 0));
    }
    // read `_minibatchSize` items from _macrobatch into `dest`
    for(auto i = 0; i < _minibatchSize; ++i) {
        popItemFromMacrobatch(dest);
    }
}

void BatchIteratorMinibatch::reset()
{
    for (auto m: _macrobatch) {
        m->reset();
    }

    _macroBatchIterator->reset();

    _i = 0;
}

void BatchIteratorMinibatch::transferBufferItem(buffer_in* dest, buffer_in* src)
{
    try {
        dest->addItem(src->getItem(_i));
    } catch (std::exception& e) {
        dest->addException(std::current_exception());
    }
}

void BatchIteratorMinibatch::popItemFromMacrobatch(buffer_in_array& dest)
{
    // load a new macrobatch if we've already iterated through the previous one
    if(_i >= _macrobatch[0]->getItemCount()) {
        for (auto m: _macrobatch) {
            m->reset();
        }

        _macroBatchIterator->read(_macrobatch);

        _i = 0;
    }

    // because the _macrobatch Buffers may have been shuffled, and its shuffle
    // reorders the index, we can't just read a large contiguous block of
    // memory out of the _macrobatch.  We must copy out each element one at
    // a time
    for (uint idx=0; idx < _macrobatch.size(); ++idx) {
        transferBufferItem(dest[idx], _macrobatch[idx]);
    }

    _i += 1;
}
