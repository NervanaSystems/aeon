#pragma once
#include <memory>
#include "etl_interface.hpp"

namespace nervana {
    template<typename T> class provider;
}

// Do we want to have a vector of transformers so that we can cascade?
template<typename T> class nervana::provider {
public:
    provider(shared_ptr<interface::extractor<T>> ex,
             shared_ptr<interface::transformer<T>> tr,
             shared_ptr<interface::loader<T>> lo)
    : _extractor(ex), _transformer(tr), _loader(lo) {
    }

    void provide(char *inbuf, int insize, char *outbuf, int outsize, param_ptr pptr)
    {
        _loader->load(outbuf, outsize,
                      _transformer->transform(pptr,
                                              _extractor->extract(inbuf, insize)));
    }

private:
    shared_ptr<interface::extractor<T>>   _extractor;
    shared_ptr<interface::transformer<T>> _transformer;
    shared_ptr<interface::loader<T>>      _loader;
};
