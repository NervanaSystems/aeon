#pragma once
#include <memory>
#include "etl_interface.hpp"

namespace nervana {
    // Do we want to have a vector of transformers so that we can cascade?
    class provider {
    public:
        provider(shared_ptr<interface::extractor> ex,
                 shared_ptr<interface::transformer> tr,
                 shared_ptr<interface::loader> lo)
        : _extractor(ex), _transformer(tr), _loader(lo) {
        }

        void provide(char *inbuf, int insize, char *outbuf, int outsize, settings_ptr txs)
        {
            _loader->load(outbuf, outsize,
                          _transformer->transform(txs,
                                                  _extractor->extract(inbuf, insize)));
        }

    private:
        shared_ptr<interface::extractor>   _extractor;
        shared_ptr<interface::transformer> _transformer;
        shared_ptr<interface::loader>      _loader;
    };

}
