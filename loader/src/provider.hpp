#pragma once
#include <memory>
#include "etl_interface.hpp"

namespace nervana {
    // Do we want to have a vector of transformers so that we can cascade?
    class provider {
    public:
        provider(shared_ptr<extractor_interface> ex,
                 shared_ptr<transformer_interface> tr,
                 shared_ptr<loader_interface> lo)
        : _extractor(ex), _transformer(tr), _loader(lo) {
        }

        void provide(char *inbuf, int insize, char *outbuf, int outsize, settings_ptr txs)
        {
            _loader->load(outbuf, outsize,
                          _transformer->transform(txs,
                                                  _extractor->extract(inbuf, insize)));
        }

    private:
        shared_ptr<extractor_interface>   _extractor;
        shared_ptr<transformer_interface> _transformer;
        shared_ptr<loader_interface>      _loader;
    };

}
