#pragma once
#include <memory>
#include "etl_interface.hpp"

namespace nervana {
    template<typename T, typename S> class provider;
    template<typename D, typename T> class train_provider;
    class image_decoder;
}

// Do we want to have a vector of transformers so that we can cascade?
template<typename T, typename S> class nervana::provider {
public:
    provider() {}
    provider(shared_ptr<interface::extractor<T>> ex,
             shared_ptr<interface::transformer<T, S>> tr,
             shared_ptr<interface::loader<T>> lo,
             shared_ptr<interface::param_factory<T, S>> fa = nullptr)
    : _extractor(ex), _transformer(tr), _loader(lo), _factory(fa) {}

    shared_ptr<S> provide(char *inbuf, int insize,
                          char *outbuf, int outsize,
                          shared_ptr<S> pptr = nullptr)
    {
        shared_ptr<T> dec = _extractor->extract(inbuf, insize);
        shared_ptr<S> optr = _factory == nullptr ? pptr : _factory->make_params(dec);
        _loader->load(outbuf, outsize, _transformer->transform(optr, dec));
        return optr;
    }

    shared_ptr<interface::extractor<T>>        _extractor;
    shared_ptr<interface::transformer<T, S>>   _transformer;
    shared_ptr<interface::loader<T>>           _loader;
    shared_ptr<interface::param_factory<T, S>> _factory;
};

template<typename D, typename T> class nervana::train_provider {
public:
    train_provider() {}
    train_provider(const string& datum_cfg, const string& tgt_cfg) {
        _dprov = make_shared<D>(datum_cfg);
        _tprov = make_shared<T>(tgt_cfg);
    }

    void provide_pair(int idx, BufferPair* in_buf, char *datum_out, char *tgt_out)
    {
        int dsz_in, tsz_in;

        char* datum_in  = in_buf->first->getItem(idx, dsz_in);
        char* target_in = in_buf->second->getItem(idx, tsz_in);

        if (datum_in == 0) {
            return;
        }

        auto pptr = _dprov->provide(datum_in, dsz_in, datum_out, _dsz_out);
        _tprov->provide(target_in, tsz_in, tgt_out, _tsz_out, pptr);
    }

    shared_ptr<D> _dprov;
    shared_ptr<T> _tprov;
    int           _dsz_out;
    int           _tsz_out;
};


