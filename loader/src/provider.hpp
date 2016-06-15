#pragma once
#include <memory>
#include "etl_interface.hpp"

namespace nervana {
    template<typename T, typename S> class provider;
    class provider_pair;
}

// Do we want to have a vector of transformers so that we can cascade?
template<typename T, typename S> class nervana::provider {
public:
    provider(shared_ptr<interface::extractor<T>> ex,
             shared_ptr<interface::transformer<T, S>> tr,
             shared_ptr<interface::loader<T>> lo)
    : _extractor(ex), _transformer(tr), _loader(lo) {
    }

    void provide(char *inbuf, int insize, char *outbuf, int outsize, shared_ptr<S> pptr)
    {
        _loader->load(outbuf, outsize,
                      _transformer->transform(pptr,
                                              _extractor->extract(inbuf, insize)));
    }

private:
    shared_ptr<interface::extractor<T>>      _extractor;
    shared_ptr<interface::transformer<T, S>> _transformer;
    shared_ptr<interface::loader<T>>         _loader;

};

// template<typename D, typename T> class nervana::provider_pair {
// public:
//     provider_pair(const string& datum_cfg, const string& tgt_cfg) {
//         _dprov = make_shared<D>(datum_cfg);
//         _tprov = make_shared<T>(tgt_cfg);
//     }

//     BufferPair& outBuf = _out->getForWrite();
//     char* dataBuf = outBuf.first->_data + _dataOffsets[id];
//     char* targetBuf = outBuf.second->_data + _targetOffsets[id];

//     void provide(int idx, BufferPair* in_buf, char *datum_out, char *tgt_out)
//     {
//         int dsz_in, tsz_in;

//         char* datum_in  = in_buf->first->getItem(idx, dsz_in);
//         char* target_in = in_buf->second->getItem(idx, tsz_in);

//         if (datum_in == 0) {
//             return;
//         }
//         _dprov->provide(datum_in, dsz_in, datum_out, dsz_out, pptr);
//         _tprov->provide(target_in, tsz_in, target_out, tsz_out, pptr);

//     }


// private:
//     shared_ptr<provider<D, P>> _dprov;
//     shared_ptr<provider<T, P>> _tprov;
// }


