#pragma once
#include <memory>
#include "etl_interface.hpp"
#include "buffer.hpp"

namespace nervana {
    template<typename T, typename S> class provider;
    template<typename D, typename T> class train_provider;
    class image_decoder;
    class train_base;
}

// Do we want to have a vector of transformers so that we can cascade?
template<typename T, typename S> class nervana::provider {
public:
    provider() {}
    provider(std::shared_ptr<interface::extractor<T>> ex,
             std::shared_ptr<interface::transformer<T, S>> tr,
             std::shared_ptr<interface::loader<T>> lo,
             std::shared_ptr<interface::param_factory<T, S>> fa = nullptr)
    : _extractor(ex), _transformer(tr), _loader(lo), _factory(fa) {}

    std::shared_ptr<S> provide(char *inbuf, int insize, char *outbuf,
                               std::shared_ptr<S> pptr = nullptr)
    {
        std::shared_ptr<T> dec = _extractor->extract(inbuf, insize);
        std::shared_ptr<S> optr = _factory == nullptr ? pptr : _factory->make_params(dec);
        _loader->load(outbuf, _transformer->transform(optr, dec));
        return optr;
    }

    const std::shared_ptr<interface::loader<T>> get_loader() { return _loader; }

    std::shared_ptr<interface::extractor<T>>        _extractor;
    std::shared_ptr<interface::transformer<T, S>>   _transformer;
    std::shared_ptr<interface::loader<T>>           _loader;
    std::shared_ptr<interface::param_factory<T, S>> _factory;
};

class nervana::train_base {
public:
    virtual void provide_pair(int idx, BufferPair* in_buf, char *datum_out, char *tgt_out) = 0;
    virtual size_t get_d_size() = 0;
    virtual size_t get_t_size() = 0;

    virtual size_t get_d_count() = 0;
    virtual size_t get_t_count() = 0;
};

template<typename D, typename T> class nervana::train_provider : public train_base {
public:
    train_provider() {}
    train_provider(const std::string& datum_cfg, const std::string& tgt_cfg) {
        _dprov = std::make_shared<D>(datum_cfg);
        _tprov = std::make_shared<T>(tgt_cfg);
    }

    size_t get_d_size() override {return _dprov->get_loader()->get_load_size();}
    size_t get_t_size() override {return _tprov->get_loader()->get_load_size();}

    size_t get_d_count() override {return _dprov->get_loader()->get_load_count();}
    size_t get_t_count() override {return _tprov->get_loader()->get_load_count();}


    void provide_pair(int idx, BufferPair* in_buf, char *datum_out, char *tgt_out) override
    {
        int dsz_in, tsz_in;

        char* datum_in  = in_buf->first->getItem(idx, dsz_in);
        char* target_in = in_buf->second->getItem(idx, tsz_in);

        if (datum_in == 0) {
            std::cout << "no data " << idx << std::endl;
            return;
        }

        auto pptr = _dprov->provide(datum_in, dsz_in, datum_out);
        _tprov->provide(target_in, tsz_in, tgt_out, pptr);
    }

    std::shared_ptr<D> _dprov;
    std::shared_ptr<T> _tprov;
};


