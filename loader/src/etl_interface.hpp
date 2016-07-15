#pragma once
#include <memory>
#include <vector>
#include <numeric>
#include <functional>
#include <exception>
#include "typemap.hpp"
#include "params.hpp"
#include "util.hpp"

namespace nervana {
    namespace interface {
        class config;
        template<typename T> class extractor;
        template<typename T, typename S> class transformer;
        template<typename T, typename S> class param_factory;
        template<typename T> class loader;
    }
}

class nervana::interface::config : public nervana::json_config_parser {
public:
    config() {}

    nervana::shape_type get_shape_type() const { return shape_type(shape, otype); }

    void base_validate() {
        if(shape.size() == 0)      throw std::invalid_argument("config missing output shape");
        if(otype.valid() == false) throw std::invalid_argument("config missing output type");
    }

protected:
    nervana::output_type otype;
    std::vector<uint32_t> shape;
};

template<typename T, typename S> class nervana::interface::param_factory {
public:
    virtual ~param_factory() {}
    virtual std::shared_ptr<S> make_params(std::shared_ptr<const T>) = 0;
};

template<typename T> class nervana::interface::extractor {
public:
    virtual ~extractor() {}
    virtual std::shared_ptr<T> extract(const char*, int) = 0;
};

template<typename T, typename S> class nervana::interface::transformer {
public:
    virtual ~transformer() {}
    virtual std::shared_ptr<T> transform(std::shared_ptr<S>, std::shared_ptr<T>) = 0;
};

template<typename T> class nervana::interface::loader {
public:
    virtual ~loader() {}
    virtual void load(char*, std::shared_ptr<T>) = 0;
};
