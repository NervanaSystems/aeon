#pragma once
#include <memory>
#include "params.hpp"

namespace nervana {
    namespace interface {
        template<typename T> class extractor;
        template<typename T> class transformer;
        template<typename T> class loader;
    }
}

template<typename T> class nervana::interface::extractor {
public:
    virtual ~extractor() {}
    virtual std::shared_ptr<T> extract(char*, int) = 0;
};

template<typename T> class nervana::interface::transformer {
public:
    virtual ~transformer() {}
    virtual std::shared_ptr<T> transform(param_ptr, std::shared_ptr<T>) = 0;
};

template<typename T> class nervana::interface::loader {
public:
    virtual ~loader() {}
    virtual void load(char*, int, std::shared_ptr<T>) = 0;
};
