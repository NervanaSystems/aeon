#pragma once
#include <memory>
#include "params.hpp"

namespace nervana {
    namespace interface {
        template<typename T> class extractor;
        template<typename T, typename S> class transformer;
        template<typename T> class loader;
    }
}

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
    virtual void load(char*, int, std::shared_ptr<T>) = 0;
};
