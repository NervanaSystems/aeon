#pragma once
#include <memory>
#include "params.hpp"

namespace nervana {
    class extractor_interface;
    class transformer_interface;
    class loader_interface;
}

class nervana::extractor_interface {
public:
    virtual ~extractor_interface();
    virtual media_ptr_t extract(char*, int) = 0;

private:
};

class nervana::transformer_interface {
public:
    virtual ~transformer_interface();
    virtual media_ptr_t transform(settings_ptr_t, const media_ptr_t&) = 0;
    virtual void fill_settings(settings_ptr_t) = 0;

private:
};

class nervana::loader_interface {
public:
    virtual ~loader_interface();
    virtual void load(char*, int, const media_ptr_t&) = 0;

private:
};
