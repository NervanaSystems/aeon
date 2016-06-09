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
    virtual std::shared_ptr<T> transform(settings_ptr, std::shared_ptr<T>) = 0;
    // virtual void fill_settings(settings_ptr, const media_ptr&) = 0;
    virtual void fill_settings(settings_ptr, std::shared_ptr<T>, std::default_random_engine &) = 0;

};

template<typename T> class nervana::interface::loader {
public:
    virtual ~loader() {}
    virtual void load(char*, int, std::shared_ptr<T>) = 0;
};

// class nervana::media_family::image {
// public:
//     virtual void config(const media_family::image_config&) = 0;
// }
// class nervana::media_family::audio {
//     virtual void config(const media_family::audio_config&) = 0;

// }
