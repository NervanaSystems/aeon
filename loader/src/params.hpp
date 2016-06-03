#pragma once
#include <memory>
#include <random>
#include "argtype.hpp"

namespace nervana {
    class decoded_media;
    class settings;
}

typedef std::shared_ptr<nervana::decoded_media>        media_ptr;
typedef std::shared_ptr<nervana::parameter_collection> param_ptr;
typedef std::shared_ptr<nervana::settings>             settings_ptr;

enum class MediaType {
    UNKNOWN = -1,
    IMAGE = 0,
    VIDEO = 1,
    AUDIO = 2,
    TEXT = 3,
    TARGET = 4,
};

/*  ABSTRACT INTERFACES */
class nervana::decoded_media {
public:
    virtual ~decoded_media() {}
    virtual MediaType get_type() = 0;
    // virtual void fill_settings(param_ptr, default_random_engine) = 0;
};

/*  ABSTRACT INTERFACES */
class nervana::settings {
public:
    virtual ~settings() {}
};
