#pragma once

#include <memory>
#include <random>
#include <stdexcept>      // std::invalid_argument

#include "json.hpp"

namespace nervana {
    class decoded_media;
    class params;
    class json_config_parser;
}

typedef std::shared_ptr<nervana::decoded_media>        media_ptr;
typedef std::shared_ptr<nervana::json_config_parser>   config_ptr;
typedef std::shared_ptr<nervana::params>               param_ptr;

enum class MediaType {
    UNKNOWN = -1,
    IMAGE = 0,
    VIDEO = 1,
    AUDIO = 2,
    TEXT = 3,
    TARGET = 4,
    RAW = 5,
};

/*  ABSTRACT INTERFACES */
class nervana::decoded_media {
public:
    virtual ~decoded_media() {}
    virtual MediaType get_type() = 0;
};

/*  ABSTRACT INTERFACES */
class nervana::params {
public:
    virtual ~params() {}
};

class nervana::json_config_parser {
public:
    enum class mode {
        OPTIONAL,
        REQUIRED
    };

//    // pass json by value so set_config gets a non-const copy
//    virtual bool set_config(nlohmann::json js) = 0;

    template<typename T> void parse_dist(T& value, const std::string key, const nlohmann::json &js)
    {
        auto val = js.find(key);
        if (val != js.end()) {
            auto params = val->get<std::vector<typename T::result_type>>();
            set_dist_params(value, params);
        }
    }

    template<typename T, typename S> inline void set_dist_params(T& dist, S& params)
    {
        dist = T{params[0], params[1]};
    }

    // Specialization for a bernoulli coin flipping random var
    inline void set_dist_params(std::bernoulli_distribution& dist, std::vector<bool>& params)
    {
        dist = std::bernoulli_distribution{params[0] ? 0.5 : 0.0};
    }

    template<typename T> static void parse_value(
                                    T& value,
                                    const std::string key,
                                    const nlohmann::json &js,
                                    mode required=mode::OPTIONAL)
    {
        auto val = js.find(key);
        if (val != js.end()) {
            value = val->get<T>();
        } else if (required == mode::REQUIRED) {
            throw std::invalid_argument("Required Argument: " + key + " not set");
        }
    }

    template<typename T> static void parse_enum(
                                    T& value,
                                    const std::string key,
                                    const nlohmann::json &js,
                                    mode required=mode::OPTIONAL )
    {
        auto val = js.find(key);
        if (val != js.end()) {
            std::string tmp = val->get<std::string>();
            from_string(value,tmp);
        } else if (required == mode::REQUIRED) {
            throw std::invalid_argument("Required Argument: " + key + " not set");
        }
    }
};
