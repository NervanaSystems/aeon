#pragma once
#include <memory>
#include <vector>
#include <numeric>
#include <functional>
#include <exception>

#include "typemap.hpp"
#include "util.hpp"
#include "json.hpp"

namespace nervana {
    namespace interface {
        class config_info_interface;
        template<typename> class config_info;
        class config;
        template<typename T> class extractor;
        template<typename T, typename S> class transformer;
        template<typename T, typename S> class param_factory;
        template<typename T> class loader;
        class decoded_media;
        class params;
    }
}

class nervana::interface::config_info_interface {
public:
    virtual const std::string& name() const = 0;
    virtual void parse(nlohmann::json js) = 0;

};

class nervana::interface::config {
public:
    config() {}

    nervana::shape_type get_shape_type() const { return shape_type(shape, otype); }

    void base_validate() {
        if(shape.size() == 0)      throw std::invalid_argument("config missing output shape");
        if(otype.valid() == false) throw std::invalid_argument("config missing output type");
    }

    enum class mode {
        OPTIONAL,
        REQUIRED
    };

#define ADD_SCALAR(var, mode) \
    std::make_shared<interface::config_info<decltype(var)>>( var, #var, mode, parse_value<decltype(var)>, [](decltype(var) v){} )
#define ADD_DISTRIBUTION(var, mode) \
    std::make_shared<interface::config_info<decltype(var)>>( var, #var, mode, parse_dist<decltype(var)>, [](decltype(var) v){} )

    template<typename T, typename S> static void set_dist_params(T& dist, S& params)
    {
        dist = T{params[0], params[1]};
    }

    // Specialization for a bernoulli coin flipping random var
    static void set_dist_params(std::bernoulli_distribution& dist, std::vector<bool>& params)
    {
        dist = std::bernoulli_distribution{params[0] ? 0.5 : 0.0};
    }

    template<typename T> static void parse_dist(T& value, const std::string& key, const nlohmann::json& js,
                                         mode required=mode::OPTIONAL)
    {
        auto val = js.find(key);
        if (val != js.end()) {
            auto params = val->get<std::vector<typename T::result_type>>();
            set_dist_params(value, params);
        }
    }

    template<typename T> static void parse_value(
                                    T& value,
                                    const std::string& key,
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

protected:
    nervana::output_type otype;
    std::vector<uint32_t> shape;
};


template<typename T>
class nervana::interface::config_info : public nervana::interface::config_info_interface {
public:
    config_info(T& var, const std::string& name, nervana::interface::config::mode m,
                std::function<void(T&,const std::string&, const nlohmann::json&, nervana::interface::config::mode)> parse,
                std::function<void(T)> validate ) :
        target_variable{var},
        var_name{name},
        parse_mode{m},
        parse_function{parse},
        validate_function{validate}
    {
    }

    const std::string& name() const override
    {
        return var_name;
    }

    void parse(nlohmann::json js) {
        parse_function(target_variable, var_name, js, parse_mode);
    }

private:
    config_info() = delete;
    T&                      target_variable;
    const std::string       var_name;
    nervana::interface::config::mode parse_mode;
    std::function<void(T&,const std::string&, const nlohmann::json&, nervana::interface::config::mode)> parse_function;
    std::function<void(T)>                                           validate_function;
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


/*  ABSTRACT INTERFACES */
class nervana::interface::decoded_media {
public:
    virtual ~decoded_media() {}
};

/*  ABSTRACT INTERFACES */
class nervana::interface::params {
public:
    virtual ~params() {}
};
