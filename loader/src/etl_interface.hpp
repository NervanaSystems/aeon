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
        class config_info_interface;
        template<typename> class config_info;
        class config;
        template<typename T> class extractor;
        template<typename T, typename S> class transformer;
        template<typename T, typename S> class param_factory;
        template<typename T> class loader;
    }
}

class nervana::interface::config_info_interface {
public:
    virtual const std::string& name() const = 0;
    virtual void parse(nlohmann::json js) = 0;

};

static std::string dummy = "dummy";

template<typename T>
class nervana::interface::config_info : public nervana::interface::config_info_interface {
public:
    config_info(T& var, const std::string& name, nervana::json_config_parser::mode m,
                std::function<void(T&,const std::string&, const nlohmann::json&, nervana::json_config_parser::mode)> parse,
                std::function<void(T)> validate ) :
        target_variable{var},
        var_name{name},
        parse_mode{m},
        parse_function{parse},
        validate_function{validate}
    {
        std::cout << "config_info ctor " << var_name << std::endl;
    }

    const std::string& name() const override
    {
        return var_name;
    }

    void parse(nlohmann::json js) {
        parse_function(target_variable, var_name, js, parse_mode);
        std::cout << "got value for " << var_name << " = " << target_variable << std::endl;
    }

private:
    config_info() = delete;
    T&                      target_variable;
    const std::string       var_name;
    nervana::json_config_parser::mode parse_mode;
    std::function<void(T&,const std::string&, const nlohmann::json&, nervana::json_config_parser::mode)> parse_function;
    std::function<void(T)>                                           validate_function;
};


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
