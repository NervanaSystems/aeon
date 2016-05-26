/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <exception>

namespace nervana {
    template<typename T> class ArgType;
    class interface_ArgType;
    class parameter_collection;
}

// GET_MACRO and ADD_ARG work together to convert the number of arguments to
// a macro name. I've named the ultimate macros FN where N is the number of
// arguments. Ultimately these result in calls to add().
// the '#t' converts the ADD_ARGS first argument into a string.
#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,NAME,...) NAME
#define ADD_ARG(...) GET_MACRO(__VA_ARGS__,F7,F6,F5,F4)(__VA_ARGS__)
#define F4(t,desc,vs,vl) add(t,#t,desc,vs,vl)
#define F5(t,desc,vs,vl,def) add(t,#t,desc,vs,vl,(decltype(t))def)
#define F7(t,desc,vs,vl,def,minimum,maximum) add(t,#t,desc,vs,vl,(decltype(t))def,(decltype(t))minimum,(decltype(t))maximum)

//=============================================================================
//
//=============================================================================

class nervana::interface_ArgType {
public:
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
    virtual bool required() const = 0;
    virtual std::string verb_short() const = 0;
    virtual std::string verb_long() const = 0;
    virtual std::string default_value() const = 0;
    // If try_parse is successful it advances the args iterator to the next argument
    // and set value to the parsed and validated value
    virtual bool try_parse( std::deque<std::string>& args ) const = 0;
    virtual bool validate( const std::string& value ) const = 0;
    virtual bool set_value( const std::string& value ) = 0;
};

typedef std::shared_ptr<nervana::interface_ArgType> argtype_t;

//=============================================================================
//
//=============================================================================

template<typename T>
class nervana::ArgType : public nervana::interface_ArgType {
public:
    virtual std::string name() const override { return _name;  }
    virtual std::string description() const override { return _description; }
    virtual bool required() const override { return _required; }
    virtual std::string verb_short() const override { return _verb_short; }
    virtual std::string verb_long() const override { return _verb_long; }
    virtual std::string default_value() const override { return make_string(_default); }
    virtual bool set_value( const std::string& value ) override {
        bool rc = true;
        try {
            *_value = parse_value(value);
        } catch( std::exception err ) {
            rc = false;
        }
        return rc;
    }

    // If try_parse is successful it advances the args iterator to the value
    virtual bool try_parse( std::deque<std::string>& args ) const override {
        bool rc = false;
        if( (args.front() == "-"+_verb_short) || (args.front() == "--"+_verb_long) ) {
            rc = true;
        }
        return rc;
    }

    bool validate( const std::string& value ) const override;

    ArgType( const std::string& name,
            T& value,
            const std::string& description,
            const std::string& verb_short,
            const std::string& verb_long
            ) :
        _name{name},
        _value{&value},
        _description{description},
        _verb_short{verb_short},
        _verb_long{verb_long},
        _required{true},
        _default{},
        _minimum_value{},
        _maximum_value{},
        _range_valid{true}
    {
    }

    ArgType( const std::string& name,
            T& value,
            const std::string& description,
            const std::string& verb_short,
            const std::string& verb_long,
            T default_value
            ) :
        _name{name},
        _value{&value},
        _description{description},
        _verb_short{verb_short},
        _verb_long{verb_long},
        _required{false},
        _default{default_value},
        _minimum_value{},
        _maximum_value{},
        _range_valid{false}
    {
    }

    ArgType( const std::string& name,
            T& value,
            const std::string& description,
            const std::string& verb_short,
            const std::string& verb_long,
            T default_value,
            T minimum_value,
            T maximum_value
            ) :
        _name{name},
        _value{&value},
        _description{description},
        _verb_short{verb_short},
        _verb_long{verb_long},
        _required{false},
        _default{default_value},
        _minimum_value{minimum_value},
        _maximum_value{maximum_value},
        _range_valid{true}
    {
    }

private:
    ArgType() = delete;
    ArgType(const ArgType&) = delete;

    std::string make_string(const std::string& s) const { return s; }
    std::string make_string(int value) const { return std::to_string(value); }
    std::string make_string(float value) const { return std::to_string(value); }
    std::string make_string(bool value) const { return ( value ? "true" : "false" ); }
    T parse_value( const std::string& value ) const;

    std::string         _name;
    T*                  _value;
    std::string         _description;
    std::string         _verb_short;
    std::string         _verb_long;
    bool                _required;
    T                   _default;
    T                   _minimum_value;
    T                   _maximum_value;
    bool                _range_valid;
};


//=============================================================================
//
//=============================================================================

class nervana::parameter_collection {
public:
    template<typename T> void add(
                        T& value,
                        const std::string& name,
                        const std::string& description,
                        const std::string& verb_short,
                        const std::string& verb_long )
    {
        auto arg = std::make_shared<nervana::ArgType<T> >(name, value, description, verb_short, verb_long);
        _arg_list.push_back(arg);
    }

    template<typename T> void add(
                        T& value,
                        const std::string& name,
                        const std::string& description,
                        const std::string& verb_short,
                        const std::string& verb_long,
                        T default_value )
    {
        auto arg = std::make_shared<nervana::ArgType<T> >(name, value, description, verb_short, verb_long, default_value);
        _arg_list.push_back(arg);
    }

    template<typename T> void add(
                        T& value,
                        const std::string& name,
                        const std::string& description,
                        const std::string& verb_short,
                        const std::string& verb_long,
                        T default_value,
                        T minimum_value,
                        T maximum_value )
    {
        auto arg = std::make_shared<nervana::ArgType<T> >(name, value, description, verb_short, verb_long,
                                                 default_value, minimum_value, maximum_value);
        _arg_list.push_back(arg);
    }

    // first of map is the friendly name of the argument
    // second is the actual argument
    std::map<std::string,argtype_t> get_args() const;

    bool parse(const std::string& args);

private:
    std::vector<argtype_t>     _arg_list;
};
