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

//=============================================================================
//
//=============================================================================

class interface_ArgType {
public:
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
    virtual bool required() const = 0;
    virtual std::string verb_short() const = 0;
    virtual std::string verb_long() const = 0;
    virtual std::string default_value() const = 0;
    // If try_parse is successful it advances the args iterator to the next argument
    // and set value to the parsed and validated value
    virtual bool try_parse( std::deque<std::string>& args, std::string& value ) const = 0;
    virtual bool validate( const std::string& value ) const = 0;
};

typedef std::shared_ptr<interface_ArgType> argtype_t;

//=============================================================================
//
//=============================================================================

template<typename T>
class ArgType : public interface_ArgType {
public:
    virtual std::string name() const override { return _name;  }
    virtual std::string description() const override { return _description; }
    virtual bool required() const override { return _required; }
    virtual std::string verb_short() const override { return _verb_short; }
    virtual std::string verb_long() const override { return _verb_long; }
    virtual std::string default_value() const override { return std::to_string(_default); }

    // If try_parse is successful it advances the args iterator to the next argument
    // and set value to the parsed and validated value
    virtual bool try_parse( std::deque<std::string>& args, std::string& value ) const override {
        bool rc = false;
        if( (args.front() == "-"+_verb_short) || (args.front() == "--"+_verb_long) ) {
            args.pop_front(); // skip verb
            if(args.size()>0) {
                value = args.front();
                args.pop_front(); // skip value
                rc = true;
            }
        }
        return rc;
    }

    bool validate( const std::string& value ) const override {
        bool rc = false;
        size_t end;
        try {
            int n = std::stoi(value, &end);
            if(end == value.size()) {
                if( _range_valid == false ) {
                    rc = true;
                }
                else if( (n >= _minimum_value) && (n < _maximum_value) ) {
                    rc = true;
                }
            }
        } catch(std::exception) {

        }
        return rc;
    } 

    ArgType( const std::string& name,
            const std::string& description,
            const std::string& verb_short,
            const std::string& verb_long,
            bool required,
            T default_value
            ) :
        _name{name},
        _description{description},
        _verb_short{verb_short},
        _verb_long{verb_long},
        _required{required},
        _default{default_value},
        _range_valid{false}
    {
    }

    ArgType( const std::string& name,
            const std::string& description,
            const std::string& verb_short,
            const std::string& verb_long,
            bool required,
            T default_value,
            T minimum_value,
            T maximum_value
            ) :
        _name{name},
        _description{description},
        _verb_short{verb_short},
        _verb_long{verb_long},
        _required{required},
        _default{default_value},
        _minimum_value{minimum_value},
        _maximum_value{maximum_value},
        _range_valid{true}
    {
    }

private:
    ArgType() = delete;
    ArgType(const ArgType&) = delete;

    std::string         _name;
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

class ParameterCollection {
public:
    template<typename T> void add(
                        const std::string& name,
                        const std::string& description,
                        const std::string& verb_short,
                        const std::string& verb_long,
                        bool required,
                        T default_value )
    {
        auto arg = std::make_shared<ArgType<T> >(name, description, verb_short, verb_long, required, default_value);
        _arg_list.push_back(arg);
    }

    template<typename T> void add(
                        const std::string& name,
                        const std::string& description,
                        const std::string& verb_short,
                        const std::string& verb_long,
                        bool required,
                        T default_value,
                        T minimum_value,
                        T maximum_value )
    {
        auto arg = std::make_shared<ArgType<T> >(name, description, verb_short, verb_long,
                                                 required, default_value, minimum_value, maximum_value);
        _arg_list.push_back(arg);
    }

    std::vector<argtype_t> get_args() const;

    bool parse(const std::string& args, std::map<argtype_t,std::string>& parsedArgs);
    
private:
    // void register_arg(const ArgType& arg);
    std::vector<argtype_t>     _arg_list;
};
