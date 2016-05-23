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

class ParameterCollection;

class ArgType{
public:
    std::string name() const { return _name;  }

    std::string description() const { return _description; }

    bool required() const { return _required; }

    std::string verb_short() const { return _verb_short; }

    std::string verb_long() const { return _verb_long; }

    virtual std::string default_value() const = 0;

    // T get_default() const { return _default; }

    // If try_parse is successful it advances the args iterator to the next argument
    // and set value to the parsed and validated value
    bool try_parse( std::vector<std::string>::const_iterator& args, std::string& value ) const;

    virtual bool validate( const std::string& value ) const = 0;

    ArgType( ParameterCollection& params,
            const std::string& name,
            const std::string& description,
            const std::string& verb_short,
            const std::string& verb_long,
            bool required
            );

private:
    ArgType() = delete;
    ArgType(const ArgType&) = delete;

    std::string         _name;
    std::string         _description;
    bool                _required;
    std::string         _verb_short;
    std::string         _verb_long;
};

class ArgType_int : public ArgType {
public:
    ArgType_int( ParameterCollection& params,
        const std::string& name,
        const std::string& description,
        const std::string& verb_short,
        const std::string& verb_long,
        bool required,
        int default_value
        );
    ArgType_int( ParameterCollection& params,
        const std::string& name,
        const std::string& description,
        const std::string& verb_short,
        const std::string& verb_long,
        bool required,
        int default_value,
        int minimum_value,
        int maximum_value
        );
    bool validate( const std::string& value ) const override ;

    std::string default_value() const override { return std::to_string(_default); }

private:
    int         _default;
    int         _minimum_value;
    int         _maximum_value;
    bool        _range_valid;
};




class ParameterCollection {
    friend class ArgType;
public:
    std::vector<const ArgType*> get_args() const;

    bool parse(const std::string& args);
    
private:
    void register_arg(const ArgType& arg);
    std::vector<const ArgType*>     _arg_list;
};
