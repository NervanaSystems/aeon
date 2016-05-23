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

    // T get_default() const { return _default; }

    bool try_parse( std::vector<std::string>::const_iterator args ) {
        return false;
    }

    virtual bool validate( const std::string& value ) const = 0;

    ArgType( ParameterCollection& params,
             const std::string& name,
             const std::string& description,
             bool required,
             const std::string& verb_short,
             const std::string& verb_long );

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
         bool required,
         int default_value,
         const std::string& verb_short,
         const std::string& verb_long );

    bool validate( const std::string& value ) const override {
        return false;
    }

private:
    int         _default;
};




class ParameterCollection {
public:
    void register_arg(const ArgType& arg);
    
private:
    std::vector<ArgType>        _arg_list;
};
