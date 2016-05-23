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

template<typename T>
class ArgType{
    friend class unit_test;
public:
    std::string name() const { return _name;  }

    std::string description() const { return _description; }

    bool required() const { return _required; }

    std::string verb_short() const { return _verb_short; }

    std::string verb_long() const { return _verb_long; }

    T get_default() const { return _default; }

    bool try_parse( std::vector<std::string>::const_iterator args ) {
        return false;
    }

    bool validate( const T& value ) {
        return false;
    }

    ArgType( const std::string& name,
             const std::string& description,
             bool required,
             const T& default_value,
             const std::string& verb_short,
             const std::string& verb_long );
    ArgType() = delete;
    ArgType(const ArgType&) = delete;

    ArgType& name(const std::string& value) { _name = value; return *this; }
    ArgType& description(const std::string& value) { _description = value; return *this; }
    ArgType& required(bool value) { _required = value; return *this; }
    ArgType& default_value(const T& value) { _default = value; return *this; }
    ArgType& verb_short(const std::string& value) { _verb_short = value; return *this; }
    ArgType& verb_long(const std::string& value) { _verb_long = value; return *this; }

    std::string         _name;
    std::string         _description;
    bool                _required;
    const T             _default;
    std::string         _verb_short;
    std::string         _verb_long;
};