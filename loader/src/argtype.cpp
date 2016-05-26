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

#include <iostream>
#include <sstream>
#include "argtype.hpp"

using namespace std;

namespace nervana {
    template<>
    bool ArgType<int>::validate( const std::string& value ) const {
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

    template<>
    bool ArgType<float>::validate( const std::string& value ) const {
        bool rc = false;
        size_t end;
        try {
            int n = std::stof(value, &end);
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

    template<>
    bool ArgType<bool>::validate( const std::string& value ) const {
        return value == "true" || value == "false";
    }

    template<>
    bool ArgType<std::string>::validate( const std::string& value ) const {
        return true;
    }

    template<>
    std::string ArgType<std::string>::default_value() const {
        return _default;
    }
}

map<string,shared_ptr<nervana::interface_ArgType> > nervana::ParameterCollection::get_args() const {
    map<string,shared_ptr<interface_ArgType> > rc;
    for( argtype_t arg : _arg_list ) {
        rc.insert({arg->name(),arg});
    }
    return rc;
}

bool nervana::ParameterCollection::parse(const std::string& args, parsed_args& parsedArgs) {
    stringstream ss(args);
    deque<string> argList;
    string arg;
    bool rc = true;
    while( ss >> arg ) { argList.push_back(arg); }

    while(argList.size()>0) {
        bool parsed = false;
        string tmpArg = argList.front();
        for( argtype_t a : _arg_list ) {
            string value;
            if( a->try_parse( argList ) ) {
                argList.pop_front(); // skip verb
                if(argList.size()>0) {
                    value = argList.front();
                    argList.pop_front(); // skip value
                    if(parsedArgs.add_value(a, value)) {
                        parsed = true;
                    }
                } else {
                    cout << "missing value for " << a->verb_short() << "|--" << a->verb_long() << endl;
                    rc = false;
                }
                break;
            }
        }
        if(rc == false) break;
        if(!parsed) {
            cout << "failed to parse arg " << tmpArg << endl;
            rc = false;
            break;
        }
    }

    // fill in any missing non-required args
    for( argtype_t a : _arg_list ) {
        if(a->required() == false && parsedArgs.contains(a->name()) == false) {
            parsedArgs.add_value(a, a->default_value());
        }
    }

    // Check for required arguments
    if(rc == true) {
        for( argtype_t a : _arg_list ) {
            if(a->required() && !parsedArgs.contains(a->name())) {
                rc = false;
                cout << "required argument -" << a->verb_short() << "|--" << a->verb_long() << " missing" << endl;
            }
        }
    }

    // if(rc == false) {
    //     parsedArgs.clear();
    // }

    return rc;
}

bool nervana::parsed_args::add_value( const argtype_t& arg, const std::string& value ) {
    bool rc = false;
    if(value_map.find(arg->name())!=value_map.end()) {
        cout << "argument -" << arg->verb_short() << "|--" << arg->verb_long() << " included more than once" << endl;
    } else if( arg->validate( value ) ) {
        rc = true;
        value_map.insert({arg->name(), value});
    }
    return rc;
}

bool nervana::parsed_args::contains( const std::string& name ) const {
    return value_map.find(name)!=value_map.end();
}

namespace nervana {
    template<> int parsed_args::value<int>( const std::string& name ) const{
        string value = value_map.at(name);
        return stoi(value);
    }
    template<> float parsed_args::value<float>( const std::string& name ) const{
        string value = value_map.at(name);
        return stof(value);
    }
    template<> bool parsed_args::value<bool>( const std::string& name ) const{
        string value = value_map.at(name);
        return name == "true";
    }
    template<> string parsed_args::value<string>( const std::string& name ) const{
        string value = value_map.at(name);
        return value;
    }
}
