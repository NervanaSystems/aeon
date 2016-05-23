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

ArgType::ArgType( ParameterCollection& params,
         const std::string& name,
         const std::string& description,
         const std::string& verb_short,
         const std::string& verb_long,
         bool required
          ) :
    _name{name},
    _description{description},
    _required{required},
    _verb_short{verb_short},
    _verb_long{verb_long}
{
    params.register_arg(*this);
}

bool ArgType::try_parse( vector<string>::const_iterator& args, string& value ) const {
    bool rc = false;
    if( (*args == "-"+_verb_short) || (*args == "--"+_verb_long) ) {
        args++; // skip verb
        value = *args++; // skip value
        rc = true;
    }
    return rc;
}

ArgType_int::ArgType_int( ParameterCollection& params,
        const std::string& name,
        const std::string& description,
        const std::string& verb_short,
        const std::string& verb_long,
        bool required,
        int default_value
         ) :
    ArgType(params, name, description, verb_short, verb_long, required),
    _default{default_value},
    _range_valid{false}
{
}
  
ArgType_int::ArgType_int( ParameterCollection& params,
        const std::string& name,
        const std::string& description,
        const std::string& verb_short,
        const std::string& verb_long,
        bool required,
        int default_value,
        int minimum_value,
        int maximum_value
         ) :
    ArgType(params, name, description, verb_short, verb_long, required),
    _default{default_value},
    _minimum_value{minimum_value},
    _maximum_value{maximum_value},
    _range_valid{true}
{
} 

bool ArgType_int::validate( const string& value ) const {
    bool rc = false;
    size_t end;
    int n = stoi(value, &end);
    if(end == value.size()) {
        if( _range_valid == false ) {
            rc = true;
        }
        else if( (n >= _minimum_value) && (n < _maximum_value) ) {
            rc = true;
        }
    }
    return rc;
} 

void ParameterCollection::register_arg(const ArgType& arg) {
    // check is this clashes with anything else in the list
    _arg_list.push_back(&arg);
}

std::vector<const ArgType*> ParameterCollection::get_args() const {
    return _arg_list;
}

bool ParameterCollection::parse(const std::string& args) {
    stringstream ss(args);
    vector<string> argList;
    string arg;
    while( ss >> arg ) { argList.push_back(arg); }

    auto it = argList.cbegin();
    while(it != argList.end()) {
        for( const ArgType* a : _arg_list ) {
            string value;
            if( a->try_parse( it, value ) ) {
                cout << "parse successful " << a->name() << "=" << value << endl;
                if( a->validate( value ) ) {
                    cout << "validation succesful" << endl;
                }
                break;
            }
        }
    }

    return true;
}


