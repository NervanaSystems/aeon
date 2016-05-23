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
#include "argtype.hpp"


using namespace std;

ArgType::ArgType( ParameterCollection& params,
         const std::string& name,
         const std::string& description,
         bool required,
         const std::string& verb_short,
         const std::string& verb_long ) :
    _name{name},
    _description{description},
    _required{required},
    _verb_short{verb_short},
    _verb_long{verb_long}
{
    params.register_arg(*this);
}

ArgType_int::ArgType_int( ParameterCollection& params,
        const std::string& name,
        const std::string& description,
        bool required,
        int default_value,
        const std::string& verb_short,
        const std::string& verb_long ) :
    ArgType(params, name, description, required, verb_short, verb_long),
    _default{default_value}
{
}
    

void ParameterCollection::register_arg(const ArgType& arg) {
    cout << "register " << arg.name() << endl;
    // _arg_list.push_back(arg);
}
