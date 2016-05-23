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
#include <map>
#include "argtype.hpp"

using namespace std;

vector<shared_ptr<interface_ArgType> > ParameterCollection::get_args() const {
    return _arg_list;
}

bool ParameterCollection::parse(const std::string& args) {
    stringstream ss(args);
    vector<string> argList;
    string arg;
    while( ss >> arg ) { argList.push_back(arg); }

    map<shared_ptr<interface_ArgType>,string> parsedArgs;
    auto it = argList.cbegin();
    while(it != argList.end()) {
        bool parsed = false;
        for( shared_ptr<interface_ArgType> a : _arg_list ) {
            string value;
            if( a->try_parse( it, value ) ) {
                if( a->validate( value ) ) {
                    parsed = true;
                }
                break;
            }
        }
        if(!parsed) {
            cout << "failed to parse arg " << *it << endl;
            break;
        }
    }

    return true;
}


