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

vector<shared_ptr<interface_ArgType> > ParameterCollection::get_args() const {
    return _arg_list;
}

bool ParameterCollection::parse(const std::string& args, map<argtype_t,string>& parsedArgs) {
    stringstream ss(args);
    deque<string> argList;
    string arg;
    bool rc = true;
    while( ss >> arg ) { argList.push_back(arg); }

    while(argList.size()>0) {
        bool parsed = false;
        for( argtype_t a : _arg_list ) {
            string value;
            if( a->try_parse( argList, value ) ) {
                if(parsedArgs.find(a)!=parsedArgs.end()) {
                    cout << "argument -" << a->verb_short() << "|--" << a->verb_long() << " included more than once" << endl;
                    rc = false;
                    break;
                }
                if( a->validate( value ) ) {
                    parsed = true;
                    parsedArgs.insert({a, value});
                }
                break;
            }
        }
        if(rc == false) break;
        if(!parsed) {
            cout << "failed to parse arg " << argList.front() << endl;
            rc = false;
            break;
        }
    }

    // Check for required arguments
    if(rc == true) {
        for( argtype_t a : _arg_list ) {
            if(a->required() && parsedArgs.find(a)==parsedArgs.end()) {
                rc = false;
                cout << "required argument -" << a->verb_short() << "|--" << a->verb_long() << " missing" << endl;
            }
        }
    }

    if(rc == false) {
        parsedArgs.clear();
    }

    return rc;
}


