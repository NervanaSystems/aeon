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

#include "gtest/gtest.h"
#include "argtype.hpp"

class unit_test{};    // grant access to ArgType privates

    // ArgType& name(const std::string& value) { _name = value; return *this; }
    // ArgType& description(const std::string& value) { _description = value; return *this; }
    // ArgType& required(bool value) { _required = value; return *this; }
    // ArgType& default(const T& value) { _default = value; return *this; }
    // ArgType& verb_short(const std::string& value) { _verb_short = value; return *this; }
    // ArgType& verb_long(const std::string& value) { _verb_long = value; return *this; }

class ParamList1 : public unit_test {
public:
    static const ArgType<int> arg1("arg1","description of arg1",false,3,"a1","arg-1");
 };

 TEST(loader,argtype) {
 }
