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

#include "manifest_maker.hpp"

#include <string.h>
#include <fstream>
#include <string>

using namespace std;

string tmp_filename() {
    char *tmpname = strdup("/tmp/tmpfileXXXXXX");
    mkstemp(tmpname);
    return tmpname;
}

string tmp_zero_file(uint size) {
    string tmpname = tmp_filename();
    ofstream f(tmpname);

    for(uint i = 0; i < size; ++i) {
        f << (char)0 << endl;
    }

    f.close();

    return tmpname;
}

string tmp_manifest_file(uint num_records, uint object_size, uint target_size) {
    string tmpname = tmp_filename();
    ofstream f(tmpname);

    for(uint i = 0; i < num_records; ++i) {
        f << tmp_zero_file(object_size) << ",";
        f << tmp_zero_file(target_size) << endl;
    }

    f.close();

    return tmpname;
}
