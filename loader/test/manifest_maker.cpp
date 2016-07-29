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
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <cstdio>

using namespace std;

vector<string> tmp_filenames;

void remove_files() {
    for(auto it = tmp_filenames.begin(); it != tmp_filenames.end(); ++it) {
        remove(it->c_str());
    }
}

void register_atexit() {
    // register an atexit to clean up files, and only ever do it once
    static bool registered = false;
    if(!registered) {
        atexit(remove_files);
        registered = true;
    }
}

string tmp_filename() {
    char *tmpname = strdup("/tmp/tmpfileXXXXXX");
    mkstemp(tmpname);
    tmp_filenames.push_back(tmpname);
    register_atexit();
    return tmpname;
}

string tmp_file_repeating(uint size, uint x) {
    // create a temp file of `size` bytes filled with uint x
    string tmpname = tmp_filename();
    ofstream f(tmpname, ios::binary);

    uint repeats = size / sizeof(x);
    for(uint i = 0; i < repeats; ++i) {
        f.write(reinterpret_cast <const char*>(&x), sizeof(x));
    }

    f.close();

    return tmpname;
}

string tmp_manifest_file(uint num_records, vector<uint> sizes) {
    string tmpname = tmp_filename();
    ofstream f(tmpname);

    for(uint i = 0; i < num_records; ++i) {
        // stick a unique uint into each file
        for(uint j = 0; j < sizes.size(); ++j) {
            if(j != 0) {
                f << ",";
            }

            f << tmp_file_repeating(sizes[j], (i * sizes.size()) + j);
        }
        f << endl;
    }

    f.close();

    return tmpname;
}

std::string tmp_manifest_file_with_invalid_filename() {
    string tmpname = tmp_filename();
    ofstream f(tmpname);

    for(uint i = 0; i < 10; ++i) {
        f << tmp_filename() + ".this_file_shouldnt_exist" << ',';
        f << tmp_filename() + ".this_file_shouldnt_exist" << endl;
    }

    f.close();
    return tmpname;
}

std::string tmp_manifest_file_with_ragged_fields() {
    string tmpname = tmp_filename();
    ofstream f(tmpname);

    for(uint i = 0; i < 10; ++i) {
        for(uint j = 0; j < i % 3 + 1; ++j) {
            f << (j != 0 ? "," : "") << tmp_filename();
        }
        f << endl;
    }

    f.close();
    return tmpname;
}
