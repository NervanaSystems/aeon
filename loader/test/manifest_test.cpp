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
#include "manifest.hpp"
#include "manifest_maker.hpp"

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

TEST(manifest, constructor) {
    Manifest manifest("manfiest.txt", false);
}

TEST(manifest, hash) {
    // TODO not 42
    Manifest manifest("manifest.txt", false);
    ASSERT_EQ(manifest.hash(), "42");
}

TEST(manifest, parse_file_doesnt_exist) {
    Manifest manifest("manifest.txt", false);

    ASSERT_EQ(manifest.getSize(), 0);
}

TEST(manifest, parse_file) {
    string tmpname = tmp_manifest_file(2, 0, 0);

    Manifest manifest(tmpname, false);
    ASSERT_EQ(manifest.getSize(), 2);
}

TEST(manifest, no_shuffle) {
    string filename = tmp_manifest_file(20, 4, 4);
    Manifest manifest1(filename, false);
    Manifest manifest2(filename, false);

    for(auto it1 = manifest1.begin(), it2 = manifest2.begin(); it1 != manifest1.end(); ++it1, ++it2) {
        ASSERT_EQ(it1->first, it2->first);
        ASSERT_EQ(it1->second, it2->second);
    }
}

TEST(manifest, shuffle) {
    string filename = tmp_manifest_file(20, 4, 4);
    Manifest manifest1(filename, false);
    Manifest manifest2(filename, true);

    bool different = false;

    for(auto it1 = manifest1.begin(), it2 = manifest2.begin(); it1 != manifest1.end(); ++it1, ++it2) {
        if(it1->first != it2->first) {
            different = true;
        }
    }
    ASSERT_EQ(different, true);
}
