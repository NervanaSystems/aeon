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
    Manifest manifest("manfiest.txt");
}

TEST(manifest, hash) {
    // TODO not 42
    Manifest manifest("manifest.txt");
    ASSERT_EQ(manifest.hash(), "42");
}

TEST(manifest, parse_file_doesnt_exist) {
    Manifest manifest("manifest.txt");

    ASSERT_EQ(manifest.getSize(), 0);
}

TEST(manifest, parse_file) {
    string tmpname = tmp_manifest_file(2, 0, 0);

    Manifest manifest(tmpname);
    ASSERT_EQ(manifest.getSize(), 2);
} 
