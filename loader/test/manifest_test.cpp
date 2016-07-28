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
#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>

using namespace std;

TEST(manifest, constructor) {
    string tmpname = tmp_manifest_file(0, {0, 0});
    Manifest manifest(tmpname, false);
}

TEST(manifest, no_file) {
    ASSERT_THROW(Manifest manifest(
        "/tmp/jsdkfjsjkfdjaskdfj_doesnt_exist", false
    ), std::runtime_error);
}

TEST(manifest, hash_eq) {
    string tmpname = tmp_manifest_file(0, {0, 0});
    Manifest manifest1(tmpname, false);
    Manifest manifest2(tmpname, false);
    ASSERT_EQ(manifest1.hash(), manifest2.hash());
}

TEST(manifest, hash_ne) {
    Manifest manifest1(tmp_manifest_file(0, {0, 0}), false);
    Manifest manifest2(tmp_manifest_file(0, {0, 0}), false);
    ASSERT_NE(manifest1.hash(), manifest2.hash());
}

TEST(manifest, version_eq) {
    string tmpname = tmp_manifest_file(0, {0, 0});
    Manifest manifest1(tmpname, false);
    Manifest manifest2(tmpname, false);
    ASSERT_EQ(manifest1.version(), manifest2.version());
}

void touch(const std::string& filename)
{
    // inspired by http://chris-sharpe.blogspot.com/2013/05/better-than-systemtouch.html
    int fd = open(
         filename.c_str(), O_WRONLY|O_CREAT|O_NOCTTY|O_NONBLOCK, 0666
    );
    assert(fd>=0);

    // update timestamp for filename
    int rc = utimensat(
        AT_FDCWD, filename.c_str(), nullptr, 0
    );
    assert(!rc);
}

TEST(manifest, version_ne) {
    string tmpname = tmp_manifest_file(0, {0, 0});
    Manifest manifest(tmpname, false);
    string v1 = manifest.version();

    sleep(1);
    touch(tmpname);

    string v2 = manifest.version();

    ASSERT_NE(v1, v2);
}

TEST(manifest, parse_file_doesnt_exist) {
    string tmpname = tmp_manifest_file(0, {0, 0});
    Manifest manifest(tmpname, false);

    ASSERT_EQ(manifest.objectCount(), 0);
}

TEST(manifest, parse_file) {
    string tmpname = tmp_manifest_file(2, {0, 0});

    Manifest manifest(tmpname, false);
    ASSERT_EQ(manifest.objectCount(), 2);
}

TEST(manifest, no_shuffle) {
    string filename = tmp_manifest_file(20, {4, 4});
    Manifest manifest1(filename, false);
    Manifest manifest2(filename, false);

    for(auto it1 = manifest1.begin(), it2 = manifest2.begin(); it1 != manifest1.end(); ++it1, ++it2) {
        ASSERT_EQ((*it1)[0], (*it2)[0]);
        ASSERT_EQ((*it1)[1], (*it2)[1]);
    }
}

TEST(manifest, shuffle) {
    string filename = tmp_manifest_file(20, {4, 4});
    Manifest manifest1(filename, false);
    Manifest manifest2(filename, true);

    bool different = false;

    for(auto it1 = manifest1.begin(), it2 = manifest2.begin(); it1 != manifest1.end(); ++it1, ++it2) {
        if((*it1)[0] != (*it2)[0]) {
            different = true;
        }
    }
    ASSERT_EQ(different, true);
}

TEST(manifest, too_many_files) {
    string filename = tmp_manifest_file(20, {4, 4, 4});
    try {
        Manifest manifest1(filename, false);
        FAIL();
    } catch (std::exception& e) {
        ASSERT_EQ(
            string("manifest file has a line with more than 2 files "),
            string(e.what()).substr(0, 48)
        );
    }
}
