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
#include "manifest_csv.hpp"
#include "csv_manifest_maker.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

#include "util.hpp"

using namespace std;
using namespace nervana;

TEST(manifest, constructor)
{
    manifest_maker mm;
    string tmpname = mm.tmp_manifest_file(0, {0, 0});
    nervana::manifest_csv manifest0(tmpname, false);
}

TEST(manifest, no_file)
{
    ASSERT_THROW(nervana::manifest_csv manifest0(
        "/tmp/jsdkfjsjkfdjaskdfj_doesnt_exist", false
    ), std::runtime_error);
}

TEST(manifest, id_eq)
{
    manifest_maker mm;
    string tmpname = mm.tmp_manifest_file(0, {0, 0});
    nervana::manifest_csv manifest1(tmpname, false);
    nervana::manifest_csv manifest2(tmpname, false);
    ASSERT_EQ(manifest1.cache_id(), manifest2.cache_id());
}

TEST(manifest, id_ne)
{
    manifest_maker mm;
    nervana::manifest_csv manifest1(mm.tmp_manifest_file(0, {0, 0}), false);
    nervana::manifest_csv manifest2(mm.tmp_manifest_file(0, {0, 0}), false);
    ASSERT_NE(manifest1.cache_id(), manifest2.cache_id());
}

TEST(manifest, version_eq)
{
    manifest_maker mm;
    string tmpname = mm.tmp_manifest_file(0, {0, 0});
    nervana::manifest_csv manifest1(tmpname, false);
    nervana::manifest_csv manifest2(tmpname, false);
    ASSERT_EQ(manifest1.version(), manifest2.version());
}

void touch(const std::string& filename)
{
    // inspired by http://chris-sharpe.blogspot.com/2013/05/better-than-systemtouch.html
    int fd = open(
         filename.c_str(), O_WRONLY|O_CREAT|O_NOCTTY|O_NONBLOCK, 0666
    );
    assert(fd>=0);
    close(fd);

    // update timestamp for filename
    int rc = utimes(filename.c_str(), nullptr);
    assert(!rc);
}

TEST(manifest, version_ne)
{
    manifest_maker mm;
    string tmpname = mm.tmp_manifest_file(0, {0, 0});
    nervana::manifest_csv manifest0(tmpname, false);
    string v1 = manifest0.version();

    sleep(1);
    touch(tmpname);

    string v2 = manifest0.version();

    ASSERT_NE(v1, v2);
}

TEST(manifest, parse_file_doesnt_exist)
{
    manifest_maker mm;
    string tmpname = mm.tmp_manifest_file(0, {0, 0});
    nervana::manifest_csv manifest0(tmpname, false);

    ASSERT_EQ(manifest0.objectCount(), 0);
}

TEST(manifest, parse_file)
{
    manifest_maker mm;
    string tmpname = mm.tmp_manifest_file(2, {0, 0});

    nervana::manifest_csv manifest0(tmpname, false);
    ASSERT_EQ(manifest0.objectCount(), 2);
}

TEST(manifest, no_shuffle)
{
    manifest_maker mm;
    string filename = mm.tmp_manifest_file(20, {4, 4});
    nervana::manifest_csv manifest1(filename, false);
    nervana::manifest_csv manifest2(filename, false);

    for(auto it1 = manifest1.begin(), it2 = manifest2.begin(); it1 != manifest1.end(); ++it1, ++it2) {
        ASSERT_EQ((*it1)[0], (*it2)[0]);
        ASSERT_EQ((*it1)[1], (*it2)[1]);
    }
}

TEST(manifest, shuffle)
{
    manifest_maker mm;
    string filename = mm.tmp_manifest_file(20, {4, 4});
    nervana::manifest_csv manifest1(filename, false);
    nervana::manifest_csv manifest2(filename, true);

    bool different = false;

    for(auto it1 = manifest1.begin(), it2 = manifest2.begin(); it1 != manifest1.end(); ++it1, ++it2) {
        if((*it1)[0] != (*it2)[0]) {
            different = true;
        }
    }
    ASSERT_EQ(different, true);
}

TEST(manifest, non_paired_manifests)
{
    {
        manifest_maker mm;
        string filename = mm.tmp_manifest_file(20, {4, 4, 4});
        nervana::manifest_csv manifest1(filename, false);
        ASSERT_EQ(manifest1.objectCount(), 20);
    }
    {
        manifest_maker mm;
        string filename = mm.tmp_manifest_file(20, {4});
        nervana::manifest_csv manifest1(filename, false);
        ASSERT_EQ(manifest1.objectCount(), 20);
    }
}

TEST(manifest, uneven_records)
{
    manifest_maker mm;
    string filename = mm.tmp_manifest_file_with_ragged_fields();
    try {
        nervana::manifest_csv manifest1(filename, false);
        FAIL();
    } catch (std::exception& e) {
        ASSERT_EQ(
            string("at line: 1, manifest file has a line with differing"),
            string(e.what()).substr(0, 51)
        );
    }
}

TEST(manifest, root_path)
{
    string manifest_file = "tmp_manifest.csv";
    {
        ofstream f(manifest_file);
        for(int i=0; i<10; i++) {
            f << "/t1/image" << i << ".png,/t1/target" << i << ".txt\n";
        }
        f.close();
        nervana::manifest_csv manifest(manifest_file, false);
        int i = 0;
        for(const vector<string>& x : manifest) {
            ASSERT_EQ(2, x.size());
            stringstream ss;
            ss << "/t1/image" << i << ".png";
            EXPECT_STREQ(x[0].c_str(), ss.str().c_str());
            ss.str("");
            ss << "/t1/target" << i << ".txt";
            EXPECT_STREQ(x[1].c_str(), ss.str().c_str());
            i++;
        }
    }
    {
        ofstream f(manifest_file);
        for(int i=0; i<10; i++) {
            f << "/t1/image" << i << ".png,/t1/target" << i << ".txt\n";
        }
        f.close();
        nervana::manifest_csv manifest(manifest_file, false, "/x1");
        int i = 0;
        for(const vector<string>& x : manifest) {
            ASSERT_EQ(2, x.size());
            stringstream ss;
            ss << "/t1/image" << i << ".png";
            EXPECT_STREQ(x[0].c_str(), ss.str().c_str());
            ss.str("");
            ss << "/t1/target" << i << ".txt";
            EXPECT_STREQ(x[1].c_str(), ss.str().c_str());
            i++;
        }
    }
    {
        ofstream f(manifest_file);
        for(int i=0; i<10; i++) {
            f << "t1/image" << i << ".png,t1/target" << i << ".txt\n";
        }
        f.close();
        nervana::manifest_csv manifest(manifest_file, false, "/x1");
        int i = 0;
        for(const vector<string>& x : manifest) {
            ASSERT_EQ(2, x.size());
            stringstream ss;
            ss << "/x1/t1/image" << i << ".png";
            EXPECT_STREQ(x[0].c_str(), ss.str().c_str());
            ss.str("");
            ss << "/x1/t1/target" << i << ".txt";
            EXPECT_STREQ(x[1].c_str(), ss.str().c_str());
            i++;
        }
    }
    remove(manifest_file.c_str());
}
