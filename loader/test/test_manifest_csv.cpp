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

#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <memory>

#include <chrono>

#include "gtest/gtest.h"
#include "manifest_csv.hpp"
#include "csv_manifest_maker.hpp"
#include "util.hpp"
#include "file_util.hpp"
#include "manifest_csv.hpp"
#include "crc.hpp"

using namespace std;
using namespace nervana;

TEST(manifest, constructor)
{
    manifest_maker        mm;
    string                tmpname = mm.tmp_manifest_file(0, {0, 0});
    nervana::manifest_csv manifest0(tmpname, false);
}

TEST(manifest, no_file)
{
    ASSERT_THROW(nervana::manifest_csv manifest0("/tmp/jsdkfjsjkfdjaskdfj_doesnt_exist", false), std::runtime_error);
}

TEST(manifest, id_eq)
{
    manifest_maker        mm;
    string                tmpname = mm.tmp_manifest_file(0, {0, 0});
    nervana::manifest_csv manifest1(tmpname, false);
    nervana::manifest_csv manifest2(tmpname, false);
    ASSERT_EQ(manifest1.cache_id(), manifest2.cache_id());
}

TEST(manifest, id_ne)
{
    manifest_maker        mm;
    nervana::manifest_csv manifest1(mm.tmp_manifest_file(0, {0, 0}), false);
    nervana::manifest_csv manifest2(mm.tmp_manifest_file(0, {0, 0}), false);
    ASSERT_NE(manifest1.cache_id(), manifest2.cache_id());
}

TEST(manifest, version_eq)
{
    manifest_maker        mm;
    string                tmpname = mm.tmp_manifest_file(0, {0, 0});
    nervana::manifest_csv manifest1(tmpname, false);
    nervana::manifest_csv manifest2(tmpname, false);
    ASSERT_EQ(manifest1.version(), manifest2.version());
}

TEST(manifest, parse_file_doesnt_exist)
{
    manifest_maker        mm;
    string                tmpname = mm.tmp_manifest_file(0, {0, 0});
    nervana::manifest_csv manifest0(tmpname, false);

    ASSERT_EQ(manifest0.object_count(), 0);
}

TEST(manifest, parse_file)
{
    manifest_maker mm;
    string         tmpname = mm.tmp_manifest_file(2, {0, 0});

    nervana::manifest_csv manifest0(tmpname, false);
    ASSERT_EQ(manifest0.object_count(), 2);
}

TEST(manifest, no_shuffle)
{
    manifest_maker        mm;
    string                filename = mm.tmp_manifest_file(20, {4, 4});
    nervana::manifest_csv manifest1(filename, false);
    nervana::manifest_csv manifest2(filename, false);

    for (auto it1 = manifest1.begin(), it2 = manifest2.begin(); it1 != manifest1.end(); ++it1, ++it2)
    {
        ASSERT_EQ((*it1)[0], (*it2)[0]);
        ASSERT_EQ((*it1)[1], (*it2)[1]);
    }
}

TEST(manifest, shuffle)
{
    manifest_maker        mm;
    string                filename = mm.tmp_manifest_file(20, {4, 4});
    nervana::manifest_csv manifest1(filename, false);
    nervana::manifest_csv manifest2(filename, true);

    bool different = false;

    for (auto it1 = manifest1.begin(), it2 = manifest2.begin(); it1 != manifest1.end(); ++it1, ++it2)
    {
        if ((*it1)[0] != (*it2)[0])
        {
            different = true;
        }
    }
    ASSERT_EQ(different, true);
}

TEST(manifest, non_paired_manifests)
{
    {
        manifest_maker        mm;
        string                filename = mm.tmp_manifest_file(20, {4, 4, 4});
        nervana::manifest_csv manifest1(filename, false);
        ASSERT_EQ(manifest1.object_count(), 20);
    }
    {
        manifest_maker        mm;
        string                filename = mm.tmp_manifest_file(20, {4});
        nervana::manifest_csv manifest1(filename, false);
        ASSERT_EQ(manifest1.object_count(), 20);
    }
}

TEST(manifest, uneven_records)
{
    manifest_maker mm;
    string         filename = mm.tmp_manifest_file_with_ragged_fields();
    try
    {
        nervana::manifest_csv manifest1(filename, false);
        FAIL();
    }
    catch (std::exception& e)
    {
        ASSERT_EQ(string("at line: 1, manifest file has a line with differing"), string(e.what()).substr(0, 51));
    }
}

TEST(manifest, root_path)
{
    string manifest_file = "tmp_manifest.csv";
    {
        ofstream f(manifest_file);
        for (int i = 0; i < 10; i++)
        {
            f << "/t1/image" << i << ".png,/t1/target" << i << ".txt\n";
        }
        f.close();
        nervana::manifest_csv manifest(manifest_file, false);
        int                   i = 0;
        for (const vector<string>& x : manifest)
        {
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
        for (int i = 0; i < 10; i++)
        {
            f << "/t1/image" << i << ".png,/t1/target" << i << ".txt\n";
        }
        f.close();
        nervana::manifest_csv manifest(manifest_file, false, "/x1");
        int                   i = 0;
        for (const vector<string>& x : manifest)
        {
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
        for (int i = 0; i < 10; i++)
        {
            f << "t1/image" << i << ".png,t1/target" << i << ".txt\n";
        }
        f.close();
        nervana::manifest_csv manifest(manifest_file, false, "/x1");
        int                   i = 0;
        for (const vector<string>& x : manifest)
        {
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

TEST(manifest, crc)
{
    const string input    = "123456789";
    uint32_t     expected = 0xe3069283;
    uint32_t     actual   = 0;

    CryptoPP::CRC32C crc;
    crc.Update((const uint8_t*)input.data(), input.size());
    crc.TruncatedFinal((uint8_t*)&actual, sizeof(actual));

    //    cout << "expected 0x" << setfill('0') << setw(2) << hex << expected << dec << endl;
    //    cout << "actual   0x" << setfill('0') << setw(2) << hex << actual << dec << endl;

    EXPECT_EQ(expected, actual);
}

// TEST(manifest, performance)
//{
//    string manifest_filename = file_util::tmp_filename();
//    string cache_root = "/this/is/supposed/to/be/long/so/we/make/it/so/";
//    cout << "tmp manifest file " << manifest_filename << endl;

//    chrono::high_resolution_clock timer;

//    // Generate a manifest file
//    {
//        auto startTime = timer.now();
//        ofstream mfile(manifest_filename);
//        for(int i=0; i<10e6; i++)
//        {
//            mfile << cache_root << "image_" << i << ".jpg,";
//            mfile << cache_root << "target_" << i << ".txt\n";
//        }
//        auto endTime = timer.now();
//        cout << "create manifest " << (chrono::duration_cast<chrono::milliseconds>(endTime - startTime)).count()  << " ms" <<
//        endl;
//    }

//    // Parse the manifest file
//    shared_ptr<manifest_csv> manifest;
//    {
//        auto startTime = timer.now();
//        manifest = make_shared<manifest_csv>(manifest_filename, false);
//        auto endTime = timer.now();
//        cout << "load manifest " << (chrono::duration_cast<chrono::milliseconds>(endTime - startTime)).count()  << " ms" << endl;
//    }

//    // compute the CRC
//    {
//        auto startTime = timer.now();
//        uint32_t crc = manifest->get_crc();
//        auto endTime = timer.now();
//        cout << "manifest crc 0x" << setfill('0') << setw(8) << hex << crc << dec << endl;
//        cout << "crc time " << (chrono::duration_cast<chrono::milliseconds>(endTime - startTime)).count()  << " ms" << endl;
//    }

//    remove(manifest_filename.c_str());
//}
