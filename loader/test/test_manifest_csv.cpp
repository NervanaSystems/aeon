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
#include "file_util.hpp"
#include "block_loader_file_async.hpp"
#include "base64.hpp"

using namespace std;
using namespace nervana;

static string test_data_directory = file_util::path_join(string(CURDIR), "test_data");

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

    ASSERT_EQ(manifest0.record_count(), 0);
}

TEST(manifest, parse_file)
{
    manifest_maker mm;
    string         tmpname = mm.tmp_manifest_file(2, {0, 0});

    nervana::manifest_csv manifest0(tmpname, false);
    ASSERT_EQ(manifest0.record_count(), 2);
}

TEST(manifest, no_shuffle)
{
    manifest_maker        mm;
    string                filename = mm.tmp_manifest_file(20, {4, 4});
    nervana::manifest_csv manifest1(filename, false);
    nervana::manifest_csv manifest2(filename, false);

    ASSERT_EQ(manifest1.record_count(), manifest2.record_count());
    ASSERT_EQ(2, manifest1.element_count());
    for (int i=0; i<manifest1.record_count(); i++)
    {
        ASSERT_EQ(manifest1[i][0], manifest2[i][0]);
        ASSERT_EQ(manifest1[i][1], manifest2[i][1]);
    }
}

TEST(manifest, shuffle)
{
    manifest_maker        mm;
    string                filename = mm.tmp_manifest_file(20, {4, 4});
    nervana::manifest_csv manifest1(filename, false);
    nervana::manifest_csv manifest2(filename, true);

    bool different = false;

    ASSERT_EQ(manifest1.record_count(), manifest2.record_count());
    for (int i=0; i<manifest1.record_count(); i++)
    {
        if (manifest1[i][0] != manifest2[i][0])
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
        ASSERT_EQ(manifest1.record_count(), 20);
    }
    {
        manifest_maker        mm;
        string                filename = mm.tmp_manifest_file(20, {4});
        nervana::manifest_csv manifest1(filename, false);
        ASSERT_EQ(manifest1.record_count(), 20);
    }
}

TEST(manifest, uneven_records)
{
    manifest_maker mm;
    string         filename = mm.tmp_manifest_file_with_ragged_fields();
    EXPECT_THROW(nervana::manifest_csv manifest1(filename, false), runtime_error);
}

TEST(manifest, root_path)
{
    string manifest_file = "tmp_manifest.csv";
    {
        ofstream f(manifest_file);
        for (int i = 0; i < 10; i++)
        {
            f << "/t1/image" << i << ".png" << manifest_csv::get_delimiter();
            f << "/t1/target" << i << ".txt\n";
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
            f << "/t1/image" << i << ".png" << manifest_csv::get_delimiter();
            f << "/t1/target" << i << ".txt\n";
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
            f << "t1/image" << i << ".png" << manifest_csv::get_delimiter();
            f << "t1/target" << i << ".txt\n";
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

TEST(manifest, file_implicit)
{
    vector<string> image_files = {"flowers.jpg", "img_2112_70.jpg"};
    vector<string> target_files = {"1.txt", "2.txt"};

    stringstream ss;

    for (int count=0; count<32; count++)
    {
        for (int i=0; i<image_files.size(); i++)
        {
            ss << image_files[i] << "\t" << target_files[i] << "\n";
        }
    }

    manifest_csv manifest{ss, false, test_data_directory};
    size_t block_size = 16;

    block_loader_file_async bload{&manifest, block_size};

    for (int i=0; i<2; i++)
    {
        variable_buffer_array* buffer = bload.filler();
        ASSERT_NE(nullptr, buffer);
        ASSERT_EQ(2, buffer->size());
        buffer_variable_size_elements image_data = buffer->at(0);
        buffer_variable_size_elements target_data = buffer->at(1);
//        ASSERT_EQ(batch_size, image_data.get_item_count());
//        ASSERT_EQ(batch_size, target_data.get_item_count());
        for (int j=0; j<image_data.get_item_count(); j++)
        {
            auto idata = image_data.get_item(j);
            auto tdata = target_data.get_item(j);
            string target{tdata.data(), tdata.size()};
//            INFO << target;
//            int value = stod(target);
//            EXPECT_EQ(j%2+1, value);
        }
    }

}

TEST(manifest, file_explicit)
{
    vector<string> image_files = {"flowers.jpg", "img_2112_70.jpg"};
    vector<string> target_files = {"1.txt", "2.txt"};

    stringstream ss;
    ss << "@FILE" << "\t" << "FILE" << "\n";
    for (int count=0; count<32; count++)
    {
        for (int i=0; i<image_files.size(); i++)
        {
            ss << image_files[i] << "\t" << target_files[i] << "\n";
        }
    }

    manifest_csv manifest{ss, false, test_data_directory};
    size_t block_size = 16;

    auto types = manifest.get_element_types();
    ASSERT_EQ(2, types.size());
    EXPECT_EQ(manifest::element_t::FILE, types[0]);
    EXPECT_EQ(manifest::element_t::FILE, types[1]);

    block_loader_file_async bload{&manifest, block_size};
    for (int i=0; i<2; i++)
    {
        variable_buffer_array* buffer = bload.filler();
        ASSERT_NE(nullptr, buffer);
        ASSERT_EQ(2, buffer->size());
        buffer_variable_size_elements image_data = buffer->at(0);
        buffer_variable_size_elements target_data = buffer->at(1);
//        ASSERT_EQ(batch_size, image_data.get_item_count());
//        ASSERT_EQ(batch_size, target_data.get_item_count());
        for (int j=0; j<image_data.get_item_count(); j++)
        {
            auto idata = image_data.get_item(j);
            auto tdata = target_data.get_item(j);
            string target{tdata.data(), tdata.size()};
            int value = stod(target);
            EXPECT_EQ(j%2+1, value);
        }
    }
}

string make_target_data(size_t index)
{
    stringstream tmp;
    tmp << "target_number" << index++;
    return tmp.str();
}

TEST(manifest, binary)
{
    vector<string> image_files = {"flowers.jpg", "img_2112_70.jpg"};
    stringstream ss;
    size_t index = 0;
    ss << "@FILE" << "\t" << "BINARY" << "\n";
    for (int count=0; count<32; count++)
    {
        for (int i=0; i<image_files.size(); i++)
        {
            vector<char> str = string2vector(make_target_data(index++));
            auto data_string = base64::encode(str);
            ss << image_files[i] << "\t" << vector2string(data_string) << "\n";
        }
    }

    manifest_csv manifest{ss, false, test_data_directory};
    size_t block_size = 16;

//    for (auto data : manifest)
//    {
//        INFO << data[0] << ", " << data[1];
//    }

    auto types = manifest.get_element_types();
    ASSERT_EQ(2, types.size());
    EXPECT_EQ(manifest::element_t::FILE, types[0]);
    EXPECT_EQ(manifest::element_t::BINARY, types[1]);

    block_loader_file_async block_loader{&manifest, block_size};
    index = 0;
    for (int i=0; i<2; i++)
    {
        variable_buffer_array* buffer = block_loader.filler();
        ASSERT_NE(nullptr, buffer);
        ASSERT_EQ(2, buffer->size());
        buffer_variable_size_elements image_data = buffer->at(0);
        buffer_variable_size_elements target_data = buffer->at(1);
//        ASSERT_EQ(batch_size, image_data.get_item_count());
//        ASSERT_EQ(batch_size, target_data.get_item_count());
        for (int j=0; j<image_data.get_item_count(); j++)
        {
            auto idata = image_data.get_item(j);
            auto tdata = target_data.get_item(j);
            string str = vector2string(tdata);
            string expected = make_target_data(index);
            index = (index+1) % manifest.record_count();
            EXPECT_STREQ(expected.c_str(), str.c_str());
        }
    }
}

TEST(manifest, string)
{
    vector<string> image_files = {"flowers.jpg", "img_2112_70.jpg"};
    stringstream ss;
    size_t index = 0;
    ss << "@FILE" << "\t" << "STRING" << "\n";
    for (int count=0; count<32; count++)
    {
        for (int i=0; i<image_files.size(); i++)
        {
            string str = make_target_data(index++);
            ss << image_files[i] << "\t" << str << "\n";
        }
    }

    manifest_csv manifest{ss, false, test_data_directory};
    size_t block_size = 16;

//    for (auto data : manifest)
//    {
//        INFO << data[0] << ", " << data[1];
//    }

    auto types = manifest.get_element_types();
    ASSERT_EQ(2, types.size());
    EXPECT_EQ(manifest::element_t::FILE, types[0]);
    EXPECT_EQ(manifest::element_t::STRING, types[1]);

    block_loader_file_async block_loader{&manifest, block_size};
    index = 0;
    for (int i=0; i<2; i++)
    {
        variable_buffer_array* buffer = block_loader.filler();
        ASSERT_NE(nullptr, buffer);
        ASSERT_EQ(2, buffer->size());
        buffer_variable_size_elements image_data = buffer->at(0);
        buffer_variable_size_elements target_data = buffer->at(1);
//        ASSERT_EQ(batch_size, image_data.get_item_count());
//        ASSERT_EQ(batch_size, target_data.get_item_count());
        for (int j=0; j<image_data.get_item_count(); j++)
        {
            auto idata = image_data.get_item(j);
            auto tdata = target_data.get_item(j);
            string str = vector2string(tdata);
            string expected = make_target_data(index);
            index = (index+1) % manifest.record_count();
            EXPECT_STREQ(expected.c_str(), str.c_str());
        }
    }
}

TEST(manifest, ascii_int)
{
}

TEST(manifest, ascii_float)
{
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
