/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

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
#include "manifest_file.hpp"
#include "manifest_builder.hpp"
#include "util.hpp"
#include "file_util.hpp"
#include "manifest_file.hpp"
#include "crc.hpp"
#include "file_util.hpp"
#include "block_loader_file.hpp"
#include "base64.hpp"
#include "gen_image.hpp"
#include "loader.hpp"

using namespace std;
using namespace nervana;

static string test_data_directory = file_util::path_join(string(CURDIR), "test_data");

TEST(manifest, constructor)
{
    manifest_builder       mm;
    auto&                  ms = mm.sizes({0, 0}).record_count(0).create();
    nervana::manifest_file manifest0(ms, false);
}

TEST(manifest, no_file)
{
    ASSERT_THROW(nervana::manifest_file manifest0("/tmp/jsdkfjsjkfdjaskdfj_doesnt_exist", false),
                 std::runtime_error);
}

TEST(manifest, version_eq)
{
    manifest_builder       mm;
    auto&                  ms = mm.sizes({0, 0}).record_count(0).create();
    nervana::manifest_file manifest1(ms, false);
    nervana::manifest_file manifest2(ms, false);
    ASSERT_EQ(manifest1.version(), manifest2.version());
}

TEST(manifest, parse_file_doesnt_exist)
{
    manifest_builder       mm;
    auto&                  ms = mm.sizes({0, 0}).record_count(0).create();
    nervana::manifest_file manifest0(ms, false);

    ASSERT_EQ(manifest0.record_count(), 0);
}

TEST(manifest, parse_file)
{
    manifest_builder mm;
    auto&            ms = mm.sizes({0, 0}).record_count(2).create();

    nervana::manifest_file manifest0(ms, false);
    ASSERT_EQ(manifest0.record_count(), 2);
}

TEST(manifest, no_shuffle)
{
    manifest_builder       mm1;
    auto&                  ms1 = mm1.sizes({4, 4}).record_count(20).create();
    nervana::manifest_file manifest1(ms1, false);

    manifest_builder       mm2;
    auto&                  ms2 = mm2.sizes({4, 4}).record_count(20).create();
    nervana::manifest_file manifest2(ms2, false);

    ASSERT_EQ(1, manifest1.block_count());
    ASSERT_EQ(1, manifest2.block_count());

    auto& m1_block = *manifest1.next();
    auto& m2_block = *manifest2.next();

    ASSERT_EQ(manifest1.record_count(), manifest2.record_count());
    ASSERT_EQ(2, manifest1.elements_per_record());
    for (int i = 0; i < manifest1.record_count(); i++)
    {
        ASSERT_EQ(m1_block[i][0], m2_block[i][0]);
        ASSERT_EQ(m1_block[i][1], m2_block[i][1]);
    }
}

namespace {
void test_multinode_manifest(bool shuffle)
{
    const uint32_t record_count = 17;
    const uint32_t batch_size = 3;
    const uint32_t node_count = 2;

    manifest_builder       mm1;
    auto&                  ms1 = mm1.sizes({4, 4}).record_count(record_count).create();
    nervana::manifest_file manifest(ms1, shuffle, "", 1.0f, 5000, 1234, 0, 0, batch_size);

    manifest_builder       mmn1;
    auto&                  msn1 = mmn1.sizes({4, 4}).record_count(record_count).create();
    nervana::manifest_file manifest_node1(msn1, shuffle, "", 1.0f, 5000, 1234, 0, 2, batch_size);

    manifest_builder       mmn2;
    auto&                  msn2 = mmn2.sizes({4, 4}).record_count(record_count).create();
    nervana::manifest_file manifest_node2(msn2, shuffle, "", 1.0f, 5000, 1234, 1, 2, batch_size);


    ASSERT_EQ(1, manifest_node1.block_count());
    ASSERT_EQ(1, manifest_node2.block_count());

    ASSERT_EQ(8, manifest_node1.record_count());
    ASSERT_EQ(8, manifest_node2.record_count());

    ASSERT_EQ(2, manifest_node1.elements_per_record());
    ASSERT_EQ(2, manifest_node2.elements_per_record());

    auto record_count_node = (manifest.record_count() / node_count) * node_count;
    uint32_t batches = ((record_count_node / node_count) / batch_size) * node_count;

    for (int i = 0; i < batches; i++)
    {
        for (int j = 0; j < batch_size; j++)
        {
            uint32_t src_index = i * batch_size + j;
            uint32_t dst_index = ( i / node_count ) * batch_size + j;
            if ( i % node_count == 0)
            {
                ASSERT_EQ(manifest[src_index][0], manifest_node1[dst_index][0]);
                ASSERT_EQ(manifest[src_index][1], manifest_node1[dst_index][1]);
            }
            else
            {
                ASSERT_EQ(manifest[src_index][0], manifest_node2[dst_index][0]);
                ASSERT_EQ(manifest[src_index][1], manifest_node2[dst_index][1]);
            }
        }
    }

    auto remain = record_count_node - batches * batch_size;
    for (int i = 0; i < remain; i++)
    {
            uint32_t src_index = batches * batch_size + i;
            uint32_t dst_index = batches * batch_size / node_count + i % node_count;

            if ( i / node_count == 0)
            {
                ASSERT_EQ(manifest[src_index][0], manifest_node1[dst_index][0]);
                ASSERT_EQ(manifest[src_index][1], manifest_node1[dst_index][1]);
            }
            else
            {
                ASSERT_EQ(manifest[src_index][0], manifest_node2[dst_index][0]);
                ASSERT_EQ(manifest[src_index][1], manifest_node2[dst_index][1]);
            }
    }

    auto block       = manifest.next();
    auto block_node1 = manifest_node1.next();
    auto block_node2 = manifest_node2.next();

    manifest.reset();
    manifest_node1.reset();    
    manifest_node2.reset();

    block       = manifest.next();
    block_node1 = manifest_node1.next();
    block_node2 = manifest_node2.next();
}

}

TEST(manifest, no_shuffle_multinode)
{
    test_multinode_manifest(false);
}

TEST(manifest, shuffle_multinode)
{
    test_multinode_manifest(true);
}

TEST(manifest, shuffle)
{
    manifest_builder       mm1;
    auto&                  ms1 = mm1.sizes({4, 4}).record_count(20).create();
    nervana::manifest_file manifest1(ms1, false);

    manifest_builder       mm2;
    auto&                  ms2 = mm2.sizes({4, 4}).record_count(20).create();
    nervana::manifest_file manifest2(ms2, true);

    bool different = false;

    ASSERT_EQ(1, manifest1.block_count());
    ASSERT_EQ(1, manifest2.block_count());

    auto& m1_block = *manifest1.next();
    auto& m2_block = *manifest2.next();

    ASSERT_EQ(manifest1.record_count(), manifest2.record_count());
    for (int i = 0; i < manifest1.record_count(); i++)
    {
        if (m1_block[i][0] != m2_block[i][0])
        {
            different = true;
        }
    }
    ASSERT_EQ(different, true);
}

TEST(manifest, non_paired_manifests)
{
    {
        manifest_builder       mm;
        auto&                  ms = mm.sizes({4, 4, 4}).record_count(20).create();
        nervana::manifest_file manifest1(ms, false);
        ASSERT_EQ(manifest1.record_count(), 20);
    }
    {
        manifest_builder       mm;
        auto&                  ms = mm.sizes({4}).record_count(20).create();
        nervana::manifest_file manifest1(ms, false);
        ASSERT_EQ(manifest1.record_count(), 20);
    }
}

TEST(manifest, root_path)
{
    string manifest_file = "tmp_manifest.tsv";
    {
        ofstream f(manifest_file);
        f << "@FILE\tFILE\n";
        for (int i = 0; i < 10; i++)
        {
            f << "/t1/image" << i << ".png" << manifest_file::get_delimiter();
            f << "/t1/target" << i << ".txt\n";
        }
        f.close();
        nervana::manifest_file manifest(manifest_file, false);
        ASSERT_EQ(1, manifest.block_count());
        auto& block = *manifest.next();
        for (int i = 0; i < manifest.record_count(); i++)
        {
            const vector<string>& x = block[i];

            ASSERT_EQ(2, x.size());
            stringstream ss;
            ss << "/t1/image" << i << ".png";
            EXPECT_STREQ(x[0].c_str(), ss.str().c_str());
            ss.str("");
            ss << "/t1/target" << i << ".txt";
            EXPECT_STREQ(x[1].c_str(), ss.str().c_str());
        }
    }
    {
        ofstream f(manifest_file);
        f << "@FILE\tFILE\n";
        for (int i = 0; i < 10; i++)
        {
            f << "/t1/image" << i << ".png" << manifest_file::get_delimiter();
            f << "/t1/target" << i << ".txt\n";
        }
        f.close();
        nervana::manifest_file manifest(manifest_file, false, "/x1");
        ASSERT_EQ(1, manifest.block_count());
        auto& block = *manifest.next();
        for (int i = 0; i < manifest.record_count(); i++)
        {
            const vector<string>& x = block[i];

            ASSERT_EQ(2, x.size());
            stringstream ss;
            ss << "/t1/image" << i << ".png";
            EXPECT_STREQ(x[0].c_str(), ss.str().c_str());
            ss.str("");
            ss << "/t1/target" << i << ".txt";
            EXPECT_STREQ(x[1].c_str(), ss.str().c_str());
        }
    }
    {
        ofstream f(manifest_file);
        f << "@FILE\tFILE\n";
        for (int i = 0; i < 10; i++)
        {
            f << "t1/image" << i << ".png" << manifest_file::get_delimiter();
            f << "t1/target" << i << ".txt\n";
        }
        f.close();
        nervana::manifest_file manifest(manifest_file, false, "/x1");
        ASSERT_EQ(1, manifest.block_count());
        auto& block = *manifest.next();
        for (int i = 0; i < manifest.record_count(); i++)
        {
            const vector<string>& x = block[i];

            ASSERT_EQ(2, x.size());
            stringstream ss;
            ss << "/x1/t1/image" << i << ".png";
            EXPECT_STREQ(x[0].c_str(), ss.str().c_str());
            ss.str("");
            ss << "/x1/t1/target" << i << ".txt";
            EXPECT_STREQ(x[1].c_str(), ss.str().c_str());
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

    EXPECT_EQ(expected, actual);
}

TEST(manifest, file_implicit)
{
    vector<string> image_files  = {"flowers.jpg", "img_2112_70.jpg"};
    vector<string> target_files = {"1.txt", "2.txt"};

    stringstream ss;

    for (int count = 0; count < 32; count++)
    {
        for (int i = 0; i < image_files.size(); i++)
        {
            ss << image_files[i] << "\t" << target_files[i] << "\n";
        }
    }

    size_t block_size = 16;
    EXPECT_THROW(manifest_file(ss, false, test_data_directory, 1.0, block_size),
                 std::invalid_argument);
}

TEST(manifest, wrong_elements_number)
{
    string image_file = "flowers.jpg";

    stringstream ss;
    ss << "@FILE"
       << "\t"
       << "FILE"
       << "\n";

    ss << image_file << "\n";

    size_t block_size = 1;

    EXPECT_THROW(manifest_file manifest(ss, false, test_data_directory, 1.0, block_size),
                 std::runtime_error);
}

TEST(manifest, changing_elements_number)
{
    string image_file  = "flowers.jpg";
    string target_file = "1.txt";

    stringstream ss;
    ss << "@FILE"
       << "\t"
       << "FILE"
       << "\n";

    ss << image_file << "\t" << target_file << "\n";
    ss << image_file << "\n";

    size_t block_size = 2;

    EXPECT_THROW(manifest_file manifest(ss, false, test_data_directory, 1.0, block_size),
                 std::runtime_error);
}
TEST(manifest, file_explicit)
{
    vector<string> image_files  = {"flowers.jpg", "img_2112_70.jpg"};
    vector<string> target_files = {"1.txt", "2.txt"};

    stringstream ss;
    ss << "@FILE"
       << "\t"
       << "FILE"
       << "\n";
    for (int count = 0; count < 32; count++)
    {
        for (int i = 0; i < image_files.size(); i++)
        {
            ss << image_files[i] << "\t" << target_files[i] << "\n";
        }
    }

    size_t block_size = 16;
    auto   manifest   = make_shared<manifest_file>(ss, false, test_data_directory, 1.0, block_size);

    auto types = manifest->get_element_types();
    ASSERT_EQ(2, types.size());
    EXPECT_EQ(manifest::element_t::FILE, types[0]);
    EXPECT_EQ(manifest::element_t::FILE, types[1]);

    block_loader_file bload{manifest, block_size};
    for (int i = 0; i < 2; i++)
    {
        encoded_record_list* buffer = bload.filler();
        ASSERT_NE(nullptr, buffer);
        ASSERT_EQ(block_size, buffer->size());
        for (int j = 0; j < buffer->size(); j++)
        {
            encoded_record record = buffer->record(j);
            auto           idata  = record.element(0);
            auto           tdata  = record.element(1);
            string         target{tdata.data(), tdata.size()};
            int            value = stod(target);
            EXPECT_EQ(j % 2 + 1, value);
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
    stringstream   ss;
    size_t         index = 0;
    ss << "@FILE"
       << "\t"
       << "BINARY"
       << "\n";
    for (int count = 0; count < 32; count++)
    {
        for (int i = 0; i < image_files.size(); i++)
        {
            vector<char> str         = string2vector(make_target_data(index++));
            auto         data_string = base64::encode(str);
            ss << image_files[i] << "\t" << vector2string(data_string) << "\n";
        }
    }

    size_t block_size = 16;
    auto   manifest   = make_shared<manifest_file>(ss, false, test_data_directory, 1.0, block_size);

    auto types = manifest->get_element_types();
    ASSERT_EQ(2, types.size());
    EXPECT_EQ(manifest::element_t::FILE, types[0]);
    EXPECT_EQ(manifest::element_t::BINARY, types[1]);

    block_loader_file block_loader{manifest, block_size};
    index = 0;
    for (int i = 0; i < 2; i++)
    {
        encoded_record_list* buffer = block_loader.filler();
        ASSERT_NE(nullptr, buffer);
        ASSERT_EQ(block_size, buffer->size());
        for (int j = 0; j < buffer->size(); j++)
        {
            encoded_record record   = buffer->record(j);
            auto           idata    = record.element(0);
            auto           tdata    = record.element(1);
            string         str      = vector2string(tdata);
            string         expected = make_target_data(index);
            index                   = (index + 1) % manifest->record_count();
            EXPECT_STREQ(expected.c_str(), str.c_str());
        }
    }
}

TEST(manifest, string)
{
    vector<string> image_files = {"flowers.jpg", "img_2112_70.jpg"};
    stringstream   ss;
    size_t         index = 0;
    ss << "@FILE"
       << "\t"
       << "STRING"
       << "\n";
    for (int count = 0; count < 32; count++)
    {
        for (int i = 0; i < image_files.size(); i++)
        {
            string str = make_target_data(index++);
            ss << image_files[i] << "\t" << str << "\n";
        }
    }

    size_t block_size = 16;
    auto   manifest   = make_shared<manifest_file>(ss, false, test_data_directory, 1.0, block_size);

    //    for (auto data : manifest)
    //    {
    //        INFO << data[0] << ", " << data[1];
    //    }

    auto types = manifest->get_element_types();
    ASSERT_EQ(2, types.size());
    EXPECT_EQ(manifest::element_t::FILE, types[0]);
    EXPECT_EQ(manifest::element_t::STRING, types[1]);

    block_loader_file block_loader{manifest, block_size};
    index = 0;
    for (int i = 0; i < 2; i++)
    {
        encoded_record_list* buffer = block_loader.filler();
        ASSERT_NE(nullptr, buffer);
        ASSERT_EQ(block_size, buffer->size());
        for (int j = 0; j < buffer->size(); j++)
        {
            encoded_record record   = buffer->record(j);
            auto           idata    = record.element(0);
            auto           tdata    = record.element(1);
            string         str      = vector2string(tdata);
            string         expected = make_target_data(index);
            index                   = (index + 1) % manifest->record_count();
            EXPECT_STREQ(expected.c_str(), str.c_str());
        }
    }
}

TEST(manifest, ascii_int)
{
    vector<string> image_files = {"flowers.jpg", "img_2112_70.jpg"};
    stringstream   ss;
    size_t         index = 0;
    ss << "@FILE"
       << "\t"
       << "ASCII_INT"
       << "\n";
    for (int count = 0; count < 32; count++)
    {
        for (int i = 0; i < image_files.size(); i++)
        {
            ss << image_files[i] << "\t" << count * 2 + i << "\n";
        }
    }

    size_t block_size = 16;
    auto   manifest   = make_shared<manifest_file>(ss, false, test_data_directory, 1.0, block_size);

    auto types = manifest->get_element_types();
    ASSERT_EQ(2, types.size());
    EXPECT_EQ(manifest::element_t::FILE, types[0]);
    EXPECT_EQ(manifest::element_t::ASCII_INT, types[1]);

    block_loader_file block_loader{manifest, block_size};
    index = 0;
    for (int i = 0; i < 2; i++)
    {
        encoded_record_list* buffer = block_loader.filler();
        ASSERT_NE(nullptr, buffer);
        ASSERT_EQ(block_size, buffer->size());
        for (int j = 0; j < buffer->size(); j++)
        {
            encoded_record record = buffer->record(j);
            auto           idata  = record.element(0);
            int32_t        tdata  = unpack<int32_t>(record.element(1).data());
            EXPECT_EQ(tdata, index);
            index++;
        }
    }
}

TEST(manifest, ascii_float)
{
    vector<string> image_files = {"flowers.jpg", "img_2112_70.jpg"};
    stringstream   ss;
    size_t         index = 0;
    ss << "@FILE"
       << "\t"
       << "ASCII_FLOAT"
       << "\n";
    for (int count = 0; count < 32; count++)
    {
        for (int i = 0; i < image_files.size(); i++)
        {
            ss << image_files[i] << "\t" << (float)(count * 2 + i) / 10. << "\n";
        }
    }

    size_t block_size = 16;
    auto   manifest   = make_shared<manifest_file>(ss, false, test_data_directory, 1.0, block_size);

    auto types = manifest->get_element_types();
    ASSERT_EQ(2, types.size());
    EXPECT_EQ(manifest::element_t::FILE, types[0]);
    EXPECT_EQ(manifest::element_t::ASCII_FLOAT, types[1]);

    block_loader_file block_loader{manifest, block_size};
    index = 0;
    for (int i = 0; i < 2; i++)
    {
        encoded_record_list* buffer = block_loader.filler();
        ASSERT_NE(nullptr, buffer);
        ASSERT_EQ(block_size, buffer->size());
        for (int j = 0; j < buffer->size(); j++)
        {
            encoded_record record   = buffer->record(j);
            auto           idata    = record.element(0);
            float          tdata    = unpack<float>(record.element(1).data());
            float          expected = (float)index / 10.;
            EXPECT_EQ(tdata, expected);
            index++;
        }
    }
}

extern string test_cache_directory;

class manifest_manager
{
public:
    manifest_manager(const string& source_dir, size_t count, int rows, int cols)
    {
        test_root         = source_dir;
        source_directory  = file_util::make_temp_directory(source_dir);
        manifest_filename = file_util::path_join(source_directory, "manifest.tsv");
        file_list.push_back(manifest_filename);
        ofstream mfile(manifest_filename);
        mfile << "@FILE\tFILE\n";
        for (size_t i = 0; i < count; i++)
        {
            cv::Mat image  = embedded_id_image::generate_image(rows, cols, i);
            string  number = to_string(i);
            string  image_filename =
                file_util::path_join(source_directory, "image" + number + ".png");
            string target_filename =
                file_util::path_join(source_directory, "target" + number + ".txt");
            //            cout << image_filename << ", " << target_filename << endl;
            file_list.push_back(image_filename);
            file_list.push_back(target_filename);
            cv::imwrite(image_filename, image);
            ofstream tfile(target_filename);
            tfile << i;
            mfile << image_filename << manifest_file::get_delimiter();
            mfile << target_filename << "\n";
        }
    }

    const string& manifest_file() const { return manifest_filename; }
    ~manifest_manager() { file_util::remove_directory(test_root); }
private:
    string         test_root;
    string         manifest_filename;
    string         source_directory;
    vector<string> file_list;
};

TEST(manifest, manifest_shuffle)
{
    const uint32_t   seed            = 1234;
    const size_t     block_size      = 4;
    const float      subset_fraction = 1.0;
    string           source_dir      = file_util::make_temp_directory(test_cache_directory);
    manifest_manager manifest_builder{source_dir, 10, 25, 25};

    string manifest_root;

    nervana::manifest_file manifest1{
        manifest_builder.manifest_file(), true, manifest_root, subset_fraction, block_size, seed};
    nervana::manifest_file manifest2{
        manifest_builder.manifest_file(), false, manifest_root, subset_fraction, block_size, seed};

    EXPECT_EQ(manifest1.get_crc(), manifest2.get_crc());
}

TEST(manifest, manifest_shuffle_repeatable)
{
    const uint32_t   seed            = 1234;
    const size_t     block_size      = 4;
    const float      subset_fraction = 1.0;
    string           source_dir      = file_util::make_temp_directory(test_cache_directory);
    manifest_manager manifest_builder{source_dir, 10, 25, 25};

    string manifest_root;

    nervana::manifest_file manifest1{
        manifest_builder.manifest_file(), true, manifest_root, subset_fraction, block_size, seed};
    nervana::manifest_file manifest2{
        manifest_builder.manifest_file(), true, manifest_root, subset_fraction, block_size, seed};

    EXPECT_EQ(manifest1.get_crc(), manifest2.get_crc());
}

TEST(manifest, subset_fraction)
{
    string           source_dir = file_util::make_temp_directory(test_cache_directory);
    manifest_manager manifest_builder{source_dir, 1000, 25, 25};

    uint32_t manifest1_crc;
    uint32_t manifest2_crc;

    const uint32_t seed             = 1234;
    float          subset_fraction  = 0.01;
    int            block_size       = 4;
    bool           shuffle_manifest = true;
    string         manifest_root;

    {
        auto manifest = make_shared<nervana::manifest_file>(manifest_builder.manifest_file(),
                                                            shuffle_manifest,
                                                            manifest_root,
                                                            subset_fraction,
                                                            block_size,
                                                            seed);

        ASSERT_NE(nullptr, manifest);

        manifest1_crc = manifest->get_crc();
    }

    {
        auto manifest = make_shared<nervana::manifest_file>(manifest_builder.manifest_file(),
                                                            shuffle_manifest,
                                                            manifest_root,
                                                            subset_fraction,
                                                            block_size,
                                                            seed);

        ASSERT_NE(nullptr, manifest);

        manifest2_crc = manifest->get_crc();
    }

    EXPECT_EQ(manifest1_crc, manifest2_crc);
}

TEST(manifest, crc_root_dir)
{
    stringstream ss;
    ss << manifest_file::get_metadata_char();
    ss << manifest_file::get_file_type_id() << manifest_file::get_delimiter()
       << manifest_file::get_string_type_id() << "\n";
    for (size_t i = 0; i < 100; i++)
    {
        ss << "relative/path/image" << i << ".jpg" << manifest_file::get_delimiter() << i << "\n";
    }

    uint32_t manifest1_crc;
    uint32_t manifest2_crc;

    float subset_fraction  = 1.0;
    int   block_size       = 4;
    bool  shuffle_manifest = false;

    {
        stringstream tmp{ss.str()};
        string       manifest_root = "/root1/";
        auto         manifest      = make_shared<nervana::manifest_file>(
            tmp, shuffle_manifest, manifest_root, subset_fraction, block_size);

        ASSERT_NE(nullptr, manifest);

        manifest1_crc = manifest->get_crc();
    }

    {
        stringstream tmp{ss.str()};
        string       manifest_root = "/root2/";
        auto         manifest      = make_shared<nervana::manifest_file>(
            tmp, shuffle_manifest, manifest_root, subset_fraction, block_size);

        ASSERT_NE(nullptr, manifest);

        manifest2_crc = manifest->get_crc();
    }

    EXPECT_EQ(manifest1_crc, manifest2_crc);
}

TEST(manifest, comma)
{
    string manifest_file = "tmp_manifest.tsv";
    {
        int            height       = 224;
        int            width        = 224;
        size_t         batch_size   = 128;
        nlohmann::json image_config = {
            {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
        nlohmann::json label_config = {{"type", "label"}, {"binary", false}};
        nlohmann::json config       = {{"manifest_filename", manifest_file},
                                 {"batch_size", batch_size},
                                 {"iteration_mode", "INFINITE"},
                                 {"etl", {image_config, label_config}}};

        loader_factory factory;
        EXPECT_THROW(factory.get_loader(config), std::runtime_error);
    }
}
