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

#include <vector>
#include <string>
#include <sstream>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gtest/gtest.h"
#include "cpio.hpp"
#include "buffer_in.hpp"
#include "file_util.hpp"
#include "block_loader_file.hpp"
#include "block_loader_cpio_cache.hpp"

#define private public

using namespace std;
using namespace nervana;

extern string test_cache_directory;

cv::Mat generate_test_image(int rows, int cols, int embedded_id)
{
    cv::Mat  image{rows, cols, CV_8UC3};
    uint8_t* p = image.data;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            *p++ = uint8_t(embedded_id >> 16);
            *p++ = uint8_t(embedded_id >> 8);
            *p++ = uint8_t(embedded_id >> 0);
        }
    }
    return image;
}

int read_embedded_id(const cv::Mat& image)
{
    uint8_t* p = image.data;
    int      id;
    id = int(*p++ << 16);
    id |= int(*p++ << 8);
    id |= int(*p++ << 0);
    return id;
}

class manifest_manager
{
public:
    manifest_manager(const string& source_dir, size_t count, int rows, int cols)
    {
        test_root         = source_dir;
        source_directory  = file_util::make_temp_directory(source_dir);
        manifest_filename = file_util::path_join(source_directory, "manifest.csv");
        file_list.push_back(manifest_filename);
        ofstream mfile(manifest_filename);
        for (size_t i = 0; i < count; i++)
        {
            cv::Mat image           = generate_test_image(rows, cols, i);
            string  number          = to_string(i);
            string  image_filename  = file_util::path_join(source_directory, "image" + number + ".png");
            string  target_filename = file_util::path_join(source_directory, "target" + number + ".txt");
            //            cout << image_filename << ", " << target_filename << endl;
            file_list.push_back(image_filename);
            file_list.push_back(target_filename);
            cv::imwrite(image_filename, image);
            ofstream tfile(target_filename);
            tfile << i;
            mfile << image_filename << ",";
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

TEST(cpio_cache, manifest_shuffle)
{
    string           source_dir = file_util::make_temp_directory(test_cache_directory);
    manifest_manager manifest_builder{source_dir, 10, 25, 25};

    string manifest_root;

    nervana::manifest_csv manifest1{manifest_builder.manifest_file(), true, manifest_root};
    nervana::manifest_csv manifest2{manifest_builder.manifest_file(), false, manifest_root};

    EXPECT_NE(manifest1.get_crc(), manifest2.get_crc());
}

TEST(cpio_cache, manifest_shuffle_repeatable)
{
    string           source_dir = file_util::make_temp_directory(test_cache_directory);
    manifest_manager manifest_builder{source_dir, 10, 25, 25};

    string manifest_root;

    nervana::manifest_csv manifest1{manifest_builder.manifest_file(), false, manifest_root};
    nervana::manifest_csv manifest2{manifest_builder.manifest_file(), true, manifest_root};
    nervana::manifest_csv manifest3{manifest_builder.manifest_file(), true, manifest_root};

    EXPECT_NE(manifest1.get_crc(), manifest2.get_crc());
    EXPECT_EQ(manifest2.get_crc(), manifest3.get_crc());
}

TEST(cpio_cache, subset_fraction)
{
    string           source_dir = file_util::make_temp_directory(test_cache_directory);
    manifest_manager manifest_builder{source_dir, 1000, 25, 25};

    uint32_t manifest1_crc;
    uint32_t manifest2_crc;

    float  subset_fraction  = 0.01;
    int    macrobatch_size  = 4;
    bool   shuffle_manifest = true;
    string manifest_root;

    {
        auto manifest = make_shared<nervana::manifest_csv>(manifest_builder.manifest_file(), shuffle_manifest, manifest_root);

        ASSERT_NE(nullptr, manifest);

        auto block_loader = make_shared<block_loader_file>(manifest, subset_fraction, macrobatch_size);
        ASSERT_NE(nullptr, block_loader);

        manifest1_crc = manifest->get_crc();
    }

    {
        auto manifest = make_shared<nervana::manifest_csv>(manifest_builder.manifest_file(), shuffle_manifest, manifest_root);

        ASSERT_NE(nullptr, manifest);

        auto block_loader = make_shared<block_loader_file>(manifest, subset_fraction, macrobatch_size);
        ASSERT_NE(nullptr, block_loader);

        manifest2_crc = manifest->get_crc();
    }

    EXPECT_EQ(manifest1_crc, manifest2_crc);
}