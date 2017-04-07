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

#include "tiff.hpp"
#include "file_util.hpp"

#define private public

#include "etl_image.hpp"

using namespace std;
using namespace nervana;

TEST(tiff, uint16_t)
{
    const vector<uint8_t> buffer = {0x00, 0x01, 0x00, 0x02};
    bstream_mem           bs{buffer};

    auto v1 = bs.readU16();
    auto v2 = bs.readU16();
    EXPECT_EQ(256, v1);
    EXPECT_EQ(512, v2);
}

TEST(tiff, uint32_t)
{
    const vector<uint8_t> buffer = {0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02};
    bstream_mem           bs{buffer};

    auto v1 = bs.readU32();
    auto v2 = bs.readU32();
    EXPECT_EQ(1 << 24, v1);
    EXPECT_EQ(2 << 24, v2);
}

TEST(tiff, uint64_t)
{
    const vector<uint8_t> buffer = {0x01,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x02,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00};
    bstream_mem bs{buffer};

    auto v1 = bs.readU64();
    auto v2 = bs.readU64();
    EXPECT_EQ((uint64_t)1, v1);
    EXPECT_EQ((uint64_t)2, v2);
}

TEST(tiff, int8_t)
{
    const vector<uint8_t> buffer = {0x01, 0x02};
    bstream_mem           bs{buffer};

    auto v1 = bs.readS8();
    auto v2 = bs.readS8();
    EXPECT_EQ(1, v1);
    EXPECT_EQ(2, v2);
}

TEST(tiff, int16_t)
{
    const vector<uint8_t> buffer = {0xFF, 0xFF, 0xFE, 0xFF};
    bstream_mem           bs{buffer};

    auto v1 = bs.readS16();
    auto v2 = bs.readS16();
    EXPECT_EQ(-1, v1);
    EXPECT_EQ(-2, v2);
}

TEST(tiff, int32_t)
{
    const vector<uint8_t> buffer = {0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00};
    bstream_mem           bs{buffer};

    auto v1 = bs.readS32();
    auto v2 = bs.readS32();
    EXPECT_EQ(1, v1);
    EXPECT_EQ(2, v2);
}

TEST(tiff, int64_t)
{
    const vector<uint8_t> buffer = {0x01,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x02,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00};
    bstream_mem bs{buffer};

    auto v1 = bs.readS64();
    auto v2 = bs.readS64();
    EXPECT_EQ((int64_t)1, v1);
    EXPECT_EQ((int64_t)2, v2);
}

TEST(DISABLED_tiff, read_3band)
{
    string data_base = "/mnt/c/Users/rkimball/dev/tiff/data/";
    string f1        = data_base + "3band/3band_013022232200_Public_img6993.tif";
    cout << f1 << endl;
    auto f1_data = file_util::read_file_contents(f1);

    tiff::reader reader{f1_data.data(), f1_data.size()};
}

TEST(DISABLED_tiff, read_8band)
{
    string data_base = "/mnt/c/Users/rkimball/dev/tiff/data/";
    string f1        = data_base + "8band/8band_013022232200_Public_img6993.tif";
    cout << f1 << endl;
    auto f1_data = file_util::read_file_contents(f1);

    tiff::reader reader{f1_data.data(), f1_data.size()};
}

TEST(DISABLED_tiff, read_compressed)
{
    string data_base = "/mnt/c/Users/rkimball/dev/tiff/data/";
    string f1        = data_base + "opencv_tiff/3band_013022232200_Public_img6993.tif";
    cout << f1 << endl;
    auto f1_data = file_util::read_file_contents(f1);

    tiff::reader reader{f1_data.data(), f1_data.size()};
}
