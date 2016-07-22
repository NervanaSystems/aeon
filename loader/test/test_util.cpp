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

#include "gtest/gtest.h"
#include "util.hpp"
#include "wav_data.hpp"
#include "cap_mjpeg_decoder.hpp"

using namespace std;
using namespace nervana;

TEST(util, unpack_le) {
    {
        char data[] = {1,0,0,0};
        int actual = unpack_le<int>(data);
        EXPECT_EQ(0x00000001,actual);
    }
    {
        char data[] = {0,1,0,0};
        int actual = unpack_le<int>(data);
        EXPECT_EQ(0x00000100,actual);
    }
    {
        char data[] = {0,0,0,1};
        int actual = unpack_le<int>(data);
        EXPECT_EQ(0x01000000,actual);
    }
    {
        char data[] = {0,0,0,1};
        int actual = unpack_le<int>(data,0,3);
        EXPECT_EQ(0,actual);
    }
    {
        char data[] = {0,0,0,1};
        int actual = unpack_le<int>(data,1,3);
        EXPECT_EQ(0x00010000,actual);
    }
    {
        char data[] = {(char)0x80,0,0,0};
        int actual = unpack_le<int>(data);
        EXPECT_EQ(128,actual);
    }
}

TEST(util, unpack_be) {
    {
        char data[] = {0,0,0,1};
        int actual = unpack_be<int>(data);
        EXPECT_EQ(0x00000001,actual);
    }
    {
        char data[] = {0,0,1,0};
        int actual = unpack_be<int>(data);
        EXPECT_EQ(0x00000100,actual);
    }
    {
        char data[] = {1,0,0,0};
        int actual = unpack_be<int>(data);
        EXPECT_EQ(0x01000000,actual);
    }
    {
        char data[] = {1,0,0,0};
        int actual = unpack_be<int>(data,0,3);
        EXPECT_EQ(0x00010000,actual);
    }
    {
        char data[] = {1,0,0,0};
        int actual = unpack_be<int>(data,1,3);
        EXPECT_EQ(0,actual);
    }
}


TEST(util, pack_le) {
    {
        char actual[] = {0,0,0,0};
        char expected[] = {1,0,0,0};
        pack_le<int>(actual,1);
        EXPECT_EQ(*(unsigned int*)expected,*(unsigned int*)actual);
    }
    {
        char actual[] = {0,0,0,0};
        char expected[] = {0,1,0,0};
        pack_le<int>(actual,0x00000100);
        EXPECT_EQ(*(unsigned int*)expected,*(unsigned int*)actual);
    }
    {
        char actual[] = {0,0,0,0};
        char expected[] = {0,0,0,1};
        pack_le<int>(actual,0x01000000);
        EXPECT_EQ(*(unsigned int*)expected,*(unsigned int*)actual);
    }
//    {
//        char actual[] = {0,0,0,0};
//        char expected[] = {0,0,0,1};
//        pack_le<int>(actual,0,3);
//        EXPECT_EQ(expected,actual);
//    }
//    {
//        char actual[] = {0,0,0,0};
//        char expected[] = {0,0,0,1};
//        pack_le<int>(actual,0x00010000,1,3);
//        EXPECT_EQ(expected,actual);
//    }
}

TEST(avi,video_file) {
    const string filename = "/home/users/alex/bb2.avi";
    shared_ptr<MotionJpegCapture> mjdecoder = make_shared<MotionJpegCapture>(filename);
    ASSERT_TRUE(mjdecoder->isOpened());
    cv::Mat image;
    int image_number = 0;
    while(mjdecoder->grabFrame() && mjdecoder->retrieveFrame(0,image)) {
        ASSERT_NE(0, image.size().area());
//        string output_name = "mjpeg_frame_"+to_string(image_number)+".jpg";
//        cv::imwrite(output_name,image);
        image_number++;
    }
    EXPECT_EQ(600,image_number);
}

TEST(avi,video_buffer) {
    const string filename = "/home/users/alex/bb2.avi";
    ifstream in(filename, ios_base::binary);
    ASSERT_TRUE(in);
    in.seekg(0,in.end);
    size_t size = in.tellg();
    in.seekg(0,in.beg);
    vector<char> data(size);
    data.assign(istreambuf_iterator<char>(in), istreambuf_iterator<char>());

    shared_ptr<MotionJpegCapture> mjdecoder = make_shared<MotionJpegCapture>(data.data(), data.size());
    ASSERT_TRUE(mjdecoder->isOpened());
    cv::Mat image;
    int image_number = 0;
    while(mjdecoder->grabFrame() && mjdecoder->retrieveFrame(0,image)) {
        ASSERT_NE(0, image.size().area());
//        string output_name = "mjpeg_frame_"+to_string(image_number)+".jpg";
//        cv::imwrite(output_name,image);
        image_number++;
    }
    EXPECT_EQ(600,image_number);
}

TEST(util,memstream) {
    string data = "abcdefghijklmnopqrstuvwxyz";
    memstream<char> ms((char*)data.data(),data.size());
    istream is(&ms);
    char buf[10];

    EXPECT_EQ(0,is.tellg());
    is.seekg(0,is.end);
    EXPECT_EQ(26,is.tellg());
    is.seekg(10,is.end);
    EXPECT_EQ(16,is.tellg());
    is.seekg(3,is.cur);
    EXPECT_EQ(19,is.tellg());
    is.seekg(3,is.beg);
    EXPECT_EQ(3,is.tellg());
    is.read(buf,2);
    EXPECT_EQ('d',buf[0]);
    EXPECT_EQ('e',buf[1]);
    is.read(buf,2);
    EXPECT_EQ('f',buf[0]);
    EXPECT_EQ('g',buf[1]);


    EXPECT_EQ(true,is.good());
    is.seekg(0,is.end);
    is.read(buf,2);     // read past end
    EXPECT_EQ(false,is.good());

    // test stream reset
}
