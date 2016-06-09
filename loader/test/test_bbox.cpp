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
#include "argtype.hpp"
#include "datagen.hpp"
#include "batchfile.hpp"

#include "params.hpp"
#include "etl_interface.hpp"
#include "etl_image.hpp"
#include "etl_label.hpp"
#include "etl_bbox.hpp"
#include "etl_lmap.hpp"
#include "provider.hpp"
#include "json.hpp"

extern DataGen _datagen;

using namespace std;
using namespace nervana;

static vector<string> label_list = {"person","dog","lion","tiger","eel","puma","rat","tick","flea","bicycle"};

static string read_file( const string& path ) {
    ifstream f(path);
    stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

TEST(etl, bbox_extractor) {
    {
        string data = read_file(CURDIR"/test_data/000001.json");
        bbox::extractor extractor{label_list};
        auto mdata = extractor.extract(&data[0],data.size());
        auto decoded = static_pointer_cast<nervana::bbox::decoded>(mdata);
        ASSERT_NE(nullptr,decoded);
        auto boxes = decoded->boxes();
        ASSERT_EQ(2,boxes.size());

        EXPECT_EQ(194,boxes[0].xmax);
        EXPECT_EQ(47,boxes[0].xmin);
        EXPECT_EQ(370,boxes[0].ymax);
        EXPECT_EQ(239,boxes[0].ymin);
        EXPECT_FALSE(boxes[0].difficult);
        EXPECT_TRUE(boxes[0].truncated);
        EXPECT_EQ(1,boxes[0].label);

        EXPECT_EQ(351,boxes[1].xmax);
        EXPECT_EQ(7,boxes[1].xmin);
        EXPECT_EQ(497,boxes[1].ymax);
        EXPECT_EQ(11,boxes[1].ymin);
        EXPECT_FALSE(boxes[1].difficult);
        EXPECT_TRUE(boxes[1].truncated);
        EXPECT_EQ(0,boxes[1].label);
    }
    {
        string data = read_file(CURDIR"/test_data/006637.json");
        bbox::extractor extractor{label_list};
        auto mdata = extractor.extract(&data[0],data.size());
        auto decoded = static_pointer_cast<nervana::bbox::decoded>(mdata);
        ASSERT_NE(nullptr,decoded);
        auto boxes = decoded->boxes();
        ASSERT_EQ(6,boxes.size());

        auto b = boxes[3];
        EXPECT_EQ(365,b.xmax);
        EXPECT_EQ(324,b.xmin);
        EXPECT_EQ(315,b.ymax);
        EXPECT_EQ(109,b.ymin);
        EXPECT_FALSE(b.difficult);
        EXPECT_FALSE(b.truncated);
        EXPECT_EQ(0,b.label);
    }
    {
        string data = read_file(CURDIR"/test_data/009952.json");
        bbox::extractor extractor{label_list};
        auto mdata = extractor.extract(&data[0],data.size());
        auto decoded = static_pointer_cast<nervana::bbox::decoded>(mdata);
        ASSERT_NE(nullptr,decoded);
        auto boxes = decoded->boxes();
        ASSERT_EQ(1,boxes.size());
    }
}

TEST(etl, bbox) {
    // Create test metadata
    cv::Rect r0 = cv::Rect( 0, 0, 10, 15 );
    cv::Rect r1 = cv::Rect( 10, 10, 12, 13 );
    cv::Rect r2 = cv::Rect( 100, 100, 120, 130 );
    auto list = {bbox::extractor::create_box( r0, "rat" ),
                  bbox::extractor::create_box( r1, "flea" ),
                  bbox::extractor::create_box( r2, "tick")};
    auto j = bbox::extractor::create_metadata(list);
    // cout << std::setw(4) << j << endl;

    string buffer = j.dump();
    // cout << "boxes\n" << buffer << endl;

    bbox::extractor extractor{label_list};
    auto data = extractor.extract( &buffer[0], buffer.size() );
    shared_ptr<bbox::decoded> decoded = static_pointer_cast<bbox::decoded>(data);
    vector<bbox::box> boxes = decoded->boxes();
    ASSERT_EQ(3,boxes.size());
    EXPECT_EQ(r0,boxes[0].rect());
    EXPECT_EQ(r1,boxes[1].rect());
    EXPECT_EQ(r2,boxes[2].rect());
    EXPECT_EQ(6,boxes[0].label);
    EXPECT_EQ(8,boxes[1].label);
    EXPECT_EQ(7,boxes[2].label);

    bbox::transformer transform;
    shared_ptr<image::params> iparam = make_shared<image::params>();
    auto tx = transform.transform( iparam, decoded );
}

TEST(etl, bbox_transform) {
    // Create test metadata
    cv::Rect r0 = cv::Rect( 10, 10, 10, 10 );   // outside
    cv::Rect r1 = cv::Rect( 30, 30, 10, 10 );   // result[0]
    cv::Rect r2 = cv::Rect( 50, 50, 10, 10 );   // result[1]
    cv::Rect r3 = cv::Rect( 70, 30, 10, 10 );   // result[2]
    cv::Rect r4 = cv::Rect( 90, 35, 10, 10 );   // outside
    cv::Rect r5 = cv::Rect( 30, 70, 10, 10 );   // result[3]
    cv::Rect r6 = cv::Rect( 70, 70, 10, 10 );   // result[4]
    cv::Rect r7 = cv::Rect( 30, 30, 80, 80 );   // result[5]
    auto list = {bbox::extractor::create_box( r0, "lion" ),
                  bbox::extractor::create_box( r1, "tiger" ),
                  bbox::extractor::create_box( r2, "eel" ),
                  bbox::extractor::create_box( r3, "eel" ),
                  bbox::extractor::create_box( r4, "eel" ),
                  bbox::extractor::create_box( r5, "eel" ),
                  bbox::extractor::create_box( r6, "eel" ),
                  bbox::extractor::create_box( r7, "eel" )};
    auto j = bbox::extractor::create_metadata(list);
    // cout << std::setw(4) << j << endl;

    string buffer = j.dump();

    bbox::extractor extractor{label_list};
    auto data = extractor.extract( &buffer[0], buffer.size() );
    shared_ptr<bbox::decoded> decoded = static_pointer_cast<bbox::decoded>(data);
    vector<bbox::box> boxes = decoded->boxes();

    ASSERT_EQ(8,boxes.size());

    bbox::transformer transform;
    shared_ptr<image::params> iparam = make_shared<image::params>();
    iparam->cropbox = cv::Rect( 35, 35, 40, 40 );
    auto tx = transform.transform( iparam, decoded );
    shared_ptr<bbox::decoded> tx_decoded = static_pointer_cast<bbox::decoded>(tx);
    vector<bbox::box> tx_boxes = tx_decoded->boxes();
    ASSERT_EQ(6,tx_boxes.size());
    EXPECT_EQ(cv::Rect(35,35,5,5),tx_boxes[0].rect());
    EXPECT_EQ(cv::Rect(50,50,10,10),tx_boxes[1].rect());
    EXPECT_EQ(cv::Rect(70,35,5,5),tx_boxes[2].rect());
    EXPECT_EQ(cv::Rect(35,70,5,5),tx_boxes[3].rect());
    EXPECT_EQ(cv::Rect(70,70,5,5),tx_boxes[4].rect());
    EXPECT_EQ(cv::Rect(35,35,40,40),tx_boxes[5].rect());
}

TEST(etl, bbox_angle) {
    // Create test metadata
    cv::Rect r0 = cv::Rect( 10, 10, 10, 10 );
    auto list = {bbox::extractor::create_box( r0, "puma" )};
    auto j = bbox::extractor::create_metadata(list);

    string buffer = j.dump();

    bbox::extractor extractor{label_list};
    auto data = extractor.extract( &buffer[0], buffer.size() );
    shared_ptr<bbox::decoded> decoded = static_pointer_cast<bbox::decoded>(data);
    vector<bbox::box> boxes = decoded->boxes();

    ASSERT_EQ(1,boxes.size());

    bbox::transformer transform;
    shared_ptr<image::params> iparam = make_shared<image::params>();
    iparam->angle = 5;
    auto tx = transform.transform( iparam, decoded );
    shared_ptr<bbox::decoded> tx_decoded = static_pointer_cast<bbox::decoded>(tx);
    EXPECT_EQ(nullptr,tx_decoded.get());
}
