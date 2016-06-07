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

// static param_ptr _iep1 = make_shared<image::extract_params>();
// static param_ptr _itp1 = make_shared<image::transform_params>();
// static param_ptr _ilp1 = make_shared<image::load_params>();

// static param_ptr _lblp1 = make_shared<label::params>();


TEST(etl, lmap) {
    {
        vector<string> vocab = {"a","and","the","quick","fox","cow","dog","blue",
            "black","brown","happy","lazy","skip","jumped","run","under","over","around"};
        lmap::extractor extractor(vocab);
        auto data = extractor.get_data();
        EXPECT_EQ(2,data["the"]);

        {
            // the word 'jump' is not in the vocab
            string t1 = "the quick brown fox jump over the lazy dog";
            auto extracted = extractor.extract(&t1[0], t1.size());
            EXPECT_EQ(nullptr, extracted);
        }
        {
            string t1 = "the quick brown fox jumped over the lazy dog";
            vector<int> expected = {2, 3, 9, 4, 13, 16, 2, 11, 6};
            auto extracted = extractor.extract(&t1[0], t1.size());
            ASSERT_NE(nullptr, extracted);
            shared_ptr<lmap::decoded> decoded = static_pointer_cast<lmap::decoded>(extracted);
            ASSERT_EQ(expected.size(),decoded->get_data().size());
            for( int i=0; i<expected.size(); i++ ) {
                EXPECT_EQ(expected[i], decoded->get_data()[i]) << "at index " << i;
            }
        }
    }
    {
        stringstream vocab("a and the quick fox cow dog blue black brown happy lazy skip jumped run under over around");
        lmap::extractor extractor(vocab);
        auto data = extractor.get_data();
        EXPECT_EQ(2,data["the"]);

        {
            // the word 'jump' is not in the vocab
            string t1 = "the quick brown fox jump over the lazy dog";
            auto extracted = extractor.extract(&t1[0], t1.size());
            EXPECT_EQ(nullptr, extracted);
        }
        {
            string t1 = "the quick brown fox jumped over the lazy dog";
            vector<int> expected = {2, 3, 9, 4, 13, 16, 2, 11, 6};
            auto extracted = extractor.extract(&t1[0], t1.size());
            ASSERT_NE(nullptr, extracted);
            shared_ptr<lmap::decoded> decoded = static_pointer_cast<lmap::decoded>(extracted);
            ASSERT_EQ(expected.size(),decoded->get_data().size());
            for( int i=0; i<expected.size(); i++ ) {
                EXPECT_EQ(expected[i], decoded->get_data()[i]) << "at index " << i;
            }
        }
    }
}

static string read_file( const string& path ) {
    ifstream f(path);
    stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

TEST(etl, bbox_extractor) {
    {
        string data = read_file(CURDIR"/test_data/000001.json");
        bbox::extractor extractor;
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
        EXPECT_STREQ("dog",boxes[0].name.c_str());

        EXPECT_EQ(351,boxes[1].xmax);
        EXPECT_EQ(7,boxes[1].xmin);
        EXPECT_EQ(497,boxes[1].ymax);
        EXPECT_EQ(11,boxes[1].ymin);
        EXPECT_FALSE(boxes[1].difficult);
        EXPECT_TRUE(boxes[1].truncated);
        EXPECT_STREQ("person",boxes[1].name.c_str());
    }
    {
        string data = read_file(CURDIR"/test_data/006637.json");
        bbox::extractor extractor;
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
        EXPECT_STREQ("person",b.name.c_str());
    }
    {
        string data = read_file(CURDIR"/test_data/009952.json");
        bbox::extractor extractor;
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

    bbox::extractor extractor;
    auto data = extractor.extract( &buffer[0], buffer.size() );
    shared_ptr<bbox::decoded> decoded = static_pointer_cast<bbox::decoded>(data);
    vector<bbox::box> boxes = decoded->boxes();
    ASSERT_EQ(3,boxes.size());
    EXPECT_EQ(r0,boxes[0].rect());
    EXPECT_EQ(r1,boxes[1].rect());
    EXPECT_EQ(r2,boxes[2].rect());
    EXPECT_STREQ("rat",boxes[0].name.c_str());
    EXPECT_STREQ("flea",boxes[1].name.c_str());
    EXPECT_STREQ("tick",boxes[2].name.c_str());

    bbox::transformer transform;
    shared_ptr<image::settings> iparam = make_shared<image::settings>();
    settings_ptr sptr = static_pointer_cast<settings>(iparam);
    auto tx = transform.transform( sptr, decoded );
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

    bbox::extractor extractor;
    auto data = extractor.extract( &buffer[0], buffer.size() );
    shared_ptr<bbox::decoded> decoded = static_pointer_cast<bbox::decoded>(data);
    vector<bbox::box> boxes = decoded->boxes();

    ASSERT_EQ(8,boxes.size());

    bbox::transformer transform;
    shared_ptr<image::settings> iparam = make_shared<image::settings>();
    iparam->cropbox = cv::Rect( 35, 35, 40, 40 );
    settings_ptr sptr = static_pointer_cast<settings>(iparam);
    auto tx = transform.transform( sptr, decoded );
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

    bbox::extractor extractor;
    auto data = extractor.extract( &buffer[0], buffer.size() );
    shared_ptr<bbox::decoded> decoded = static_pointer_cast<bbox::decoded>(data);
    vector<bbox::box> boxes = decoded->boxes();

    ASSERT_EQ(1,boxes.size());

    bbox::transformer transform;
    shared_ptr<image::settings> iparam = make_shared<image::settings>();
    iparam->angle = 5;
    settings_ptr sptr = static_pointer_cast<settings>(iparam);
    auto tx = transform.transform( sptr, decoded );
    shared_ptr<bbox::decoded> tx_decoded = static_pointer_cast<bbox::decoded>(tx);
    EXPECT_EQ(nullptr,tx_decoded.get());
}

TEST(myloader, argtype) {

    {
        /* Create extractor with default num channels param */
        auto ie_p = make_shared<image::extractor>(make_shared<image::extract_params>("{}"));
        EXPECT_EQ(ie_p->get_channel_count(), 3);
    }


    {
        string argString = R"(
            {
                "height": 30,
                "width" : 30,
                "dist_params/angle" : [-20, 20],
                "dist_params/scale" : [0.2, 0.8],
                "dist_params/lighting" : [0.0, 0.1],
                "dist_params/aspect_ratio" : [0.75, 1.33],
                "dist_params/flip" : [false]
            }
        )";

        auto itpj = make_shared<image::transform_params>(argString);

        // output the fixed parameters
        cout << "HEIGHT: " << itpj->height << endl;
        cout << "WIDTH: "  << itpj->width  << endl;

        // output the random parameters
        default_random_engine r_eng(0);

        auto its = make_shared<image::settings>();
        its->dump();

        auto imgt = make_shared<image::transformer>(itpj);
        auto input_img_ptr = make_shared<decoded_images>(cv::Mat(256, 320, CV_8UC3));
        imgt->fill_settings(its, input_img_ptr, r_eng);
        its->dump();

    }


    {

        string argString = R"(
            {
                "extract offset":  20,
                "transform scale dist params":  [-8, 8],
                "transform shift dist params":  [-5, 5]
            }
        )";
        auto lblp = make_shared<label::params>(argString);

        BatchFileReader bf;
        auto dataFiles = _datagen.GetFiles();
        ASSERT_GT(dataFiles.size(),0);
        string batchFileName = dataFiles[0];
        bf.open(batchFileName);

        // Just get a single item
        auto data = bf.read();
        auto labels = bf.read();
        bf.close();

        auto lstg = make_shared<label::settings>();

        default_random_engine r_eng(0);
        lstg->scale = lblp->tx_scale(r_eng);
        lstg->shift = lblp->tx_shift(r_eng);

        cout << "Set scale: " << lstg->scale << " ";
        cout << "Set shift: " << lstg->shift << endl;

        int reference = ((int) (*labels)[0] + lblp->ex_offset)* lstg->scale + lstg->shift;

        // Take the int and do provision with it.
        auto lble = make_shared<label::extractor>(lblp);
        auto lblt = make_shared<label::transformer>(lblp);

        {
            auto lbll = make_shared<label::loader>(lblp);

            int reference_target = reference;
            int loaded_target = 0;
            provider pp(lble, lblt, lbll);
            pp.provide(labels->data(), 4, (char *)(&loaded_target), 4, lstg);
            EXPECT_EQ(reference_target, loaded_target);
        }

        {
            // This is a float loader that loads into a float with an offset
            string lArgString = R"(
                {
                    "load do float":  true,
                    "load offset": 0.8
                }
            )";
            auto flt_lbl_params = make_shared<label::params>(lArgString);

            auto lbll = make_shared<label::loader>(flt_lbl_params);

            float reference_target = reference + 0.8;
            float loaded_target = 0.0;
            provider pp(lble, lblt, lbll);
            pp.provide(labels->data(), 4, (char *)(&loaded_target), 4, lstg);
            EXPECT_EQ(reference_target, loaded_target);
        }

    }

}
