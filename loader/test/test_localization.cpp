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
#include "gen_image.hpp"

#include "params.hpp"
#include "etl_image.hpp"
#include "etl_localization.hpp"
#include "json.hpp"

using namespace std;
using namespace nervana;

TEST(localization,generate_anchors) {
    // Verify that we compute the same anchors as Shaoqing's matlab implementation:
    //    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
    //    >> anchors
    //
    //    anchors =
    //       -83   -39   100    56
    //      -175   -87   192   104
    //      -359  -183   376   200
    //       -55   -55    72    72
    //      -119  -119   136   136
    //      -247  -247   264   264
    //       -35   -79    52    96
    //       -79  -167    96   184
    //      -167  -343   184   360
    //
    // base_size 16, ratios [0.5, 1, 2], scales [ 8 16 32]

    vector<float> data = {  -83.0,  -39.0, 100.0,   56.0,
                           -175.0,  -87.0, 192.0,  104.0,
                           -359.0, -183.0, 376.0,  200.0,
                            -55.0,  -55.0,  72.0,   72.0,
                           -119.0, -119.0, 136.0,  136.0,
                           -247.0, -247.0, 264.0,  264.0,
                            -35.0,  -79.0,  52.0,   96.0,
                            -79.0, -167.0,  96.0,  184.0,
                           -167.0, -343.0, 184.0,  360.0 };
    cv::Mat expected = cv::Mat(9,4,CV_32FC1,&data[0]);

    // subtract 1 from the expected vector as it was generated with 1's based matlab
    expected -= 1;

    int base_size = 16;
    vector<float> ratios = {0.5, 1, 2};
    vector<float> scales = {8, 16, 32};

    cv::Mat actual = nervana::localization::transformer::generate_anchors(base_size, ratios, scales);
    ASSERT_EQ(expected.size(),actual.size());
    ASSERT_EQ(expected.type(),actual.type());
    float* actual_ptr = actual.ptr<float>();
    float* expected_ptr = expected.ptr<float>();
    for(int i=0; i<expected.size().area(); i++) {
        EXPECT_EQ((int)expected_ptr[i], (int)actual_ptr[i]);
    }
}

