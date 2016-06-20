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

#include "etl_video.hpp"
#include "gen_video.hpp"

using namespace std;
using namespace nervana;

TEST(etl, video_integration) {
    // 1 second video at 25 FPS
    vector<unsigned char> vid = gen_video().encode(1000);

    auto config = make_shared<video::config>();
    video::extractor extractor = video::extractor(config);
    auto decoded_vid = extractor.extract((char*)vid.data(), vid.size());

    ASSERT_EQ(decoded_vid->get_image_count(), 25);
}
