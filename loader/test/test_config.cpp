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

#include <stdexcept>
#include "gtest/gtest.h"

#include "loader.hpp"

using namespace std;
using namespace nervana;

TEST(config,loader)
{
    // config is valid
    nlohmann::json js = {{"type","image,label"},
                         {"manifest_filename", "blah"},
                         {"minibatch_size", 128},
                         {"image", {
                            {"height",128},
                            {"width",128},
                            {"channel_major",false},
                            {"flip_enable",true}}},
                         };
    EXPECT_NO_THROW(loader_config   cfg{js});
}

TEST(config,throws)
{
    // config is missing a required parameter
    nlohmann::json js = {{"type","image,label"},
                         {"minibatch_size", 128},
                         {"image", {
                            {"height",128},
                            {"width",128},
                            {"channel_major",false},
                            {"flip_enable",true}}},
                         };
    EXPECT_THROW(loader_config cfg{js}, invalid_argument);
}
