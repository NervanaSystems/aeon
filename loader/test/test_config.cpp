#include <stdexcept>
#include "gtest/gtest.h"

#include "loader.hpp"

using namespace std;
using namespace nervana;

TEST(config,loader) {
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

TEST(config,throws) {
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
