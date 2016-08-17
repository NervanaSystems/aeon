#include <iostream>
#include <cstdio>

#include "gtest/gtest.h"
#include "log.hpp"

using namespace std;
using namespace nervana;

TEST(logging,conststring) {
    {
        const char* s = find_last("this/is/a/test",'/');
        EXPECT_STREQ("test",s);
    }
    {
        const char* s = find_last("test",'/');
        EXPECT_STREQ("test",s);
    }
}

TEST(logging,error) {
    INFO << "This is info";
    WARN << "This is warn";
    ERR << "This is error";
}
