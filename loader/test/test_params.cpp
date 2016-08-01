#include "gtest/gtest.h"
#include "util.hpp"
#include "interface.hpp"

using namespace std;
using namespace nervana;

enum class foo {
    TEST1,
    TEST2,
    TEST3
};

static void from_string(foo& v, const std::string& s) {
    string tmp = nervana::tolower(s);
         if(tmp == "test1") v = foo::TEST1;
    else if(tmp == "test2") v = foo::TEST2;
    else if(tmp == "test3") v = foo::TEST3;
}

TEST(params,parse_enum) {
    foo bar;
    nlohmann::json j1 = {{"test", "TEST2"}};

    interface::config::parse_enum(bar, "test", j1);
    EXPECT_EQ(bar, foo::TEST2);
}
