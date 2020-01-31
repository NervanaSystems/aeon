/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <linux/limits.h>

#include "gtest/gtest.h"
#include "file_util.hpp"

#define private public

using namespace std;
using namespace nervana;

TEST(file_util, path_join)
{
    {
        string s1 = "";
        string s2 = "";

        EXPECT_STREQ("", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "";
        string s2 = "/test1/test2";

        EXPECT_STREQ("/test1/test2", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "";
        string s2 = "/test1/test2/";

        EXPECT_STREQ("/test1/test2/", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "";
        string s2 = "test1/test2";

        EXPECT_STREQ("test1/test2", file_util::path_join(s1, s2).c_str());
    }

    {
        string s1 = "/x1/x2";
        string s2 = "";

        EXPECT_STREQ("/x1/x2", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "/x1/x2/";
        string s2 = "/";

        EXPECT_STREQ("/", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "/x1/x2";
        string s2 = "/test1/test2";

        EXPECT_STREQ("/test1/test2", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "/x1/x2/";
        string s2 = "test1/test2";

        EXPECT_STREQ("/x1/x2/test1/test2", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "/x1/x2";
        string s2 = "test1/test2";

        EXPECT_STREQ("/x1/x2/test1/test2", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "/";
        string s2 = "test1/test2";

        EXPECT_STREQ("/test1/test2", file_util::path_join(s1, s2).c_str());
    }
}

class file_util_tmp_dir_env : public ::testing::Test
{
public:
    void SetTmpDirEnvVar(std::string new_value)
    {
        char* prevous_value = getenv(var_name);
        if (prevous_value == nullptr) {
            env_var_already_existed = false;
        } else {
            env_var_already_existed = true;
            env_var_old_value = prevous_value;
        }
        setenv(var_name, new_value.c_str(), true);
    }

private:
    void TearDown() override
    {
        if (env_var_already_existed)
            setenv(var_name, env_var_old_value.c_str(), true);
        else
            unsetenv(var_name);
    }

    const char* var_name = "NERVANA_AEON_TMP";
    bool        env_var_already_existed;
    std::string env_var_old_value;
    const char* default_path = "/tmp";
};

TEST_F(file_util_tmp_dir_env, get_temp_directory_short)
{
    std::string short_path = "/tmp/aeon/short/path";
    SetTmpDirEnvVar(short_path);
    EXPECT_STREQ(short_path.c_str(), file_util::get_temp_directory(PATH_MAX - 50).c_str());
}

TEST_F(file_util_tmp_dir_env, get_temp_directory_empty)
{
    std::string empty_path;
    SetTmpDirEnvVar(empty_path);
    EXPECT_STREQ(empty_path.c_str(), file_util::get_temp_directory(PATH_MAX - 50).c_str());
}

TEST_F(file_util_tmp_dir_env, get_temp_directory_too_long)
{
    SetTmpDirEnvVar(std::string(6000, 'A'));
    std::string tmp_path;
    EXPECT_NO_THROW(tmp_path = file_util::get_temp_directory(PATH_MAX + 50));
    EXPECT_STREQ(default_path, tmp_path.c_str());
}

TEST_F(file_util_tmp_dir_env, make_temp_directory_by_env) 
{
    std::string prefix = "/tmp/aeon_test";
    mkdir(prefix.c_str(), S_IRWXU);
    SetTmpDirEnvVar(prefix);
    auto tmp_path = file_util::make_temp_directory();
    EXPECT_STREQ(prefix.c_str(), tmp_path.substr(0,prefix.size()).c_str());
    struct stat info;
    EXPECT_EQ(0, stat(tmp_path.c_str(), &info));
    EXPECT_TRUE(info.st_mode & S_IFDIR);
    rmdir(prefix.c_str());
}

TEST_F(file_util_tmp_dir_env, make_temp_directory_by_provided) 
{
    std::string prefix = "/tmp/aeon_test";
    mkdir(prefix.c_str(), S_IRWXU);
    auto tmp_path = file_util::make_temp_directory(prefix);
    EXPECT_STREQ(prefix.c_str(), tmp_path.substr(0,prefix.size()).c_str());
    struct stat info;
    EXPECT_EQ(0, stat(tmp_path.c_str(), &info));
    EXPECT_TRUE(info.st_mode & S_IFDIR);
    rmdir(prefix.c_str());
}

TEST_F(file_util_tmp_dir_env, make_temp_directory_too_long_by_env) 
{
    std::string tmp_path = "/tmp/aeon_test/";
    tmp_path += std::string(PATH_MAX - tmp_path.size() - 13, 'A');
    SetTmpDirEnvVar(tmp_path);
    EXPECT_THROW(file_util::make_temp_directory(), std::runtime_error);
}

TEST_F(file_util_tmp_dir_env, make_temp_directory_too_long_by_provided) 
{
    std::string tmp_path = "/tmp/aeon_test/";
    tmp_path += std::string(PATH_MAX - tmp_path.size() - 13, 'A');
    EXPECT_THROW(file_util::make_temp_directory(tmp_path), std::runtime_error);
}
