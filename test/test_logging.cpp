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

#include <iostream>
#include <cstdio>
#include <stdlib.h>

#include "gtest/gtest.h"

#define private public
#include "log.hpp"

using namespace std;
using namespace nervana;

namespace
{
    void push_env();
    void pop_env();
}

TEST(logging, conststring)
{
    {
        const char* s = find_last("this/is/a/test", '/');
        EXPECT_STREQ("test", s);
    }
    {
        const char* s = find_last("test", '/');
        EXPECT_STREQ("test", s);
    }
}

TEST(logging, log_level_unset)
{
    push_env();
    unsetenv(log_level_env_var);

    {
        nervana::log_helper log(nervana::log_level::level_error, nervana::get_file_name(""), 0, "");
        EXPECT_TRUE(log.log_to_be_printed());
    }

    {
        nervana::log_helper log(
            nervana::log_level::level_warning, nervana::get_file_name(""), 0, "");
        EXPECT_TRUE(log.log_to_be_printed());
    }

    {
        nervana::log_helper log(nervana::log_level::level_info, nervana::get_file_name(""), 0, "");
        EXPECT_FALSE(log.log_to_be_printed());
    }

    pop_env();
}

TEST(logging, log_level_info)
{
    push_env();
    setenv(log_level_env_var, "INFO", 1);

    {
        nervana::log_helper log(nervana::log_level::level_error, nervana::get_file_name(""), 0, "");
        EXPECT_TRUE(log.log_to_be_printed());
    }

    {
        nervana::log_helper log(
            nervana::log_level::level_warning, nervana::get_file_name(""), 0, "");
        EXPECT_TRUE(log.log_to_be_printed());
    }

    {
        nervana::log_helper log(nervana::log_level::level_info, nervana::get_file_name(""), 0, "");
        EXPECT_TRUE(log.log_to_be_printed());
    }

    pop_env();
}

TEST(logging, log_level_warning)
{
    push_env();
    setenv(log_level_env_var, "WARNING", 1);

    {
        nervana::log_helper log(nervana::log_level::level_error, nervana::get_file_name(""), 0, "");
        EXPECT_TRUE(log.log_to_be_printed());
    }

    {
        nervana::log_helper log(
            nervana::log_level::level_warning, nervana::get_file_name(""), 0, "");
        EXPECT_TRUE(log.log_to_be_printed());
    }

    {
        nervana::log_helper log(nervana::log_level::level_info, nervana::get_file_name(""), 0, "");
        EXPECT_FALSE(log.log_to_be_printed());
    }

    pop_env();
}

TEST(logging, log_level_error)
{
    push_env();
    setenv(log_level_env_var, "ERROR", 1);

    {
        nervana::log_helper log(nervana::log_level::level_error, nervana::get_file_name(""), 0, "");
        EXPECT_TRUE(log.log_to_be_printed());
    }

    {
        nervana::log_helper log(
            nervana::log_level::level_warning, nervana::get_file_name(""), 0, "");
        EXPECT_FALSE(log.log_to_be_printed());
    }

    {
        nervana::log_helper log(nervana::log_level::level_info, nervana::get_file_name(""), 0, "");
        EXPECT_FALSE(log.log_to_be_printed());
    }

    pop_env();
}

namespace
{
    static string stored_var;
    static bool   env_var_set{false};

    void push_env()
    {
        const char* var = getenv(log_level_env_var);
        if (var == nullptr)
        {
            env_var_set = false;
        }
        else
        {
            env_var_set = true;
            stored_var  = var;
        }
    }

    void pop_env()
    {
        if (env_var_set)
        {
            setenv(log_level_env_var, stored_var.c_str(), 1);
        }
        else
        {
            unsetenv(log_level_env_var);
        }
    }
}
