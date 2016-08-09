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

#include <random>

#include "gtest/gtest.h"
#include "block_loader_cpio_cache.hpp"

using namespace std;


string load_string(block_loader_cpio_cache cache) {
    // call loadBlock from cache and cast the resulting item to a uint
    buffer_in_array bp(2);  // 2 buffer_in:  1 for datum, 1 for target

    cache.loadBlock(bp, 1);

    vector<char>& x = bp[0]->getItem(0);
    string str(x.data(), x.size());
    return str;
}

block_loader_cpio_cache make_cache(const string& rootCacheDir,
                                   const string& hash,
                                   const string& version) {
    block_loader_cpio_cache cache(
        rootCacheDir, hash, version, make_shared<block_loader_random>(1)
    );
    return cache;
}

TEST(block_loader_cpio_cache, integration) {
    // load the same block twice and make sure it has the same value.
    // block_loader_random always returns a different uint value no matter
    // the block_num.  The only way two consequetive calls are the same
    // is if the cache is working properly

    auto cache = make_cache("/tmp", block_loader_random::randomString(), "version123");

    ASSERT_EQ(load_string(cache), load_string(cache));
}

TEST(block_loader_cpio_cache, same_version) {
    string hash = block_loader_random::randomString();
    ASSERT_EQ(
        load_string(make_cache("/tmp", hash, "version123")),
        load_string(make_cache("/tmp", hash, "version123"))
    );
}

TEST(block_loader_cpio_cache, differnt_version) {
    string hash = block_loader_random::randomString();
    ASSERT_NE(
        load_string(make_cache("/tmp", hash, "version123")),
        load_string(make_cache("/tmp", hash, "version456"))
    );
}

TEST(block_loader_cpio_cache, differnt_hash) {
    ASSERT_NE(
        load_string(make_cache("/tmp", block_loader_random::randomString(), "version123")),
        load_string(make_cache("/tmp", block_loader_random::randomString(), "version123"))
    );
}
