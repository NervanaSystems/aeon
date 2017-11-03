/*
 Copyright 2016 Intel(R) Nervana(TM)
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

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "async_manager.hpp"

using namespace std;
using namespace nervana;

class data_source : public async_manager_source<int>
{
public:
    data_source(int count, int delay_ms)
        : m_data{count}
        , m_data_index{0}
        , m_delay_ms{delay_ms}
    {
        m_data.resize(count);
        iota(m_data.begin(), m_data.end(), 0);
    }

    int* next() override
    {
        int* rc = nullptr;
        if (m_data_index != m_data.size())
        {
            rc = &m_data[m_data_index++];
            usleep(m_delay_ms * 1000);
        }
        return rc;
    }
    size_t record_count() const override { return m_data.size(); }
    size_t elements_per_record() const override { return 1; }
    void   reset() override { m_data_index = 0; }
private:
    vector<int> m_data;
    int         m_data_index;
    int         m_delay_ms;
};

typedef array<int, 2> minibatch;

class integer_batcher : public async_manager<int, minibatch>
{
public:
    integer_batcher(shared_ptr<data_source> d)
        : async_manager<int, minibatch>(d, "test")
    {
    }

    virtual minibatch* filler() override
    {
        minibatch* rc             = nullptr;
        int        number_fetched = 0;
        minibatch* output         = get_pending_buffer();

        for (int i = 0; i < 2; i++)
        {
            const int* value_ptr = m_source->next();
            if (value_ptr)
            {
                (*output)[i] = *value_ptr;
                ++number_fetched;
            }
            else
            {
                break;
            }
        }
        if (number_fetched == 2)
        {
            rc = output;
        }
        return rc;
    }

    size_t record_count() const override { return 100; }
    size_t elements_per_record() const override { return 1; }
};

TEST(async_manager, source)
{
    data_source datagen(5, 0);

    EXPECT_EQ(0, *datagen.next());
    EXPECT_EQ(1, *datagen.next());
    EXPECT_EQ(2, *datagen.next());
    EXPECT_EQ(3, *datagen.next());
    EXPECT_EQ(4, *datagen.next());
    EXPECT_EQ(nullptr, datagen.next());
    EXPECT_EQ(nullptr, datagen.next());
}
