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

#include <algorithm>
#include <fstream>
#include <dirent.h>

#include "helpers.hpp"
#include "gtest/gtest.h"
#include "log.hpp"

using namespace std;
using namespace nervana;

vector<string> buffer_to_vector_of_strings(encoded_record_list& b)
{
    vector<string> words;

    if (b.size() > 0)
    {
        for (auto i = 0; i != b.size(); ++i)
        {
            vector<char>& s = b.record(i).element(0);
            words.push_back(string(s.data(), s.size()));
        }
    }

    return words;
}

bool sorted(vector<string> words)
{
    return std::is_sorted(words.begin(), words.end());
}

void dump_vector_of_strings(vector<string>& words)
{
    for (auto word = words.begin(); word != words.end(); ++word)
    {
        cout << *word << endl;
    }
}

void assert_vector_unique(vector<string>& words)
{
    sort(words.begin(), words.end());
    for (auto word = words.begin(); word != words.end() - 1; ++word)
    {
        ASSERT_NE(*word, *(word + 1));
    }
}
