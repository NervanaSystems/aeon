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
#include <string>

#include "util.hpp"

using namespace std;

void nervana::dump( const void* _data, size_t _size ) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(_data);
    int len = _size;
    assert(len % 16 == 0);
    int index = 0;
    while (index < len) {
        printf("%08x", index);
        for (int i = 0; i < 8; i++) {
            printf(" %02x", data[i]);
        }
        printf("  ");
        for (int i = 8; i < 16; i++) {
            printf(" %02x", data[i]);
        }
        printf(" ");
        for (int i = 0; i < 16; i++) {
            printf("%c", (data[i] < 32)? '.' : data[i]);
        }
        printf("\n");
        data += 16;
        index += 16;
    }
}

std::string nervana::tolower(const std::string& s) {
    std::string rc = s;
    std::transform(rc.begin(), rc.end(), rc.begin(), ::tolower);
    return rc;
}

vector<string> nervana::split(const string& src, const string& delimiter)
{
    size_t pos = 0;
    string s = src;
    string token;
    vector<string> rc;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        rc.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    if(s.size() > 0) rc.push_back(s);
    return rc;
}

int nervana::LevenshteinDistance(const string& s, const string& t)
{
    // degenerate cases
    if (s == t) return 0;
    if (s.size() == 0) return t.size();
    if (t.size() == 0) return s.size();

    // create two work vectors of integer distances
    vector<int> v0(t.size() + 1);
    vector<int> v1(t.size() + 1);

    // initialize v0 (the previous row of distances)
    // this row is A[0][i]: edit distance for an empty s
    // the distance is just the number of characters to delete from t
    for (int i = 0; i < v0.size(); i++)
        v0[i] = i;

    for (int i = 0; i < s.size(); i++)
    {
        // calculate v1 (current row distances) from the previous row v0

        // first element of v1 is A[i+1][0]
        //   edit distance is delete (i+1) chars from s to match empty t
        v1[0] = i + 1;

        // use formula to fill in the rest of the row
        for (int j = 0; j < t.size(); j++)
        {
            auto cost = (s[i] == t[j]) ? 0 : 1;
            v1[j + 1] = std::min(std::min(v1[j] + 1, v0[j + 1] + 1), v0[j] + cost);
        }

        // copy v1 (current row) to v0 (previous row) for next iteration
        for (int j = 0; j < v0.size(); j++)
            v0[j] = v1[j];
    }

    return v1[t.size()];
}
