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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gtest/gtest.h"
#include "base64.hpp"
#include "gen_image.hpp"

using namespace std;
using namespace nervana;

static string plain_text =
    "Man is distinguished, not only by his reason, but by this singular passion from "
    "other animals, which is a lust of the mind, that by a perseverance of delight "
    "in the continued and indefatigable generation of knowledge, exceeds the short "
    "vehemence of any carnal pleasure.";

static string encoded_text =
    "TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlz"
    "IHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2Yg"
    "dGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGlu"
    "dWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRo"
    "ZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=";

TEST(base64, encode)
{
    //    string s = base64::gen_decode_table();
    //    INFO << s;

    vector<char> encoded = base64::encode(plain_text.data(), plain_text.size());
    string       actual{encoded.data(), encoded.size()};
    EXPECT_STREQ(encoded_text.c_str(), actual.c_str());
}

TEST(base64, decode)
{
    vector<char> decoded = base64::decode(encoded_text.data(), encoded_text.size());
    string       actual{decoded.data(), decoded.size()};
    EXPECT_STREQ(plain_text.c_str(), actual.c_str());
}

TEST(base64, padding_encode)
{
    vector<pair<string, string>> padding = {
        {"any carnal pleasure.", "YW55IGNhcm5hbCBwbGVhc3VyZS4="},
        {"any carnal pleasure", "YW55IGNhcm5hbCBwbGVhc3VyZQ=="},
        {"any carnal pleasur", "YW55IGNhcm5hbCBwbGVhc3Vy"},
        {"any carnal pleasu", "YW55IGNhcm5hbCBwbGVhc3U="},
        {"any carnal pleas", "YW55IGNhcm5hbCBwbGVhcw=="}};

    for (auto p : padding)
    {
        string       decoded  = p.first;
        string       expected = p.second;
        vector<char> encoded  = base64::encode(decoded.data(), decoded.size());
        string       actual{encoded.data(), encoded.size()};
        EXPECT_STREQ(expected.c_str(), actual.c_str());
    }
}

TEST(base64, padding_decode)
{
    vector<pair<string, string>> padding = {
        {"any carnal pleasure.", "YW55IGNhcm5hbCBwbGVhc3VyZS4="},
        {"any carnal pleasure.", "YW55IGNhcm5hbCBwbGVhc3VyZS4"},
        {"any carnal pleasure", "YW55IGNhcm5hbCBwbGVhc3VyZQ=="},
        {"any carnal pleasure", "YW55IGNhcm5hbCBwbGVhc3VyZQ="},
        {"any carnal pleasure", "YW55IGNhcm5hbCBwbGVhc3VyZQ"},
        {"any carnal pleasur", "YW55IGNhcm5hbCBwbGVhc3Vy"},
        {"any carnal pleasu", "YW55IGNhcm5hbCBwbGVhc3U="},
        {"any carnal pleasu", "YW55IGNhcm5hbCBwbGVhc3U"},
        {"any carnal pleas", "YW55IGNhcm5hbCBwbGVhcw=="},
        {"any carnal pleas", "YW55IGNhcm5hbCBwbGVhcw="},
        {"any carnal pleas", "YW55IGNhcm5hbCBwbGVhcw"}};

    for (auto p : padding)
    {
        string       expected = p.first;
        string       encoded  = p.second;
        vector<char> decoded  = base64::decode(encoded.data(), encoded.size());
        string       actual{decoded.data(), decoded.size()};
        EXPECT_STREQ(expected.c_str(), actual.c_str());
    }
}

TEST(base64, binary)
{
    vector<char> source;
    for (size_t i = 0; i < 256; i++)
    {
        source.push_back(i);
    }

    vector<char> encoded = base64::encode(source);
    vector<char> decoded = base64::decode(encoded);
    for (size_t i = 0; i < 256; i++)
    {
        ASSERT_EQ((uint8_t)source[i], (uint8_t)decoded[i]) << i;
    }
}
