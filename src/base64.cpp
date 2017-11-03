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

#include <sstream>
#include <iomanip>

#include "base64.hpp"

using namespace std;

const uint8_t nervana::base64::character_codes[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
const uint8_t nervana::base64::decode_codes[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00, 0x3f,
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
    0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x00, 0x00, 0x00, 0x00, 0x00};

vector<char> nervana::base64::encode(const vector<char>& data)
{
    return encode(data.data(), data.size());
}
vector<char> nervana::base64::encode(const char* data, size_t size)
{
    vector<char> rc;

    for (int i = 0; i < size; i += 3)
    {
        const uint8_t* p = (uint8_t*)&data[i];
        rc.push_back(character_codes[0x3F & (p[0] >> 2)]);
        rc.push_back(character_codes[0x3F & (p[0] << 4 | p[1] >> 4)]);
        if (i + 1 == size)
        {
            break;
        }
        rc.push_back(character_codes[0x3F & (p[1] << 2 | p[2] >> 6)]);
        if (i + 2 == size)
        {
            break;
        }
        rc.push_back(character_codes[0x3F & p[2]]);
    }

    while (rc.size() % 4 != 0)
    {
        rc.push_back('=');
    }

    return rc;
}

vector<char> nervana::base64::decode(const vector<char>& data)
{
    return decode(data.data(), data.size());
}
vector<char> nervana::base64::decode(const char* data, size_t size)
{
    vector<char> rc;

    for (int i = 0; i < size; i += 4)
    {
        const char* p = &data[i];
        rc.push_back(0xFF & (decode_codes[(size_t)p[0]] << 2 | decode_codes[(size_t)p[1]] >> 4));
        if (i + 1 == size || p[2] == '=')
        {
            break;
        }
        rc.push_back(0xFF & (decode_codes[(size_t)p[1]] << 4 | decode_codes[(size_t)p[2]] >> 2));
        if (i + 2 == size || p[3] == '=')
        {
            break;
        }
        rc.push_back(0xFF & (decode_codes[(size_t)p[2]] << 6 | decode_codes[(size_t)p[3]]));
    }

    return rc;
}

string nervana::base64::gen_decode_table()
{
    string       codes = (const char*)character_codes;
    stringstream ss;
    for (int i = 0; i < 128; ++i)
    {
        int index = codes.find((char)i);
        if (i > 0)
        {
            ss << ", ";
        }
        if (index == string::npos)
        {
            ss << "0x00";
        }
        else
        {
            ss << "0x" << hex << setw(2) << setfill('0') << index << dec;
        }
    }
    return ss.str();
}
