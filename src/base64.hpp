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

#include <vector>
#include <string>
#include <cstdlib>

namespace nervana
{
    class base64;
}

class nervana::base64
{
public:
    static std::vector<char> encode(const std::vector<char>& data);
    static std::vector<char> encode(const char* data, size_t size);

    static std::vector<char> decode(const std::vector<char>& data);
    static std::vector<char> decode(const char* data, size_t size);

    static std::string gen_decode_table();

private:
    static const uint8_t character_codes[];
    static const uint8_t decode_codes[];
};
