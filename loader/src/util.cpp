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
#include <cmath>
#include <cassert>
#include <iomanip>
#include "util.hpp"
#include <sox.h>
#include "log.hpp"

using namespace std;

void nervana::dump( ostream& out, const void* _data, size_t _size )
{
    const uint8_t* data = reinterpret_cast<const uint8_t*>(_data);
    int len = _size;
    int index = 0;
    while (index < len)
    {
        out << std::hex << std::setw(8) << std::setfill('0') << index;
        for (int i = 0; i < 8; i++) {
            if (index+i<len) {
                out << " " << std::hex << std::setw(2) << std::setfill('0') << (uint32_t)data[i];
            } else {
                out << "   ";
            }
        }
        cout << "  ";
        for (int i = 8; i < 16; i++) {
            if (index+i<len) {
                out << " " << std::hex << std::setw(2) << std::setfill('0') << (uint32_t)data[i];
            } else {
                out << "   ";
            }
        }
        cout << "  ";
        for (int i = 0; i < 16; i++) {
            char ch = (index+i<len ? data[i] : ' ');
            cout << ((ch < 32) ? '.' : ch);
        }
        cout << "\n";
        data += 16;
        index += 16;
    }
}

std::string nervana::to_lower(const std::string& s)
{
    std::string rc = s;
    std::transform(rc.begin(), rc.end(), rc.begin(), ::tolower);
    return rc;
}

vector<string> nervana::split(const string& src, char delimiter)
{
    size_t pos;
    string token;
    size_t start = 0;
    vector<string> rc;
    while ((pos = src.find(delimiter, start)) != std::string::npos) {
        token = src.substr(start, pos-start);
        start = pos + 1;
        rc.push_back(token);
    }
    if(start <= src.size()) {
        token = src.substr(start);
        rc.push_back(token);
    }
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

size_t nervana::unbiased_round(float x)
{
    float i;
    float fracpart = std::modf(x, &i);
    int intpart = int(i);
    int rc;

    if (std::fabs(fracpart) == 0.5) {
        if(intpart % 2 == 0) {
            rc = intpart;
        } else {

            // return nearest even integer
            rc = std::fabs(x) + 0.5;
            rc = x < 0.0 ? -rc : rc;
        }
    } else {

        // round to closest
        rc = std::floor(std::fabs(x) + 0.5);
        rc = x < 0.0 ? -rc : rc;

    }
    return rc;
}

void nervana::affirm(bool cond, const std::string& msg)
{
    if (!cond)
    {
        throw std::runtime_error(msg);
    }
}

void nervana::set_global_random_seed(uint32_t newval)
{
    nervana::global_random_seed = newval;
}

uint32_t nervana::get_global_random_seed()
{
    return nervana::global_random_seed;
}

cv::Mat nervana::read_audio_from_mem(const char* item, int itemSize)
{
    SOX_SAMPLE_LOCALS;
    sox_format_t* in = sox_open_mem_read((void *) item, itemSize, NULL, NULL, NULL);

    if (in != NULL) {
        affirm(in->signal.channels == 1, "input audio must be single channel");
        affirm(in->signal.precision == 16, "input audio must be signed short");

        sox_sample_t* sample_buffer = new sox_sample_t[in->signal.length];
        size_t number_read = sox_read(in, sample_buffer, in->signal.length);

        size_t nclipped = 0;

        cv::Mat samples_mat(in->signal.length, 1, CV_16SC1);

        affirm(in->signal.length == number_read, "unable to read all samples of input audio");

        for (uint i=0; i<in->signal.length; ++i) {
            samples_mat.at<int16_t>(i, 0) = SOX_SAMPLE_TO_SIGNED_16BIT(sample_buffer[i], nclipped);
        }
        delete [] sample_buffer;

        return samples_mat;
    } else {
        std::cout << "Unable to read";
        cv::Mat samples_mat(1, 1, CV_16SC1);
        return samples_mat;
    }
}

std::string nervana::path_join(const std::string& s1, const std::string& s2)
{
    string rc;
    if(s2.size() > 0) {
        if(s2[0] == '/') {
            rc = s2;
        } else if(s1.size() > 0) {
            rc = s1;
            if(rc[rc.size()-1] != '/') {
                rc += "/";
            }
            rc += s2;
        } else {
            rc = s2;
        }
    } else {
        rc = s1;
    }
    return rc;
}
