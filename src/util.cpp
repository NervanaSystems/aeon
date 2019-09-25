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

#include <algorithm>
#include <string>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <sox.h>

#include "util.hpp"
#include "log.hpp"

using namespace std;

// we use int as opencv uses it in its interface
static_assert(sizeof(int) == 4, "int size is not 4 bytes");

map<string, nervana::stopwatch*> nervana::stopwatch_statistics;

static string multibyte_conversion_error_message =
    "multibyte to wide characters conversion error (it's possible that locale LC_CTYPE environment "
    "variable needs to be set to some UTF-8 variant)";

void nervana::dump(ostream& out, const void* _data, size_t _size)
{
    auto           flags = out.flags();
    const uint8_t* data  = reinterpret_cast<const uint8_t*>(_data);
    int            len   = _size;
    int            index = 0;
    while (index < len)
    {
        out << std::hex << std::setw(8) << std::setfill('0') << index;
        for (int i = 0; i < 8; i++)
        {
            if (index + i < len)
            {
                out << " " << std::hex << std::setw(2) << std::setfill('0') << (uint32_t)data[i];
            }
            else
            {
                out << "   ";
            }
        }
        out << "  ";
        for (int i = 8; i < 16; i++)
        {
            if (index + i < len)
            {
                out << " " << std::hex << std::setw(2) << std::setfill('0') << (uint32_t)data[i];
            }
            else
            {
                out << "   ";
            }
        }
        out << "  ";
        for (int i = 0; i < 16; i++)
        {
            char ch = (index + i < len ? data[i] : ' ');
            out << ((ch < 32) ? '.' : ch);
        }
        out << "\n";
        data += 16;
        index += 16;
    }
    out.flags(flags);
}

std::string nervana::to_lower(const std::string& s)
{
    std::string rc = s;
    std::transform(rc.begin(), rc.end(), rc.begin(), ::tolower);
    return rc;
}

string nervana::trim(const string& s)
{
    string rc = s;
    // trim trailing spaces
    size_t pos = rc.find_last_not_of(" \t");
    if (string::npos != pos)
    {
        rc = rc.substr(0, pos + 1);
    }

    // trim leading spaces
    pos = rc.find_first_not_of(" \t");
    if (string::npos != pos)
    {
        rc = rc.substr(pos);
    }
    return rc;
}

vector<string> nervana::split(const string& src, char delimiter, bool do_trim)
{
    size_t         pos;
    string         token;
    size_t         start = 0;
    vector<string> rc;
    while ((pos = src.find(delimiter, start)) != std::string::npos)
    {
        token = src.substr(start, pos - start);
        start = pos + 1;
        if (do_trim)
        {
            token = trim(token);
        }
        rc.push_back(token);
    }
    if (start <= src.size())
    {
        token = src.substr(start);
        if (do_trim)
        {
            token = trim(token);
        }
        rc.push_back(token);
    }
    return rc;
}

std::wstring nervana::to_wstring(const string& s, size_t max_size)
{
    max_size = std::min(s.size(), max_size);
    std::shared_ptr<wchar_t> buffer(new wchar_t[max_size], std::default_delete<wchar_t[]>());
    size_t                   size = std::mbstowcs(buffer.get(), s.c_str(), max_size);
    if (static_cast<int>(size) == -1)
    {
        throw runtime_error(multibyte_conversion_error_message);
    }

    return wstring(buffer.get(), size);
}

size_t nervana::wstring_length(const string& s)
{
    size_t size = std::mbstowcs(NULL, s.c_str(), s.size());
    if (static_cast<int>(size) == -1)
    {
        throw runtime_error(multibyte_conversion_error_message);
    }
    return size;
}

int nervana::LevenshteinDistance(const string& s, const string& t)
{
    // degenerate cases
    if (s == t)
    {
        return 0;
    }
    if (s.size() == 0)
    {
        return t.size();
    }
    if (t.size() == 0)
    {
        return s.size();
    }

    // create two work vectors of integer distances
    vector<int> v0(t.size() + 1);
    vector<int> v1(t.size() + 1);

    // initialize v0 (the previous row of distances)
    // this row is A[0][i]: edit distance for an empty s
    // the distance is just the number of characters to delete from t
    for (int i = 0; i < v0.size(); i++)
    {
        v0[i] = i;
    }

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
        {
            v0[j] = v1[j];
        }
    }

    return v1[t.size()];
}

size_t nervana::unbiased_round(float x)
{
    float i;
    float fracpart = std::modf(x, &i);
    int   intpart  = int(i);
    int   rc;

    if (std::fabs(fracpart) == 0.5)
    {
        if (intpart % 2 == 0)
        {
            rc = intpart;
        }
        else
        {
            // return nearest even integer
            rc = std::fabs(x) + 0.5;
            rc = x < 0.0 ? -rc : rc;
        }
    }
    else
    {
        // round to closest
        rc = std::floor(std::fabs(x) + 0.5);
        rc = x < 0.0 ? -rc : rc;
    }
    return rc;
}

bool nervana::almost_equal(float a, float b)
{
    return fabs(a - b) < epsilon;
};

bool nervana::almost_equal_or_less(float a, float b)
{
    return a <= b + epsilon;
};

bool nervana::almost_equal_or_greater(float a, float b)
{
    return a >= b - epsilon;
};

void nervana::affirm(bool cond, const std::string& msg)
{
    if (!cond)
    {
        throw std::runtime_error(msg);
    }
}

namespace nervana
{
    thread_local static random_engine_t local_random_engine{random_device{}()};
}
nervana::random_engine_t& nervana::get_thread_local_random_engine()
{
    return local_random_engine;
}

cv::Mat nervana::read_audio_from_mem(const char* item, int itemSize)
{
    SOX_SAMPLE_LOCALS;
    sox_format_t* in = sox_open_mem_read((void*)item, itemSize, NULL, NULL, NULL);

    if (in != NULL)
    {
        affirm(in->signal.channels == 1, "input audio must be single channel");
        affirm(in->signal.precision == 16, "input audio must be signed short");

        sox_sample_t* sample_buffer = new sox_sample_t[in->signal.length];
        size_t        number_read   = sox_read(in, sample_buffer, in->signal.length);

        size_t nclipped = 0;

        cv::Mat samples_mat(in->signal.length, 1, CV_16SC1);

        affirm(in->signal.length == number_read, "unable to read all samples of input audio");

        for (uint i = 0; i < in->signal.length; ++i)
        {
            samples_mat.at<int16_t>(i, 0) = SOX_SAMPLE_TO_SIGNED_16BIT(sample_buffer[i], nclipped);
        }
        delete[] sample_buffer;
        sox_close(in);
        return samples_mat;
    }
    else
    {
        std::cout << "Unable to read";
        cv::Mat samples_mat(1, 1, CV_16SC1);
        sox_close(in);
        return samples_mat;
    }
}

// Audio buffer must be int16
void nervana::write_audio_to_file(cv::Mat buffer, std::string path, sox_rate_t sample_rate_hz)
{
    sox_signalinfo_t signal{sample_rate_hz, 1, 16, buffer.total(), NULL};
    auto             t = sox_open_write(path.c_str(), &signal, NULL, NULL, NULL, NULL);

    sox_sample_t sample_buffer[buffer.total()];
    for (auto i = 0; i < buffer.total(); i++)
    {
        sample_buffer[i] = SOX_SIGNED_16BIT_TO_SAMPLE(buffer.at<int16_t>(i), 0);
    }
    sox_write(t, sample_buffer, buffer.total());
}

std::vector<char> nervana::string2vector(const std::string& s)
{
    return vector<char>{s.begin(), s.end()};
}

std::string nervana::vector2string(const std::vector<char>& v)
{
    return string{v.data(), v.size()};
}

// Parses cpu_list string. For example: "0-4,30,10,32-33"
std::vector<int> nervana::parse_cpu_list(const std::string& cpu_list)
{
    std::vector<int>         thread_affinity_map;
    std::vector<std::string> split_list = nervana::split(cpu_list, ',', false);
    try
    {
        for (std::string& str : split_list)
        {
            auto dash_pos = str.find('-');
            if (dash_pos == std::string::npos)
            {
                thread_affinity_map.push_back(stoi(str));
            }
            else
            {
                int from = stoi(str.substr(0, dash_pos));
                int to   = stoi(str.substr(dash_pos + 1));
                for (int i = from; i <= to; ++i)
                {
                    thread_affinity_map.push_back(i);
                }
            }
        }
    }
    catch (...)
    {
        ERR << "Failed to parse cpu list.";
        throw;
    }
    std::sort(thread_affinity_map.begin(), thread_affinity_map.end());
    auto ip = std::unique(thread_affinity_map.begin(), thread_affinity_map.end());
    if (ip != thread_affinity_map.end())
    {
        auto num_removed_duplicates = std::distance(ip, thread_affinity_map.end());
        WARN << "Removed " << std::to_string(num_removed_duplicates)
             << " duplicate entries in cpu list.";
    }
    thread_affinity_map.resize(std::distance(thread_affinity_map.begin(), ip));

    if (thread_affinity_map.back() >= std::thread::hardware_concurrency())
    {
        throw std::invalid_argument("One or more indexes computed from cpu list '" + cpu_list +
                                    "' exceed number of logical cores. Use values "
                                    "in "
                                    "range [0, " +
                                    std::to_string(std::thread::hardware_concurrency() - 1) + "].");
    }

    return thread_affinity_map;
}

// Parses cpu affinity from loader config or environment variable.
// Environment has precedence over config. If both are empty, provides the default.
std::vector<int> nervana::get_thread_affinity_map(const std::string& config_cpu_list,
                                                  int                max_count_of_free_threads,
                                                  int                free_threads_ratio)
{
    std::vector<int> thread_affinity_map;

    const char* env_cpu_list = getenv("AEON_CPU_LIST");
    if (env_cpu_list != nullptr && strlen(env_cpu_list) > 0) // set affinity from environment
    {
        std::string str_cpu_list{env_cpu_list};
        thread_affinity_map = nervana::parse_cpu_list(str_cpu_list);

        if (!config_cpu_list.empty())
        {
            WARN << "Both AEON_CPU_LIST and 'cpu_list' are detected. AEON_CPU_LIST environment "
                    "variable will be used in this case.";
        }
    }
    else if (!config_cpu_list.empty()) // set affinity from config
    {
        thread_affinity_map = nervana::parse_cpu_list(config_cpu_list);
    }

    if (thread_affinity_map.empty()) //  automatically determine thread_affinity_map
    {
        // we don't use all threads, some of them we leave for other pipeline objects and system
        int nthreads =
            std::thread::hardware_concurrency() -
            std::min(max_count_of_free_threads,
                     static_cast<int>(std::thread::hardware_concurrency() / free_threads_ratio));

        thread_affinity_map.resize(nthreads);
        std::iota(thread_affinity_map.begin(), thread_affinity_map.end(), 0);
    }

    return thread_affinity_map;
}
