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

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <random>
#include <opencv2/core/core.hpp>
#include <sox.h>
#include <thread>
#include <chrono>
#include <map>
#include <mutex>

namespace nervana
{
    const float epsilon = 0.00001;

    class stopwatch;
    extern std::map<std::string, stopwatch*> stopwatch_statistics;

    enum class endian
    {
        LITTLE,
        BIG
    };

#define BYTEIDX(idx, width, endianess) (endianess == endian::LITTLE ? idx : width - idx - 1)

#define DUMP_VALUE(a) cout << __FILE__ << " " << __LINE__ << " " #a " " << a << endl;

    template <class T>
    class singleton
    {
    public:
        singleton()                 = delete;
        singleton(const singleton&) = delete;
        singleton& operator=(const singleton&) = delete;

        template <typename... Args>
        static std::shared_ptr<T> get(Args... args)
        {
            std::lock_guard<std::mutex> lg(m_mutex);
            std::shared_ptr<T>          instance = m_singleton.lock();
            if (!instance)
            {
                instance.reset(new T(args...));
                m_singleton = instance;
            }
            return instance;
        }

    private:
        static std::weak_ptr<T> m_singleton;
        static std::mutex       m_mutex;
    };

    template <class T>
    std::weak_ptr<T> singleton<T>::m_singleton;
    template <class T>
    std::mutex singleton<T>::m_mutex;

    template <typename T>
    T unpack(const void* _data, size_t offset = 0, endian e = endian::LITTLE)
    {
        const char* data  = (const char*)_data;
        T           value = 0;
        char*       v     = (char*)&value;
        for (size_t i = 0; i < sizeof(T); i++)
        {
            v[i] = data[offset + BYTEIDX(i, sizeof(T), e)];
        }
        return value;
    }

    template <typename T>
    void pack(void* _data, T value, size_t offset = 0, endian e = endian::LITTLE)
    {
        char* data = (char*)_data;
        char* v    = (char*)&value;
        for (size_t i = 0; i < sizeof(T); i++)
        {
            data[offset + i] = v[BYTEIDX(i, sizeof(T), e)];
        }
    }
    template <typename T>
    std::string join(const T& v, const std::string& sep)
    {
        std::ostringstream ss;
        for (const auto& x : v)
        {
            if (&x != &v[0])
            {
                ss << sep;
            }
            ss << x;
        }
        return ss.str();
    }

    void dump(std::ostream& out, const void*, size_t);

    std::string to_lower(const std::string& s);
    std::string trim(const std::string& s);
    std::vector<std::string> split(const std::string& s, char delimiter, bool trim = false);

    std::wstring to_wstring(const std::string& s, size_t max_size = SIZE_MAX);
    size_t wstring_length(const std::string& s);

    size_t unbiased_round(float f);
    bool almost_equal(float a, float b);
    bool almost_equal_or_less(float a, float b);
    bool almost_equal_or_greater(float a, float b);
    int LevenshteinDistance(const std::string& s1, const std::string& s2);

    template <typename CharT, typename TraitsT = std::char_traits<CharT>>
    class memstream : public std::basic_streambuf<CharT, TraitsT>
    {
    public:
        memstream(CharT* data, size_t size) { this->setg(data, data, data + size); }
        std::ios::pos_type seekoff(std::ios::off_type      off,
                                   std::ios_base::seekdir  dir,
                                   std::ios_base::openmode which) override
        {
            switch (dir)
            {
            case std::ios_base::beg:
                this->setg(this->eback(), this->eback() + off, this->egptr());
                break;
            case std::ios_base::cur:
                this->setg(this->eback(), this->gptr() + off, this->egptr());
                break;
            case std::ios_base::end:
                this->setg(this->eback(), this->egptr() - off, this->egptr());
                break;
            default: break;
            }
            return this->gptr() - this->eback();
        }
        std::ios::pos_type seekpos(std::ios::pos_type pos, std::ios_base::openmode which) override
        {
            this->setg(this->eback(), this->eback() + pos, this->egptr());
            return this->gptr() - this->eback();
        }
    };

    class memory_stream : public std::istream
    {
    public:
        memory_stream(char* data, size_t size)
            : std::istream{&wrapper}
            , wrapper{data, size}
        {
        }

    private:
        memstream<char> wrapper;
    };

    void affirm(bool cond, const std::string& msg);

    typedef std::minstd_rand0 random_engine_t;
    random_engine_t&          get_thread_local_random_engine();

    cv::Mat read_audio_from_mem(const char* item, int itemSize);

    std::vector<char> string2vector(const std::string& s);
    std::string vector2string(const std::vector<char>& s);

    class stopwatch
    {
    public:
        stopwatch() {}
        stopwatch(const std::string& name)
            : m_name{name}
        {
            stopwatch_statistics.insert({m_name, this});
        }

        ~stopwatch()
        {
            if (m_name.size() > 0)
            {
                stopwatch_statistics.find(m_name);
            }
        }

        void start()
        {
            if (m_active == false)
            {
                m_total_count++;
                m_active     = true;
                m_start_time = m_clock.now();
            }
        }

        void stop()
        {
            if (m_active == true)
            {
                auto end_time = m_clock.now();
                m_last_time   = end_time - m_start_time;
                m_total_time += m_last_time;
                m_active = false;
            }
        }

        size_t get_call_count() const { return m_total_count; }
        size_t get_seconds() const { return get_nanoseconds() / 1e9; }
        size_t get_milliseconds() const { return get_nanoseconds() / 1e6; }
        size_t get_microseconds() const { return get_nanoseconds() / 1e3; }
        size_t get_nanoseconds() const
        {
            if (m_active)
            {
                return (m_clock.now() - m_start_time).count();
            }
            else
            {
                return m_last_time.count();
            }
        }

        size_t get_total_seconds() const { return get_total_nanoseconds() / 1e9; }
        size_t get_total_milliseconds() const { return get_total_nanoseconds() / 1e6; }
        size_t get_total_microseconds() const { return get_total_nanoseconds() / 1e3; }
        size_t get_total_nanoseconds() const { return m_total_time.count(); }
    private:
        std::chrono::high_resolution_clock                          m_clock;
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time;
        bool                                                        m_active = false;
        std::chrono::nanoseconds                                    m_total_time =
            std::chrono::high_resolution_clock::duration::zero();
        std::chrono::nanoseconds m_last_time;
        size_t                   m_total_count = 0;
        std::string              m_name;
    };
}
