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

namespace nervana
{
    enum class endian
    {
        LITTLE,
        BIG
    };

#define BYTEIDX(idx, width, endianess) (endianess == endian::LITTLE ? idx : width - idx - 1)

#define DUMP_VALUE(a) cout << __FILE__ << " " << __LINE__ << " " #a " " << a << endl;

    template <typename T>
    T unpack(const char* data, int offset = 0, endian e = endian::LITTLE)
    {
        T     value = 0;
        char* v     = (char*)&value;
        for (int i = 0; i < sizeof(T); i++)
        {
            v[i] = data[offset + BYTEIDX(i, sizeof(T), e)];
        }
        return value;
    }

    template <typename T>
    void pack(char* data, T value, int offset = 0, endian e = endian::LITTLE)
    {
        char* v = (char*)&value;
        for (int i = 0; i < sizeof(T); i++)
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
                ss << sep;
            ss << x;
        }
        return ss.str();
    }

    void dump(std::ostream& out, const void*, size_t);

    std::string to_lower(const std::string& s);
    std::vector<std::string> split(const std::string& s, char delimiter);

    size_t unbiased_round(float f);
    int LevenshteinDistance(const std::string& s1, const std::string& s2);

    template <typename CharT, typename TraitsT = std::char_traits<CharT>>
    class memstream : public std::basic_streambuf<CharT, TraitsT>
    {
    public:
        memstream(CharT* data, size_t size) { this->setg(data, data, data + size); }
        std::ios::pos_type seekoff(std::ios::off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override
        {
            switch (dir)
            {
            case std::ios_base::beg: this->setg(this->eback(), this->eback() + off, this->egptr()); break;
            case std::ios_base::cur: this->setg(this->eback(), this->gptr() + off, this->egptr()); break;
            case std::ios_base::end: this->setg(this->eback(), this->egptr() - off, this->egptr()); break;
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

    static uint32_t global_random_seed = 0;

    void set_global_random_seed(uint32_t newval);
    uint32_t get_global_random_seed();
    cv::Mat read_audio_from_mem(const char* item, int itemSize);

    class async
    {
    public:
        async()
            : thread{nullptr}
            , ready{false}
        {
        }

        ~async()
        {
            if (thread)
            {
                thread->detach();
                delete thread;
            }
            thread = nullptr;
        }

        void run(std::function<void(void*)> f, void* param = nullptr)
        {
            func  = f;
            ready = false;
            if (thread)
            {
                thread->detach();
                delete thread;
            }
            thread = new std::thread(&async::entry, this, param);
        }

        void wait()
        {
            if (thread)
            {
                thread->join();
                delete thread;
                thread = nullptr;
            }
        }
        bool is_ready() { return ready; }
        bool is_busy() { return thread != nullptr; }
        void rethrow_exception() { std::rethrow_exception(stored_exception); }
    private:
        async(const async&) = delete;

        void entry(void* param)
        {
            try
            {
                func(param);
            }
            catch (std::exception e)
            {
                stored_exception = std::current_exception();
            }
            ready = true;
        }

        std::function<void(void*)> func;
        std::thread*               thread;
        bool                       ready;
        std::exception_ptr         stored_exception;
    };
}
