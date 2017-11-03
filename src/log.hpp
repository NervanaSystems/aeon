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

#pragma once

#include <sstream>
#include <stdexcept>
#include <deque>

namespace nervana
{
    enum class log_level
    {
        level_error,
        level_warning,
        level_info,
        level_undefined,
    };

    static const char*              log_level_env_var = "AEON_LOG_LEVEL";
    static const nervana::log_level default_log_level = nervana::log_level::level_warning;

    class conststring
    {
    public:
        template <size_t SIZE>
        constexpr conststring(const char (&p)[SIZE])
            : _string(p)
            , _size(SIZE)
        {
        }

        constexpr char operator[](size_t i) const
        {
            return i < _size ? _string[i] : throw std::out_of_range("");
        }
        constexpr const char* get_ptr(size_t offset) const { return &_string[offset]; }
        constexpr size_t                     size() const { return _size; }
    private:
        const char* _string;
        size_t      _size;
    };

    constexpr const char* find_last(conststring s, size_t offset, char ch)
    {
        return offset == 0 ? s.get_ptr(0) : (s[offset] == ch ? s.get_ptr(offset + 1)
                                                             : find_last(s, offset - 1, ch));
    }

    constexpr const char* find_last(conststring s, char ch)
    {
        return find_last(s, s.size() - 1, ch);
    }
    constexpr const char* get_file_name(conststring s) { return find_last(s, '/'); }
    log_level from_string(const std::string& level);
    std::string to_string(log_level);

    class log_helper
    {
    public:
        log_helper(log_level, const char* file, int line, const char* func);
        ~log_helper();

        std::ostream& stream() { return _stream; }
    private:
        log_level         get_log_level();
        bool              log_to_be_printed();
        std::stringstream _stream;
        log_level         _level;
    };

    class logger
    {
        friend class log_helper;

    public:
        static void set_log_path(const std::string& path);
        static void start();
        static void stop();

    private:
        static void log_item(const std::string& s);
        static void process_event(const std::string& s);
        static void thread_entry(void* param);
        static std::deque<std::string> queue;
    };

#define ERR                                                                                        \
    nervana::log_helper(nervana::log_level::level_error,                                           \
                        nervana::get_file_name(__FILE__),                                          \
                        __LINE__,                                                                  \
                        __PRETTY_FUNCTION__)                                                       \
        .stream()
#define WARN                                                                                       \
    nervana::log_helper(nervana::log_level::level_warning,                                         \
                        nervana::get_file_name(__FILE__),                                          \
                        __LINE__,                                                                  \
                        __PRETTY_FUNCTION__)                                                       \
        .stream()
#define INFO                                                                                       \
    nervana::log_helper(nervana::log_level::level_info,                                            \
                        nervana::get_file_name(__FILE__),                                          \
                        __LINE__,                                                                  \
                        __PRETTY_FUNCTION__)                                                       \
        .stream()
}
