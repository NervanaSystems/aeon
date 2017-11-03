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

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ctime>
#include <thread>
#include <mutex>
#include <map>
#include <condition_variable>
#include <cassert>

#include "log.hpp"

using namespace std;

namespace nervana
{
    class thread_starter;
}

deque<string>             nervana::logger::queue;
static mutex              queue_mutex;
static condition_variable queue_condition;
static unique_ptr<thread> queue_thread;
static bool               active = false;

class nervana::thread_starter
{
public:
    thread_starter() { nervana::logger::start(); }
    virtual ~thread_starter() { nervana::logger::stop(); }
};

static nervana::thread_starter _starter;

void nervana::logger::start()
{
    active       = true;
    queue_thread = unique_ptr<thread>(new thread(&thread_entry, nullptr));
}

void nervana::logger::stop()
{
    {
        unique_lock<std::mutex> lk(queue_mutex);
        active = false;
        queue_condition.notify_one();
    }
    queue_thread->join();
}

void nervana::logger::process_event(const string& s)
{
    cout << s << "\n";
}

void nervana::logger::thread_entry(void* param)
{
    unique_lock<std::mutex> lk(queue_mutex);
    while (active)
    {
        queue_condition.wait(lk);
        while (!queue.empty())
        {
            process_event(queue.front());
            queue.pop_front();
        }
    }
}

void nervana::logger::log_item(const string& s)
{
    unique_lock<std::mutex> lk(queue_mutex);
    queue.push_back(s);
    queue_condition.notify_one();
}

nervana::log_helper::log_helper(nervana::log_level level,
                                const char*        file,
                                int                line,
                                const char*        func)
{
    _level = level;
    switch (level)
    {
    case log_level::level_error: _stream << "[ERROR] "; break;
    case log_level::level_info: _stream << "[INFO] "; break;
    case log_level::level_warning: _stream << "[WARNING] "; break;
    case log_level::level_undefined:
        _stream << "[UNDEFINED] ";
        assert(false);
        break;
    }

    std::time_t tt = chrono::system_clock::to_time_t(chrono::system_clock::now());
    auto        tm = std::gmtime(&tt);
    char        buffer[256];
    //    strftime(buffer,sizeof(buffer), "%d/%b/%Y:%H:%M:%S %z", tm);
    //    strftime(buffer,sizeof(buffer), "%Y-%m-%d %H:%M:%S UTC", tm);
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%Sz", tm);
    _stream << buffer << " ";

    _stream << file;
    _stream << " " << line;
    //    _stream << " " << func;
    _stream << "\t";
}

nervana::log_helper::~log_helper()
{
    if (log_to_be_printed())
    {
        cout << _stream.str() << endl;
    }
}

nervana::log_level nervana::log_helper::get_log_level()
{
    const char* level_env = std::getenv(log_level_env_var);
    log_level   level;
    if (level_env == nullptr)
    {
        level = default_log_level;
    }
    else
    {
        level = from_string(level_env);
        if (level == log_level::level_undefined)
        {
            return default_log_level;
        }
    }
    return level;
}

bool nervana::log_helper::log_to_be_printed()
{
    log_level global_level = get_log_level();
    switch (global_level)
    {
    case log_level::level_info: break;
    case log_level::level_warning:
        if (_level == log_level::level_info)
            return false;
        ;
        break;
    case log_level::level_error:
        if (_level == log_level::level_info || _level == log_level::level_warning)
            return false;
        break;
    case log_level::level_undefined: assert(false);
    };
    return true;
}

nervana::log_level nervana::from_string(const string& level)
{
    static const map<string, log_level> level_map{{"INFO", log_level::level_info},
                                                  {"WARNING", log_level::level_warning},
                                                  {"ERROR", log_level::level_error}};
    auto iter = level_map.find(level);
    if (iter == level_map.end())
    {
        return log_level::level_undefined;
    }
    return iter->second;
}

string nervana::to_string(nervana::log_level level)
{
    static const map<log_level, string> level_map{{log_level::level_info, "INFO"},
                                                  {log_level::level_warning, "WARNING"},
                                                  {log_level::level_error, "ERROR"}};
    auto iter = level_map.find(level);
    if (iter == level_map.end())
    {
        return "UNDEFINED";
    }
    return iter->second;
}
