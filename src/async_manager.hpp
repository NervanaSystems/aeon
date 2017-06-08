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

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <utility>
#include <algorithm>
#include <functional>
#include <future>
#include <map>
#include "log.hpp"

namespace nervana
{
    template <typename OUTPUT>
    class async_manager_source;
    template <typename INPUT, typename OUTPUT>
    class async_manager;
    class async_manager_info;

    enum class async_state
    {
        idle,
        wait_for_buffer,
        fetching_data,
        processing
    };
    extern std::vector<async_manager_info*> async_manager_status;
}

class nervana::async_manager_info
{
public:
    virtual ~async_manager_info() {}
    virtual async_state        get_state() const = 0;
    virtual const std::string& get_name() const  = 0;
};

template <typename OUTPUT>
class nervana::async_manager_source
{
public:
    async_manager_source() {}
    virtual ~async_manager_source() {}
    virtual OUTPUT* next()                      = 0;
    virtual size_t  record_count() const        = 0;
    virtual size_t  elements_per_record() const = 0;
    virtual void    reset()                     = 0;

    async_manager_source(const async_manager_source&) = default;
};

template <typename INPUT, typename OUTPUT>
class nervana::async_manager : public virtual nervana::async_manager_source<OUTPUT>,
                               public async_manager_info
{
public:
    async_manager(async_manager_source<INPUT>* source, const std::string& name)
        : m_source(source)
        , m_state{async_state::idle}
        , m_name{name}
    {
        // Make the container pair?  Currently letting child handle it in filler()
        async_manager_status.push_back(this);
    }

    OUTPUT* next() override
    {
        // Special case for first time through
        OUTPUT* result = nullptr;
        if (m_first)
        {
            m_first = false;
            // Just run this one in blocking mode
            m_pending_result = std::async(
                std::launch::async, &nervana::async_manager<INPUT, OUTPUT>::filler, this);
        }
        result = m_pending_result.get();
        if (result != nullptr)
        {
            swap();

            // Now kick off this one in async
            m_pending_result = std::async(
                std::launch::async, &nervana::async_manager<INPUT, OUTPUT>::filler, this);
        }
        return result;
    }

    // do the work to fill up m_containers
    virtual OUTPUT* filler() = 0;

    virtual void reset() override
    {
        finalize();
        m_source->reset();
        initialize();
    }

    virtual ~async_manager() { finalize(); }
    virtual void initialize() { m_first = true; }
    void         finalize()
    {
        if (m_pending_result.valid())
        {
            m_pending_result.get();
        }
    }

    async_state        get_state() const override { return m_state; }
    const std::string& get_name() const override { return m_name; }
protected:
    async_manager(const async_manager&) = delete;
    void swap()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_index_done = m_index_pend;
        m_index_pend = m_index_pend == 1 ? 0 : 1;
    }

    OUTPUT*                      get_pending_buffer() { return &m_containers[m_index_pend]; }
    std::mutex                   m_mutex;
    OUTPUT                       m_containers[2];
    int                          m_index_pend{0};
    int                          m_index_done{0};
    std::future<OUTPUT*>         m_pending_result;
    bool                         m_first{true};
    async_manager_source<INPUT>* m_source;

    async_state m_state = async_state::idle;
    std::string m_name;
};
