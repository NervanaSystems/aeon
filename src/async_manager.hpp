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
#include <tuple>
#include <exception>

#include "log.hpp"
#include "blocking_queue.h"

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
    virtual void    suspend_output() {}
    async_manager_source(const async_manager_source&) = default;
};

template <typename INPUT, typename OUTPUT>
class nervana::async_manager : public virtual nervana::async_manager_source<OUTPUT>,
                               public async_manager_info
{
public:
    async_manager(std::shared_ptr<async_manager_source<INPUT>> source, const std::string& name)
        : m_source(source)
        , m_state{async_state::idle}
        , m_name{name}
    {
        // Make the container pair?  Currently letting child handle it in filler()
        async_manager_status.push_back(this);
    }
    virtual ~async_manager() { finalize(); }
    OUTPUT* next() override
    {
        if (!m_active_thread)
            initialize();

        inner_buffer_t output_buffer;
        if (!m_bfirst_next)
        {
            m_bq_output.top(output_buffer);
            if (std::get<0>(output_buffer) == nullptr)
            {
                return nullptr;
            }
            m_bq_output.pop(output_buffer);
            m_bq_input.push(output_buffer);
        }
        m_bfirst_next = false;

        m_bq_output.top(output_buffer);
        if (std::get<1>(output_buffer))
            std::rethrow_exception(std::get<1>(output_buffer));

        return std::get<0>(output_buffer);
    }

    // do the work to fill up m_containers
    virtual OUTPUT* filler() = 0;

    virtual void reset() override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_active_thread)
        {
            m_active_thread = false;
            m_bq_input.clear();
            m_bq_input.push(inner_buffer_t(nullptr, nullptr));
            m_source->suspend_output();
            fill_thread->join();
        }
        m_source->reset();
    }

    virtual void initialize()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_active_thread)
        {
            m_active_thread = true;
            m_bfirst_next   = true;
            m_bq_input.clear();
            m_bq_output.clear();
            m_bq_input.push(inner_buffer_t(&m_containers[0], nullptr));
            m_bq_input.push(inner_buffer_t(&m_containers[1], nullptr));
            fill_thread.reset(new std::thread(&async_manager::run_filler, this));
        }
    }

    void         finalize() { reset(); }
    virtual void suspend_output() override
    {
        m_bq_output.clear();
        m_bq_output.push(inner_buffer_t(nullptr, nullptr));
    }

    async_state        get_state() const override { return m_state; }
    const std::string& get_name() const override { return m_name; }
protected:
    typedef std::tuple<OUTPUT*, std::exception_ptr> inner_buffer_t;

    async_manager(const async_manager&) = delete;

    void run_filler()
    {
        for (;;)
        {
            inner_buffer_t free_buffer;
            m_bq_input.pop(free_buffer);

            m_pending_buffer = std::get<0>(free_buffer);

            if (!m_active_thread)
                return;
            if (m_pending_buffer == nullptr)
            {
                m_bq_output.push(inner_buffer_t(nullptr, nullptr));
                return;
            }

            OUTPUT* buff;
            try
            {
                buff = filler();
            }
            catch (...)
            {
                m_bq_output.push(inner_buffer_t(nullptr, std::current_exception()));
                return;
            }

            if (!m_active_thread)
                return;
            m_bq_output.push(inner_buffer_t(buff, nullptr));
        }
    }

    OUTPUT* get_pending_buffer()
    {
        if (m_active_thread)
            return m_pending_buffer;
        else
            return &m_containers[0];
    }
    OUTPUT                                       m_containers[2];
    OUTPUT*                                      m_pending_buffer;
    std::shared_ptr<async_manager_source<INPUT>> m_source;

    async_state m_state = async_state::idle;
    std::string m_name;

    BlockingQueue<inner_buffer_t> m_bq_input;
    BlockingQueue<inner_buffer_t> m_bq_output;
    std::shared_ptr<std::thread>  fill_thread;
    bool                          m_bfirst_next{true};
    volatile bool                 m_active_thread{false};
    std::mutex                    m_mutex;
};
