/*
 Copyright 2017 Nervana Systems Inc.
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

#include <thread>
#include <atomic>
#include <mutex>
#include <exception>

#ifdef __linux__
#include <pthread.h>
#endif

namespace nervana
{
    template <typename T, void (T::*process_func)(int index)>
    class thread_pool;
    class thread_barrier;
}

#ifdef __linux__
class nervana::thread_barrier
{
public:
    thread_barrier(unsigned int count) { pthread_barrier_init(&barrier, NULL, count); }
    ~thread_barrier() { pthread_barrier_destroy(&barrier); }
    void wait() { pthread_barrier_wait(&barrier); }
private:
    pthread_barrier_t barrier;
};
#else
class nervana::thread_barrier
{
public:
    thread_barrier(unsigned int count)
        : m_count(count)
        , m_active_thread_count(m_count)
    {
    }
    void wait()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        unsigned int                 last_iteration = m_iteration;

        if (--m_active_thread_count != 0)
        {
            m_cond.wait(lock, [this, last_iteration] { return last_iteration != m_iteration; });
        }
        else
        {
            m_iteration++;
            m_active_thread_count = m_count;
            m_cond.notify_all();
        }
    }

private:
    std::condition_variable m_cond;
    std::mutex              m_mutex;
    unsigned int            m_count;
    unsigned int            m_active_thread_count;
    unsigned int            m_iteration = 0;
};
#endif

template <typename T, void (T::*process_func)(int index)>
class nervana::thread_pool
{
public:
    thread_pool(T* worker, int thread_count, int task_count)
        : m_worker(worker)
        , m_task_count(task_count)
    {
        int nthreads;

        if (thread_count == 0) // automatically determine number of threads
        {
            // we don't use all threads, some of them we leave for other pipeline objects and system
            nthreads = std::thread::hardware_concurrency() -
                       std::min(m_max_count_of_free_threads,
                                static_cast<int>(std::thread::hardware_concurrency() /
                                                 m_free_threads_ratio));
            nthreads = std::min(nthreads, m_task_count);
        }
        else
        {
            // don't return more threads than we can get
            nthreads =
                std::min(static_cast<int>(std::thread::hardware_concurrency()), thread_count);
        }

        m_br_wake.reset(new thread_barrier(nthreads + 1));
        m_br_endtasks.reset(new thread_barrier(nthreads + 1));

        if (nthreads == m_task_count)
        {
            for (int i = 0; i < nthreads; i++)
                m_threads.emplace_back(&thread_pool::process<false>, this, i);
        }
        else
        {
            for (int i = 0; i < nthreads; i++)
                m_threads.emplace_back(&thread_pool::process<true>, this, i);
        }
    }

    ~thread_pool()
    {
        m_thread_pool_stop.store(true, std::memory_order_relaxed);
        m_br_wake->wait();
        for (auto& thread : m_threads)
            thread.join();
    }

    void run()
    {
        m_current_task_id = 0;
        m_pool_exception  = nullptr;
        m_br_wake->wait();
        m_br_endtasks->wait();
        if (m_pool_exception)
            std::rethrow_exception(m_pool_exception);
    }

private:
    const int                       m_max_count_of_free_threads = 2;
    const int                       m_free_threads_ratio        = 8;
    T*                              m_worker;
    int                             m_task_count;
    std::unique_ptr<thread_barrier> m_br_wake;
    std::unique_ptr<thread_barrier> m_br_endtasks;
    std::atomic<bool>               m_thread_pool_stop{false};
    std::vector<std::thread>        m_threads;
    std::atomic<size_t>             m_current_task_id;
    std::exception_ptr              m_pool_exception;
    std::mutex                      m_mutex;

    template <bool dynamic_task_scheduling>
    void process(int thread_id)
    {
#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif

        for (;;)
        {
            m_br_wake->wait();

            if (m_thread_pool_stop.load(std::memory_order_relaxed))
                return;

            try
            {
                if (!dynamic_task_scheduling)
                {
                    (m_worker->*process_func)(thread_id);
                }
                else
                {
                    for (;;)
                    {
                        const size_t next_task_id =
                            m_current_task_id.fetch_add(1, std::memory_order_relaxed);
                        if (next_task_id >= m_task_count)
                            break;
                        (m_worker->*process_func)(next_task_id);
                    }
                }
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_pool_exception)
                    m_pool_exception = std::current_exception();
            }

            m_br_endtasks->wait();
        }
    }
};
