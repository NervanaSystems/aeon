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

namespace nervana {
    class thread_pool;
}

/* thread_pool
 *
 * A collection of a constant number of threads implemented
 * using std::thread.  Methods are provided to start, stop and join all
 * N threads simultaneously.
 *
 */
class nervana::thread_pool
{
public:
    explicit thread_pool(int count) :
        m_count(count),
        m_done(false)
    {
        m_stopped = new bool[count];
        for (int i = 0; i < count; i++) {
            m_stopped[i] = false;
        }
    }

    virtual ~thread_pool()
    {
        for (auto t : m_threads) {
            t->join();
            delete t;
        }
        delete[] m_stopped;
    }

    virtual void start()
    {
        for (int i = 0; i < m_count; i++) {
            m_threads.push_back(new std::thread(&thread_pool::run, this, i));
        }
    }

    virtual void stop()
    {
        m_done = true;
    }

    bool stopped()
    {
        for (int i = 0; i < m_count; i++) {
            if (m_stopped[i] == false) {
                return false;
            }
        }
        return true;
    }

    void join()
    {
        for (auto t : m_threads) {
            t->join();
        }
    }

protected:
    virtual void work(int id) = 0;

    virtual void run(int id)
    {
        while (m_done == false) {
            work(id);
        }
        m_stopped[id] = true;
    }

protected:
    int                         m_count;
    std::vector<std::thread*>   m_threads;
    bool                        m_done;
    bool*                       m_stopped;
};
