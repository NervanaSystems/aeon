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

#include "thread_pool_decode.hpp"

using namespace std;
using namespace nervana;

decode_thread_pool::decode_thread_pool(int count, const shared_ptr<buffer_pool_in>& in, const shared_ptr<buffer_pool_out>& out,
                                       const shared_ptr<python_backend>& pbe)
    : thread_pool(count)
    , m_buffer_pool_encoded(in)
    , m_buffer_pool_decoded(out)
    , m_python_backend(pbe)
    , m_batch_size(m_python_backend->m_batch_size)
{
    m_items_per_thread = (m_batch_size - 1) / m_count + 1;
    affirm(m_items_per_thread * count >= m_batch_size, "m_items_per_thread * count >= m_batch_size");
    affirm(m_items_per_thread * (count - 1) < m_batch_size, "m_items_per_thread * (count - 1) < m_batch_size");
}

void decode_thread_pool::add_provider(std::shared_ptr<nervana::provider_interface> prov)
{
    m_providers.push_back(prov);
    m_start_signaled.emplace_back(false);
    m_start_inds.push_back(0);
    m_end_inds.push_back(0);
}

decode_thread_pool::~decode_thread_pool()
{
    if (m_manager != 0)
    {
        m_manager->join();
        delete m_manager;
    }
    // Other thread objects are freed in the destructor of the parent class.
}

void decode_thread_pool::start()
{
    for (int i = 0; i < m_count; i++)
    {
        m_threads.push_back(new thread(&decode_thread_pool::run, this, i));
    }
    m_manager = new thread(&decode_thread_pool::manage, this);
}

void decode_thread_pool::stop()
{
    thread_pool::stop();
    while (stopped() == false)
    {
        std::this_thread::yield();
        m_buffer_pool_encoded->switch_write_buffer();
        m_buffer_pool_encoded->signal_available_read_buffer();
    }

    m_stop_manager = true;
    while (m_manager_stopped == false)
    {
        std::this_thread::yield();
        m_buffer_pool_encoded->switch_write_buffer();
        m_buffer_pool_encoded->signal_available_read_buffer();
        m_end_signaled++;
        m_ended.notify_one();
    }
}

void decode_thread_pool::run(int id)
{
    // Initialize worker threads by computing memory offsets for the
    // data this thread should work on
    try
    {
        affirm(id < m_count, "id < m_count");
        m_start_inds[id] = id * m_items_per_thread;
        int itemCount    = m_items_per_thread;
        if (id == m_count - 1)
        {
            itemCount = m_batch_size - id * m_items_per_thread;
        }

        m_end_inds[id] = m_start_inds[id] + itemCount;

        while (m_done == false)
        {
            work(id);
        }

        m_stopped[id] = true;
    }
    catch (std::exception& e)
    {
        cerr << "fatal exception in decode_thread_pool::run: " << e.what() << endl;
        // TODO: fail gracefully, not seg fault
    }
}


void decode_thread_pool::work(int id)
{
    // Thread function.
    {
        unique_lock<mutex> lock(m_mutex);
        while (!m_start_signaled[id])
        {
            m_started.wait(lock);
            if (m_done == true)
            {
                return;
            }
        }
        m_start_signaled[id] = !m_start_signaled[id];
        affirm(m_start_signaled[id] == false, "startSignaled not cleared");
    }

    // No locking required because threads write into non-overlapping regions.
    try
    {
        buffer_in_array &input_buf_array = m_buffer_pool_encoded->get_read_buffer();
        affirm(input_buf_array[0]->get_item_count() != 0, "input buffer pool is empty");

        for (int i = m_start_inds[id]; i < m_end_inds[id]; i++)
        {
            m_providers[id]->provide(i,
                                     m_buffer_pool_encoded->get_read_buffer(),
                                     m_buffer_pool_decoded->get_write_buffer());
        }
    }
    catch (std::exception& e)
    {
        cout << "decode_thread_pool exception: " << e.what() << endl;
        m_buffer_pool_decoded->write_exception(std::current_exception());
    }

    m_end_signaled++;
    affirm(m_end_signaled <= m_count, "endSignaled > count");

    m_ended.notify_one();
}

void decode_thread_pool::produce()
{
    // lock on output buffers and copy to device
    {
        // Make sure we have somewhere to write
        unique_lock<mutex> out_lock(m_buffer_pool_decoded->get_mutex());
        while (m_buffer_pool_decoded->no_write_buffers())
        {
            m_buffer_pool_decoded->wait_for_available_write_buffer(out_lock);
        }

        for (unsigned int i = 0; i < m_start_signaled.size(); i++)
        {
            m_start_signaled[i] = true;
        }

        // Let all the providers know they can start
        m_started.notify_all();

        {
            // Checks to make sure that all worker threads have returned
            unique_lock<mutex> lock(m_mutex);
            while (m_end_signaled < m_count)
            {
                m_ended.wait(lock);
            }
            m_end_signaled = 0;
        }

        try
        {
            // At this point, we have decoded data for the whole minibatch.
            // Do any messy cross datum stuff you may need to do that requires minibatch consistency
            m_providers[0]->post_process(m_buffer_pool_decoded->get_write_buffer());

            // Copy to device.
            m_python_backend->call_backend_transfer(m_buffer_pool_decoded->get_write_buffer(),
                                                    m_buffer_index);
        }
        catch (std::exception& e)
        {
            cout << "exception in provider post_process/call to backend transfer: " << e.what();
        }
        m_buffer_index = (m_buffer_index == 0) ? 1 : 0;
        m_buffer_pool_decoded->switch_write_buffer();
    }
    m_buffer_pool_decoded->signal_available_read_buffer();
}

void decode_thread_pool::consume()
{
    // lock on input buffers and call produce
    {
        // Make sure we have something to read
        unique_lock<mutex> in_lock(m_buffer_pool_encoded->get_mutex());
        while (m_buffer_pool_encoded->no_read_buffers())
        {
            m_buffer_pool_encoded->wait_for_available_read_buffer(in_lock);

            if (m_stop_manager)
            {
                return;
            }
        }

        // Signal Workers to start and wait for them to finish
        produce();

        m_buffer_pool_encoded->switch_read_buffer();

    }
    m_buffer_pool_encoded->signal_available_write_buffer();
}

void decode_thread_pool::manage()
{
    try
    {
        while (m_stop_manager == false)
        {
            consume();
        }
        m_manager_stopped = true;
    }
    catch (std::exception& e)
    {
        cerr << "exception in decode_thread_pool::manage: " << e.what() << endl;
        // TODO: fail gracefully, not seg fault
    }
}
