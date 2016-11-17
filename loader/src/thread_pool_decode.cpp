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


decode_thread_pool::decode_thread_pool(int count,
                                       const shared_ptr<buffer_pool_in>& in,
                                       const shared_ptr<buffer_pool_out>& out,
                                       const shared_ptr<python_backend>& pbe) :
    thread_pool(count),
    m_in(in),
    m_out(out),
    m_python_backend(pbe),
    m_batch_size(m_python_backend->m_batch_size)
{
    m_items_per_thread = (m_batch_size - 1) / m_count + 1;
    affirm(m_items_per_thread * count >= m_batch_size, "m_items_per_thread * count >= m_batch_size");
    affirm(m_items_per_thread * (count - 1) < m_batch_size, "m_items_per_thread * (count - 1) < m_batch_size");
}

void decode_thread_pool::add_provider(std::shared_ptr<nervana::provider_interface> prov)
{
    m_providers.push_back(prov);
    m_start_signaled.push_back(0);

    m_start_inds.push_back(0);
    m_end_inds.push_back(0);
}

decode_thread_pool::~decode_thread_pool()
{
    if (m_manager != 0) {
        m_manager->join();
        delete m_manager;
    }
    // Other thread objects are freed in the destructor of the parent class.
}

void decode_thread_pool::start()
{
    for (int i = 0; i < m_count; i++) {
        m_threads.push_back(new thread(&decode_thread_pool::run, this, i));
    }
    m_manager = new thread(&decode_thread_pool::manage, this);
}

void decode_thread_pool::stop()
{
    thread_pool::stop();
    while (stopped() == false) {
        std::this_thread::yield();
        m_in->advance_write_pos();
        m_in->signal_not_empty();
    }

    m_stop_manager = true;
    while (m_manager_stopped == false) {
        std::this_thread::yield();
        m_in->advance_write_pos();
        m_in->signal_not_empty();
        m_end_signaled++;
        m_ended.notify_one();
    }
}

void decode_thread_pool::run(int id)
{
    // Initialize worker threads by computing memory offsets for the
    // data this thread should work on
    try {
        affirm(id < m_count, "id < m_count");
        m_start_inds[id] = id * m_items_per_thread;
        int itemCount = m_items_per_thread;
        if (id == m_count - 1) {
            itemCount = m_batch_size - id * m_items_per_thread;
        }

        m_end_inds[id] = m_start_inds[id] + itemCount;

        while (m_done == false) {
            work(id);
        }

        m_stopped[id] = true;
    } catch (std::exception& e) {
        cerr << "fatal exception in decode_thread_pool::run: " << e.what() << endl;
        // TODO: fail gracefully, not seg fault
    }
}

void decode_thread_pool::work(int id)
{
    // Thread function.
    {
        unique_lock<mutex> lock(m_mutex);
        while (m_start_signaled[id] == 0) {
            m_started.wait(lock);
            if (m_done == true) {
                return;
            }
        }
        m_start_signaled[id]--;
        affirm(m_start_signaled[id] == 0, "startSignaled not cleared");
    }

    // No locking required because threads write into non-overlapping regions.
    try {
        affirm((*m_input_buf)[0]->get_item_count() != 0, "input buffer to decoded_thread_pool is empty");

        for (int i = m_start_inds[id]; i < m_end_inds[id]; i++) {
            m_providers[id]->provide(i, *m_input_buf, m_out->get_for_write());
        }
    } catch (std::exception& e) {
        cout << "decode_thread_pool exception: " << e.what() << endl;
        m_out->write_exception(std::current_exception());
    }

    {
        lock_guard<mutex> lock(m_mutex);
        m_end_signaled++;
        affirm(m_end_signaled <= m_count, "endSignaled > count");
    }
    m_ended.notify_one();
}

void decode_thread_pool::produce()
{
    // lock on output buffers and copy to device
    {
        unique_lock<mutex> lock(m_out->get_mutex());
        while (m_out->full() == true) {
            m_out->wait_for_non_full(lock);
        }
        {
            lock_guard<mutex> lock(m_mutex);
            for (unsigned int i = 0; i < m_start_signaled.size(); i++) {
                m_start_signaled[i] = 1;
            }
        }
        m_started.notify_all();
        {
            unique_lock<mutex> lock(m_mutex);
            while (m_end_signaled < m_count) {
                m_ended.wait(lock);
            }
            m_end_signaled = 0;
        }

        try {
            // At this point, we have decoded data for the whole minibatch.
            buffer_out_array& outBuf = m_out->get_for_write();

            // Do any messy cross datum stuff you may need to do that requires minibatch consistency
            m_providers[0]->post_process(outBuf);

            // Copy to device.
            m_python_backend->call_backend_transfer(outBuf, m_buffer_index);
        } catch (std::exception& e) {
            cout << "exception in provider post_process/call to backend transfer: " << e.what();
        }

        m_buffer_index = (m_buffer_index == 0) ? 1 : 0;
        m_out->advance_write_pos();
    }
    m_out->signal_not_empty();
}

void decode_thread_pool::consume()
{
    // lock on input buffers and call produce
    {
        unique_lock<mutex> lock(m_in->get_mutex());
        while (m_in->empty() == true) {
            m_in->wait_for_not_empty(lock);
            if (m_stop_manager == true) {
                return;
            }
        }
        m_input_buf = &m_in->get_for_read();
        produce();
        m_in->advance_read_pos();
    }
    m_in->signal_not_full();
}

void decode_thread_pool::manage()
{
    try {
        // Thread function.
        int result = 0;
        if (result != 0) {
            m_stop_manager = true;
        }
        while (m_stop_manager == false) {
            consume();
        }
        m_manager_stopped = true;
    } catch (std::exception& e) {
        cerr << "exception in decode_thread_pool::manage: " << e.what() << endl;
        // TODO: fail gracefully, not seg fault
    }
}
