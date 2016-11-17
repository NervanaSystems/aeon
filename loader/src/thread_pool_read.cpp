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

#include "thread_pool_read.hpp"
#include "util.hpp"

using namespace nervana;
using namespace std;

read_thread_pool::read_thread_pool(const shared_ptr<buffer_pool_in>& out,
                       const shared_ptr<batch_iterator>& b_it) :
    thread_pool(1),
    m_out(out),
    m_batch_iterator(b_it)
{
    affirm(m_count == 1, "thread pool count > 1");
}

void read_thread_pool::work(int id)
{
    // Fill input buffers.
    {
        unique_lock<mutex> lock(m_out->get_mutex());
        while (m_out->full() == true) {
            m_out->wait_for_non_full(lock);
        }

        uint32_t tries = 0;
        while(tries < 3) {
            try {
                tries += 1;
                m_batch_iterator->read(m_out->get_for_write());
                break;
            } catch(std::exception& e) {
                cout << "read_thread_pool exception:" << e.what() << endl;
                m_out->write_exception(std::current_exception());
            }
        }
        if(tries == 3) {
            cout << "tried reading 3 times and failed.  Giving up";
            throw std::runtime_error("tried 3 times to read from batch_iterator and failed each time.");
        }

        m_out->advance_write_pos();
    }
    m_out->signal_not_empty();
}
