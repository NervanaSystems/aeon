/*
 Copyright 2017 Intel(R) Nervana(TM)
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

#include <map>
#include <mutex>
#include <vector>

#include <rdma/fi_endpoint.h>

// Macro helper for defining sub messages within an opcode message.
#define SUB_MESSAGE_TYPE(opcode, elements)                                                         \
    struct opcode##_T                                                                              \
    {                                                                                              \
        enum                                                                                       \
        {                                                                                          \
            type = opcode                                                                          \
        };                                                                                         \
                                                                                                   \
        elements                                                                                   \
    };

namespace nervana
{
    namespace ofi
    {
        namespace detail
        {
            const size_t page_size = 4096;

            template <typename T>
            inline T floor(T value)
            {
                const uint64_t page_mask = (~(page_size - 1));
                return (T)((uint64_t)value & (page_mask));
            }

            template <typename T>
            inline T ceil(T value)
            {
                return floor(value + page_size - 1);
            }
        }

        class message
        {
        public:
            message() {}
            virtual ~message() { deallocate(); }
            message(const message* other) { copy(other); }
            const message& operator=(const message& other)
            {
                deallocate();
                copy(other);
                return *this;
            }

            char* allocate(size_t s);
            void deallocate();

            // pointer to data excluding header
            char* buffer();
            // total buffer size
            uint64_t size() const { return m_size; }
        protected:
            char*    m_buffer{nullptr};
            uint64_t m_size{0};

        private:
            void copy(const message& other);

            const size_t header_size{64};
            const size_t header_data_size{header_size - sizeof(m_size)};
            const size_t max_message_size{detail::page_size - 1};
        };

        class rdma_memory final
        {
            friend class ofi;

        public:
            rdma_memory() {}
            rdma_memory(size_t size) { allocate(size); }
            rdma_memory(const rdma_memory* other) { copy(other); }
            const rdma_memory& operator=(const rdma_memory& other)
            {
                deallocate();
                copy(other);
                return *this;
            }
            ~rdma_memory() { deallocate(); }
            void allocate(size_t size);
            void deallocate();

            void*    get_buffer() { return m_buffer; }
            size_t   get_buffer_size() const { return m_size; }
            bool     is_allocated() const { return m_buffer != nullptr; }
            bool     is_registered() const { return m_mr_fid != nullptr; }
            uint64_t get_key() const;

        private:
            void copy(const rdma_memory& other);
            void*   m_buffer{nullptr};
            size_t  m_size{0};
            fid_mr* m_mr_fid{nullptr};
        };

        class connection_info
        {
        public:
            connection_info() {}
            connection_info(const std::string& address, const std::string& port)
                : m_address(address)
                , m_port(port)
            {
            }
            void set_address(const std::string& address) { m_address = address; }
            void set_port(const std::string& port) { m_port = port; }
            const std::string&               get_address() const { return m_address; }
            const std::string&               get_port() const { return m_port; }
        private:
            std::string m_address;
            std::string m_port;
        };

        class ofi
        {
        public:
            ofi();
            ~ofi();

            void bind_and_listen(unsigned int port, const std::string& address);
            void connect(const connection_info& info);
            bool wait_for_connect(ofi& connection, int timeout_ms = 10 * 1000);
            void disconnect();

            void read_from_remote_host(rdma_memory& memory,
                                       uint64_t     remote_address,
                                       size_t       size,
                                       uint64_t     key);

            void register_memory(rdma_memory& memory);
            void unregister_memory(rdma_memory& memory);

            void send(message& out_message);
            void receive(message& in_message);
            bool receive_ready(int timeout_ms = 500);

            const std::string& get_address() const { return m_address; }
            unsigned int       get_port() const { return m_port; }
        private:
            enum class status
            {
                uninitialized = 0,
                listener,
                communicator
            };

            struct ofi_header
            {
                uint64_t data_size;
            };

            static const int rx_tx_buffer_size{detail::page_size};
            static const int rx_buffers_number{256};
            static const int status_check_frequency{1000};

            void register_msg_mrs();
            void unregister_msg_mrs();

            void assure_connection_ok();
            bool handle_rx_queue(int timeout_ms);

            uint8_t* get_rx_buffer(uint32_t buffer_id)
            {
                return (m_msg_rx_buf + buffer_id * rx_tx_buffer_size);
            }

            uint8_t* get_rx_buffer_shadow(uint32_t buffer_id)
            {
                return (m_msg_rx_buf_shadow + buffer_id * rx_tx_buffer_size);
            }

            status       m_status;
            bool         m_initialized;
            std::string  m_address;
            unsigned int m_port;
            std::mutex   m_lock;

            ssize_t m_receive_ready_cnt;

            fid_pep* m_fid_pep;

            fi_info*    m_fi_info;
            fi_info*    m_fi_hints;
            fid_fabric* m_fid_fabric;
            fid_domain* m_fid_domain;

            ssize_t m_current_rx_shadow_id;

            fid_mr*  m_msg_tx_mr;
            uint8_t* m_msg_tx_buf;
            fid_mr*  m_msg_rx_mr;
            uint8_t* m_msg_rx_buf;
            uint8_t* m_msg_rx_buf_shadow;

            fid_eq*    m_fid_eq;
            fi_eq_attr m_fid_eq_attr;

            fid_cq*    m_fid_cq_rx;
            fid_cq*    m_fid_cq_tx;
            fi_cq_attr m_fi_cq_attr;

            fid_ep* m_fid_ep;
        };
    }
}
