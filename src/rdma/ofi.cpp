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

#include <algorithm>
#include <iostream>
#include <cstring>
#include <cassert>
#include <stdexcept>

#include <netinet/in.h>
#include <unistd.h>
#include <linux/if_link.h>
#include <linux/if_arp.h>
#include <sys/time.h>
#include <netdb.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>

#include "ofi.hpp"
#include "../service/log.hpp"

#define CHECK_OFI_ERROR(ret, label, error_msg, res_val)                                            \
    if (ret)                                                                                       \
    {                                                                                              \
        res_val = error_msg;                                                                       \
        goto label;                                                                                \
    }

#define GET_QUEUE_ERROR_FUNCTION(queue_type)                                                       \
    string ofi_##queue_type##_read_error(fid_##queue_type* queue)                                  \
    {                                                                                              \
        fi_##queue_type##_err_entry error_entry;                                                   \
        int                         length = fi_##queue_type##_readerr(queue, &error_entry, 0);    \
        if (length != sizeof(error_entry))                                                         \
        {                                                                                          \
            nervana::log::error(                                                                   \
                "%s: %s", "fi_" #queue_type "_readerr error: ", fi_strerror(-length));             \
            return "";                                                                             \
        }                                                                                          \
        const char* provider_error = fi_##queue_type##_strerror(                                   \
            queue, error_entry.prov_errno, error_entry.err_data, NULL, 0);                         \
        std::ostringstream stream;                                                                 \
        stream << "fi_" #queue_type " error: " << fi_strerror(error_entry.err) << "; ";            \
        stream << "fi_" #queue_type " provider error: " << provider_error;                         \
        return stream.str();                                                                       \
    }

using std::map;
using std::runtime_error;
using std::string;
using std::lock_guard;
using std::mutex;
using std::mutex;

namespace
{
    void assert_ofi_success(int status, const string& message);
    bool other_side_died(fid_cq* queue);
    GET_QUEUE_ERROR_FUNCTION(cq)
    GET_QUEUE_ERROR_FUNCTION(eq)
}

nervana::ofi::ofi::ofi()
    : m_status(status::uninitialized)
    , m_initialized(false)
    , m_receive_ready_cnt(0)
    , m_fid_pep(NULL)
    , m_fi_info(NULL)
    , m_fi_hints(NULL)
    , m_fid_fabric(NULL)
    , m_fid_domain(NULL)
    , m_current_rx_shadow_id(0)
    , m_msg_tx_mr(NULL)
    , m_msg_tx_buf(NULL)
    , m_msg_rx_mr(NULL)
    , m_msg_rx_buf(NULL)
    , m_msg_rx_buf_shadow(NULL)
    , m_fid_eq(NULL)
    , m_fid_cq_rx(NULL)
    , m_fid_cq_tx(NULL)
    , m_fid_ep(NULL)
{
    m_fi_hints                    = fi_allocinfo();
    m_fi_hints->ep_attr->type     = FI_EP_MSG;
    m_fi_hints->ep_attr->protocol = FI_PROTO_RDMA_CM_IB_RC;
    m_fi_hints->caps              = FI_MSG;
    m_fi_hints->mode              = FI_LOCAL_MR;
    m_fi_hints->addr_format       = FI_SOCKADDR_IN;

    memset(&m_fid_eq_attr, 0, sizeof(fi_eq_attr));
    m_fid_eq_attr.wait_obj = FI_WAIT_FD;

    memset(&m_fi_cq_attr, 0, sizeof(fi_cq_attr));
    m_fi_cq_attr.format   = FI_CQ_FORMAT_CONTEXT;
    m_fi_cq_attr.wait_obj = FI_WAIT_FD;
    m_fi_cq_attr.size     = 1024;
};

nervana::ofi::ofi::~ofi()
{
    m_status = status::uninitialized;
    disconnect();

    if (m_fid_pep)
    {
        fi_close(&m_fid_pep->fid);
        m_fid_pep = NULL;
    }

    if (m_fid_ep)
    {
        fi_close(&m_fid_ep->fid);
        m_fid_ep = NULL;
    }

    unregister_msg_mrs();

    if (m_fid_cq_rx)
    {
        fi_close(&m_fid_cq_rx->fid);
        m_fid_cq_rx = NULL;
    }

    if (m_fid_cq_tx)
    {
        fi_close(&m_fid_cq_tx->fid);
        m_fid_cq_tx = NULL;
    }

    if (m_fid_eq)
    {
        fi_close(&m_fid_eq->fid);
        m_fid_eq = NULL;
    }

    if (m_fid_domain)
    {
        fi_close(&m_fid_domain->fid);
        m_fid_domain = NULL;
    }

    if (m_fid_fabric)
    {
        fi_close(&m_fid_fabric->fid);
        m_fid_fabric = NULL;
    }

    if (m_fi_info)
    {
        fi_freeinfo(m_fi_info);
        m_fi_info = NULL;
    }

    if (m_fi_hints)
    {
        fi_freeinfo(m_fi_hints);
        m_fi_hints = NULL;
    }

    return;
}

void nervana::ofi::ofi::bind_and_listen(unsigned int port, const string& address)
{
    lock_guard<mutex> lg(m_lock);
    m_address = address;
    m_port    = port;

    int status = fi_getinfo(fi_version(),
                            m_address.c_str(),
                            std::to_string(port).c_str(),
                            FI_SOURCE,
                            m_fi_hints,
                            &m_fi_info);
    string error_msg;
    CHECK_OFI_ERROR(status, error_info, "fi_getinfo failed", error_msg);

    status = fi_fabric(m_fi_info->fabric_attr, &m_fid_fabric, NULL);
    CHECK_OFI_ERROR(status, error_fabric, "fi_fabric failed", error_msg);

    status = fi_passive_ep(m_fid_fabric, m_fi_info, &m_fid_pep, NULL);
    CHECK_OFI_ERROR(status, error_passive_ep, "fi_passive_ep failed", error_msg);

    status = fi_eq_open(m_fid_fabric, &m_fid_eq_attr, &m_fid_eq, NULL);
    CHECK_OFI_ERROR(status, error_eq, "fi_eq_open failed", error_msg);

    status = fi_pep_bind(m_fid_pep, &m_fid_eq->fid, 0);
    CHECK_OFI_ERROR(status, error_eq, "fi_pep_bind failed", error_msg);

    status = fi_listen(m_fid_pep);
    CHECK_OFI_ERROR(status, error_eq, "fi_listen failed", error_msg);

    m_status      = status::listener;
    m_initialized = true;

    return;

error_eq:
    if (m_fid_eq)
    {
        fi_close(&m_fid_eq->fid);
        m_fid_eq = NULL;
    }

error_passive_ep:
    if (m_fid_pep)
    {
        fi_close(&m_fid_pep->fid);
        m_fid_pep = NULL;
    }

error_fabric:
    if (m_fid_fabric)
    {
        fi_close(&m_fid_fabric->fid);
        m_fid_fabric = NULL;
    }

error_info:
    if (m_fi_info)
    {
        fi_freeinfo(m_fi_info);
        m_fi_info = NULL;
    }

    throw std::runtime_error("cannot bind and listen with OFI: " + error_msg);
}

void nervana::ofi::ofi::connect(const connection_info& connection)
{
    const int timeout_ms = 1000 * 10;

    lock_guard<mutex> lg(m_lock);

    int status = fi_getinfo(fi_version(),
                            connection.get_address().c_str(),
                            connection.get_port().c_str(),
                            0,
                            m_fi_hints,
                            &m_fi_info);
    assert_ofi_success(status, "fi_get_info failed");

    status = fi_fabric(m_fi_info->fabric_attr, &m_fid_fabric, NULL);
    assert_ofi_success(status, "fi_fabric failed");

    status = fi_domain(m_fid_fabric, m_fi_info, &m_fid_domain, NULL);
    assert_ofi_success(status, "fi_domain failed");

    status = fi_eq_open(m_fid_fabric, &m_fid_eq_attr, &m_fid_eq, NULL);
    assert_ofi_success(status, "fi_eq_open failed");

    status = fi_cq_open(m_fid_domain, &m_fi_cq_attr, &m_fid_cq_tx, NULL);
    assert_ofi_success(status, "couldn't open send completion queue");

    status = fi_cq_open(m_fid_domain, &m_fi_cq_attr, &m_fid_cq_rx, NULL);
    assert_ofi_success(status, "couldn't open recv completion queue");

    status = fi_endpoint(m_fid_domain, m_fi_info, &m_fid_ep, NULL);
    assert_ofi_success(status, "fi_endpoint failed");

    status = fi_ep_bind(m_fid_ep, &m_fid_eq->fid, 0);
    assert_ofi_success(status, "binding event queue to endpoint failed");

    status = fi_ep_bind(m_fid_ep, &m_fid_cq_tx->fid, FI_SEND);
    assert_ofi_success(status, "binding send completion queue to endpoint failed");

    status = fi_ep_bind(m_fid_ep, &m_fid_cq_rx->fid, FI_RECV);
    assert_ofi_success(status, "binding recv completion queue to endpoint failed");

    status = fi_connect(m_fid_ep, m_fi_info->dest_addr, NULL, 0);
    assert_ofi_success(status, "fi_connect failed");

    fi_eq_cm_entry entry;
    uint32_t       event;
    ssize_t        ret_size = fi_eq_sread(m_fid_eq, &event, &entry, sizeof(entry), timeout_ms, 0);

    if (ret_size != sizeof(entry))
    {
        string error_message = ofi_eq_read_error(m_fid_eq);
        error_message        = "no connection acq: " + error_message;
        throw runtime_error(error_message);
    }

    if (event != FI_CONNECTED || entry.fid != &m_fid_ep->fid)
    {
        string message = "wrong event type: " + std::to_string(event);
        throw runtime_error(message);
    }

    try
    {
        register_msg_mrs();
    }
    catch (const std::exception&)
    {
        disconnect();
        throw;
    }

    for (int i = 0; i < rx_buffers_number; i++)
    {
        status = fi_recv(m_fid_ep,
                         get_rx_buffer(i),
                         rx_tx_buffer_size,
                         fi_mr_desc(m_msg_rx_mr),
                         0,
                         get_rx_buffer(i));
        assert_ofi_success(status, "fi_recv failed");
    }

    m_status      = status::communicator;
    m_initialized = true;
}

bool nervana::ofi::ofi::wait_for_connect(ofi& connection, int timeout_ms)
{
    lock_guard<mutex> lg(m_lock);

    if (m_status != status::listener)
    {
        throw runtime_error("wrong OFI object state: bind_and_listen was not called");
    }

    fi_eq_cm_entry entry;
    uint32_t       event;
    ssize_t        ret_size = fi_eq_sread(m_fid_eq, &event, &entry, sizeof(entry), timeout_ms, 0);

    if (ret_size == -FI_EAGAIN)
    {
        return false;
    }

    if (ret_size != sizeof(entry))
    {
        string error_message = ofi_eq_read_error(m_fid_eq);
        throw runtime_error(string("size of entry structure does not match: ") +
                            error_message.c_str());
    }

    if (event != FI_CONNREQ)
    {
        throw runtime_error("wrong event type: " + std::to_string(event) + " expected: " +
                            std::to_string(FI_CONNREQ));
    }

    fid_t handle = entry.info->handle;

    int status = fi_fabric(m_fi_info->fabric_attr, &connection.m_fid_fabric, NULL);
    assert_ofi_success(status, "fi_fabric failed");

    status = fi_domain(connection.m_fid_fabric, entry.info, &(connection.m_fid_domain), NULL);
    assert_ofi_success(status, "fi_domain failed");

    status = fi_eq_open(connection.m_fid_fabric, &m_fid_eq_attr, &(connection.m_fid_eq), NULL);
    assert_ofi_success(status, "fi_eq_open failed");

    status = fi_cq_open(connection.m_fid_domain, &m_fi_cq_attr, &(connection.m_fid_cq_tx), NULL);
    assert_ofi_success(status, "fi_cq_open failed for tx queue");

    status = fi_cq_open(connection.m_fid_domain, &m_fi_cq_attr, &(connection.m_fid_cq_rx), NULL);
    assert_ofi_success(status, "fi_cq_open failed for rx queue");

    status = fi_endpoint(connection.m_fid_domain, entry.info, &(connection.m_fid_ep), NULL);
    assert_ofi_success(status, "fi_endpoint ailed");

    fi_freeinfo(entry.info);

    status = fi_ep_bind(connection.m_fid_ep, &(connection.m_fid_eq->fid), 0);
    assert_ofi_success(status, "binding event queue to endpoint failed");

    status = fi_ep_bind(connection.m_fid_ep, &(connection.m_fid_cq_tx->fid), FI_SEND);
    assert_ofi_success(status, "binding send completion queue to endpoint failed");

    status = fi_ep_bind(connection.m_fid_ep, &(connection.m_fid_cq_rx->fid), FI_RECV);
    assert_ofi_success(status, "binding recv completion queue to endpoint failed");

    try
    {
        connection.register_msg_mrs();
    }
    catch (const std::exception&)
    {
        status = fi_reject(m_fid_pep, handle, NULL, 0);
        throw;
    }

    status = fi_enable(connection.m_fid_ep);
    assert_ofi_success(status, "fi_enable failed");

    for (int i = 0; i < rx_buffers_number; i++)
    {
        status = fi_recv(connection.m_fid_ep,
                         connection.get_rx_buffer(i),
                         rx_tx_buffer_size,
                         fi_mr_desc(connection.m_msg_rx_mr),
                         0,
                         connection.get_rx_buffer(i));

        assert_ofi_success(status, "fi_recv failed");
    }

    status = fi_accept(connection.m_fid_ep, NULL, 0);
    assert_ofi_success(status, "fi_accept failed");

    ret_size = fi_eq_sread(connection.m_fid_eq, &event, &entry, sizeof(entry), timeout_ms, 0);

    if (ret_size != sizeof(entry))
    {
        string error_message = ofi_eq_read_error(connection.m_fid_eq);
        error_message        = "fi_eq_sread failed: " + error_message;
        throw runtime_error(error_message);
    }

    if (event != FI_CONNECTED || entry.fid != &(connection.m_fid_ep->fid))
    {
        throw runtime_error("wrong event type: " + std::to_string(event));
    }

    connection.m_initialized = true;

    return true;
}

void nervana::ofi::ofi::disconnect()
{
    if (m_fid_ep)
    {
        int status = fi_shutdown(m_fid_ep, 0);
        assert_ofi_success(status, "fi_shutdown failed");
    }
}

void nervana::ofi::ofi::register_msg_mrs()
{
    posix_memalign(reinterpret_cast<void**>(&m_msg_rx_buf_shadow),
                   detail::page_size,
                   rx_tx_buffer_size * rx_buffers_number);
    if (m_msg_rx_buf_shadow == NULL)
    {
        throw runtime_error("cannot allocate msg_rx_buf_shadow memory");
    }

    posix_memalign(reinterpret_cast<void**>(&m_msg_rx_buf),
                   detail::page_size,
                   rx_tx_buffer_size * rx_buffers_number);
    if (m_msg_rx_buf == NULL)
    {
        throw runtime_error("cannot allocate msg_rx_buf memory");
    }
    int status = fi_mr_reg(m_fid_domain,
                           m_msg_rx_buf,
                           rx_tx_buffer_size * rx_buffers_number,
                           FI_RECV,
                           0,
                           0,
                           0,
                           &m_msg_rx_mr,
                           NULL);
    if (m_msg_rx_mr == NULL)
    {
        throw runtime_error("fi_mr_reg returned null");
    }
    if (status)
        goto error;

    posix_memalign(reinterpret_cast<void**>(&m_msg_tx_buf), detail::page_size, rx_tx_buffer_size);
    if (m_msg_tx_buf == NULL)
    {
        throw runtime_error("cannot allocate msg_tx_buf memory");
    }
    status = fi_mr_reg(
        m_fid_domain, m_msg_tx_buf, rx_tx_buffer_size, FI_SEND, 0, 0, 0, &m_msg_tx_mr, NULL);
    if (m_msg_tx_mr == NULL)
    {
        throw runtime_error("fi_mr_reg returned null");
    }
    if (status)
        goto error;

    return;

error:
    if (m_msg_tx_mr)
    {
        fi_close(&m_msg_tx_mr->fid);
        m_msg_tx_mr = NULL;
    }
    free(m_msg_tx_buf);
    m_msg_tx_buf = NULL;

    if (m_msg_rx_mr)
    {
        fi_close(&m_msg_rx_mr->fid);
        m_msg_rx_mr = NULL;
    }
    free(m_msg_rx_buf);
    m_msg_rx_buf = NULL;

    free(m_msg_rx_buf_shadow);
    m_msg_rx_buf_shadow = NULL;

    throw runtime_error("register_msg_mrs failed");
}

void nervana::ofi::ofi::unregister_msg_mrs()
{
    if (m_msg_tx_mr)
    {
        fi_close(&m_msg_tx_mr->fid);
        m_msg_tx_mr = NULL;
    }
    free(m_msg_tx_buf);
    m_msg_tx_buf = NULL;

    if (m_msg_rx_mr)
    {
        fi_close(&m_msg_rx_mr->fid);
        m_msg_rx_mr = NULL;
    }
    free(m_msg_rx_buf);
    m_msg_rx_buf = NULL;

    free(m_msg_rx_buf_shadow);
    m_msg_rx_buf_shadow = NULL;
}

void nervana::ofi::ofi::read_from_remote_host(rdma_memory& memory,
                                              uint64_t     remote_address,
                                              size_t       size,
                                              uint64_t     key)
{
    void* context{memory.get_buffer()};
    void* description{fi_mr_desc(memory.m_mr_fid)};

    int status = fi_read(m_fid_ep,
                         memory.get_buffer(),
                         size,
                         description,
                         (fi_addr_t)NULL,
                         remote_address,
                         key,
                         context);

    if (status)
        throw runtime_error("read from remote host failed");

    int status_check_cnt{0};
    do
    {
        fi_cq_entry entry;
        status = fi_cq_read(m_fid_cq_tx, &entry, 1);
        if (status == 1)
        {
            if (entry.op_context != context)
            {
                throw runtime_error("received completion event from other part");
            }
            break;
        }

        if (status < 0 && status != -FI_EAGAIN)
        {
            if (status == -FI_EAVAIL && other_side_died(m_fid_cq_tx))
            {
                throw runtime_error("other side died");
            }
            throw runtime_error("fi_cq_read failed");
        }

        if (status_check_cnt > status_check_frequency)
        {
            status_check_cnt = 0;
            assure_connection_ok();
        }
        status_check_cnt++;
    } while (status == -FI_EAGAIN);
}

void nervana::ofi::ofi::assure_connection_ok()
{
    if (!m_initialized)
    {
        throw runtime_error("connection is not initialized");
    }

    fi_eq_cm_entry entry;
    uint32_t       event;
    auto           status = fi_eq_read(m_fid_eq, &event, &entry, sizeof(entry), 0);
    if (status == -EAGAIN)
    {
        return;
    }
    if (status == sizeof(entry) && entry.fid == &(m_fid_ep->fid) && event == FI_SHUTDOWN)
    {
        throw runtime_error("other side is disconnected");
    }
}

void nervana::ofi::ofi::register_memory(rdma_memory& memory)
{
    if (memory.is_registered())
        return;

    if (!memory.is_allocated())
    {
        throw runtime_error("cannot register memory: memory is not allocated");
    }

    int status = fi_mr_reg(m_fid_domain,
                           memory.get_buffer(),
                           memory.get_buffer_size(),
                           FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE,
                           0,
                           0,
                           0,
                           &memory.m_mr_fid,
                           0);
    if (status)
        throw runtime_error("fi_mr_reg failed");
}

void nervana::ofi::ofi::unregister_memory(rdma_memory& memory)
{
    fi_close(&(memory.m_mr_fid->fid));
    memory.m_mr_fid = nullptr;
}

void nervana::ofi::ofi::send(message& out_message)
{
    if (!m_initialized)
    {
        throw runtime_error("ofi is not initialized!");
    }

    uint64_t data_size{out_message.size()};
    uint64_t bytes_sent{0};
    uint8_t* message_buffer{reinterpret_cast<uint8_t*>(m_msg_tx_buf)};
    uint64_t to_send_size{rx_tx_buffer_size};

    ofi_header header;
    header.data_size = data_size;
    memcpy(message_buffer, &header, sizeof(ofi_header));
    message_buffer += sizeof(ofi_header);
    to_send_size -= sizeof(ofi_header);

    while (bytes_sent < data_size)
    {
        uint32_t to_send_current =
            ((data_size - bytes_sent) > to_send_size) ? to_send_size : (data_size - bytes_sent);
        void* buf_to_send = reinterpret_cast<void*>(out_message.buffer() + bytes_sent);

        memcpy(message_buffer, buf_to_send, to_send_current);
        int status = fi_send(
            m_fid_ep, m_msg_tx_buf, rx_tx_buffer_size, fi_mr_desc(m_msg_tx_mr), 0, m_msg_tx_buf);
        if (status)
        {
            throw runtime_error("fi_send failed");
        }

        uint16_t status_check_cnt = 0;
        do
        {
            fi_cq_entry entry;
            status = fi_cq_read(m_fid_cq_tx, &entry, 1);

            if (status == 1)
            {
                if (entry.op_context != m_msg_tx_buf)
                {
                    throw runtime_error("received completion form other part");
                }
                break;
            }

            if (status < 0 && status != -FI_EAGAIN)
            {
                if (status == -FI_EAVAIL && other_side_died(m_fid_cq_tx))
                {
                    throw runtime_error("other side died");
                }
                throw runtime_error("fi_cq_read failed");
            }

            if (status_check_cnt > status_check_frequency)
            {
                assure_connection_ok();
                status_check_cnt = 0;
            }
            status_check_cnt++;
        } while (status == -FI_EAGAIN);

        bytes_sent += to_send_current;

        message_buffer = m_msg_tx_buf;
        to_send_size   = rx_tx_buffer_size;
    }
}

void nervana::ofi::ofi::receive(message& in_message)
{
    if (!m_initialized)
    {
        throw runtime_error("ofi is not initialized");
    }

    void*    data{NULL};
    uint64_t data_size{0};
    uint64_t bytes_received{0};

    do
    {
        uint64_t to_receive_size = rx_tx_buffer_size;
        uint8_t* rx_buffer       = get_rx_buffer_shadow(m_current_rx_shadow_id);

        while (!receive_ready())
            ;

        if (bytes_received == 0)
        {
            ofi_header header = {0};
            memcpy(&header, rx_buffer, sizeof(header));
            to_receive_size -= sizeof(header);
            rx_buffer += sizeof(header);

            data_size = header.data_size;
            data      = in_message.allocate(data_size);
        }

        uint32_t len_to_recv = ((data_size - bytes_received) > to_receive_size)
                                   ? to_receive_size
                                   : (data_size - bytes_received);

        void* buf_to_recv = (void*)((uint64_t)data + (uint64_t)bytes_received);

        memcpy(buf_to_recv, rx_buffer, len_to_recv);
        m_current_rx_shadow_id = (m_current_rx_shadow_id + 1) % rx_buffers_number;

        m_receive_ready_cnt--;

        bytes_received += len_to_recv;
    } while (bytes_received < data_size);
}

bool nervana::ofi::ofi::receive_ready(int timeout_ms)
{
    if (!m_initialized)
    {
        throw runtime_error("ofi not initialized");
    }

    if (m_receive_ready_cnt > 0)
    {
        return true;
    }
    return handle_rx_queue(timeout_ms);
}

bool nervana::ofi::ofi::handle_rx_queue(int timeout_ms)
{
    uint16_t status_check_cnt{0};

    timeval start_time;
    gettimeofday(&start_time, NULL);

    while (true)
    {
        fi_cq_entry completions[rx_buffers_number];
        auto        completions_num = fi_cq_read(m_fid_cq_rx, &completions, rx_buffers_number);
        if (completions_num > 0)
        {
            for (int iter = 0; iter < completions_num; iter++)
            {
                uint8_t* buffer = reinterpret_cast<uint8_t*>(completions[iter].op_context);
                if (buffer < get_rx_buffer(0) || buffer > get_rx_buffer(rx_buffers_number - 1))
                {
                    throw runtime_error("wrong memory received");
                }

                ssize_t rx_id = (reinterpret_cast<uint64_t>(buffer) -
                                 reinterpret_cast<uint64_t>(get_rx_buffer(0))) /
                                rx_tx_buffer_size;
                memcpy(get_rx_buffer_shadow(rx_id), buffer, rx_tx_buffer_size);
                m_receive_ready_cnt++;

                if (fi_recv(
                        m_fid_ep, buffer, rx_tx_buffer_size, fi_mr_desc(m_msg_rx_mr), 0, buffer))
                {
                    throw runtime_error("fi_recv failed");
                }
            }
            return true;
        }
        else if (completions_num == -FI_EAGAIN)
        {
            if (status_check_cnt > status_check_frequency)
            {
                assure_connection_ok();
                status_check_cnt = 0;
            }
            status_check_cnt++;

            timeval current_time;
            gettimeofday(&current_time, NULL);
            long long time_elapsed = (current_time.tv_sec * 1000000LL + current_time.tv_usec) -
                                     (start_time.tv_sec * 1000000LL + start_time.tv_usec);

            if (time_elapsed > timeout_ms * 1000LL)
            {
                return false;
            }
            else if (time_elapsed > 10000LL)
            {
                usleep(1000);
            }
        }
        else
        {
            string error_message = ofi_cq_read_error(m_fid_cq_rx);
            throw runtime_error(error_message);
        }
    }
}

char* nervana::ofi::message::allocate(size_t size)
{
    void* memory{NULL};
    int   status{0};

    deallocate();
    m_size = size;

    if (m_size <= header_data_size)
    {
        memory = malloc(header_size);
    }
    else if (m_size <= (max_message_size - sizeof(uint64_t)))
    {
        memory = malloc(m_size + sizeof(uint64_t));
    }
    else
    {
        status =
            posix_memalign(&memory, detail::page_size, detail::ceil(m_size) + detail::page_size);
    }

    if (!memory || status)
    {
        throw(std::bad_alloc());
    }

    memset(memory, 0, m_size);
    // size written to header
    *(uint64_t*)memory = m_size;
    m_buffer           = (char*)memory;

    return buffer();
}

void nervana::ofi::message::deallocate()
{
    if (m_buffer)
    {
        free(m_buffer);
    }
    m_buffer = NULL;
}

char* nervana::ofi::message::buffer()
{
    if (m_size <= (max_message_size - sizeof(uint64_t)))
    {
        return m_buffer + sizeof(m_size);
    }
    return m_buffer + detail::page_size;
}

void nervana::ofi::message::copy(const message& other)
{
    allocate(other.m_size);
    memcpy(m_buffer, other.m_buffer, other.m_size);
    m_size = other.m_size;
}

void nervana::ofi::rdma_memory::allocate(size_t size)
{
    int status{0};

    deallocate();
    m_size = size;
    status = posix_memalign(&m_buffer, detail::page_size, detail::ceil(size) + detail::page_size);

    if (!m_buffer || status)
    {
        throw(std::bad_alloc());
    }

    memset(m_buffer, 0, m_size);
}

void nervana::ofi::rdma_memory::deallocate()
{
    if (m_buffer)
    {
        free(m_buffer);
        m_buffer = NULL;
    }
}

void nervana::ofi::rdma_memory::copy(const rdma_memory& other)
{
    allocate(other.m_size);
    memcpy(m_buffer, other.m_buffer, other.m_size);
    m_size = other.m_size;
}

uint64_t nervana::ofi::rdma_memory::get_key() const
{
    if (!m_mr_fid)
        throw runtime_error("cannot get key: memory is not registered");
    return m_mr_fid->key;
}

namespace
{
    void assert_ofi_success(int status, const string& message)
    {
        if (status)
            throw runtime_error(message + "; code: " + std::to_string(status));
    }

    bool other_side_died(fid_cq* queue)
    {
        fi_cq_err_entry err_entry;
        fi_cq_readerr(queue, &err_entry, 0);
        return err_entry.err == FI_EIO;
    }
}
