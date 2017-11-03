#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "service.hpp"
#include "json.hpp"
#include "loader.hpp"
#include "typemap.hpp"

#include "base64.hpp"

using namespace web::http;
using nlohmann::json;
using namespace std;
using namespace nervana;

namespace nervana
{
    namespace aeon
    {
        namespace
        {
            constexpr char not_found_loader[] = "loader doesn't exist";

            web::json::value not_found_json()
            {
                web::json::value response_json         = web::json::value::object();
                response_json["status"]["type"]        = web::json::value::string("FAILURE");
                response_json["status"]["description"] = web::json::value::string("not found");
                return response_json;
            }

#if defined(ENABLE_OPENFABRICS_CONNECTOR)
            web::json::value failure_json(const string& description)
            {
                web::json::value reply_json         = web::json::value::object();
                reply_json["status"]["type"]        = web::json::value::string("FAILURE");
                reply_json["status"]["description"] = web::json::value::string(description);
                return reply_json;
            }
#endif
        }

        uint32_t loader_manager::register_agent(const nlohmann::json& config)
        {
            lock_guard<mutex> lg(m_mutex);

            if (m_loaders.size() >= max_loader_number - 1)
                throw std::runtime_error("the number of loaders exceeded max value");

            uint32_t id;
            while (m_loaders.find(id = (m_id_generator() % max_loader_number)) != m_loaders.end())
            {
            };

            m_loaders[id] = std::unique_ptr<aeon::loader_adapter>(new aeon::loader_adapter(config));
            log::info("Created new session with ID %d", id);

            return id;
        }

        void loader_manager::unregister_agent(uint32_t id)
        {
            lock_guard<mutex> lg(m_mutex);

            auto it = m_loaders.find(id);
            if (it == m_loaders.end())
                throw invalid_argument(not_found_loader);
            m_loaders.erase(it);
        }

        aeon::loader_adapter& loader_manager::loader(uint32_t id)
        {
            lock_guard<mutex> lg(m_mutex);

            auto it = m_loaders.find(id);
            if (it == m_loaders.end())
                throw invalid_argument(not_found_loader);
            return *it->second.get();
        }

        void service::handle_get(http_request request)
        {
            string path = web::uri::decode(request.relative_uri().path());
            log::info("[GET] %s", path);
            auto                          query = uri::split_query(request.request_uri().query());
            statused_response<next_tuple> reply = m_parser.get(path);
            if (std::get<1>(reply.value).empty())
                request.reply(reply.status_code, get<0>(reply.value));
            else
            {
#if !defined(ENABLE_OPENFABRICS_CONNECTOR)
                request.reply(reply.status_code, get<1>(reply.value));
#else
                // batch data sending
                string connection_id = query["connection_id"];
                if (connection_id.empty() || reply.status_code != http::status_ok)
                {
                    request.reply(reply.status_code, get<1>(reply.value));
                }
                else
                {
                    reply_data_with_ofi(request, connection_id, reply);
                }
#endif /* ENABLE_OPENFABRICS_CONNECTOR */
            }
        }

        void service::handle_delete(http_request request)
        {
            log::info("DELETE %s", request.relative_uri().path());
            auto   query         = uri::split_query(request.request_uri().query());
            string connection_id = query["connection_id"];

            auto answer = m_parser.del(web::uri::decode(request.relative_uri().path()));
#if defined(ENABLE_OPENFABRICS_CONNECTOR)
            if (!connection_id.empty() && answer.status_code == status_codes::OK)
            {
                m_ofi_connections->remove_connection(connection_id);
            }
#endif /* ENABLE_OPENFABRICS_CONNECTOR */

            request.reply(answer.status_code, answer.value);
        }

#if defined(ENABLE_OPENFABRICS_CONNECTOR)
        void service::reply_data_with_ofi(http_request&                  request,
                                          const string&                  connection_id,
                                          statused_response<next_tuple>& reply)
        {
            if (m_ofi_connections == nullptr)
            {
                static const string error_msg =
                    "cannot respond via RDMA, because RDMA connections are not initialized";
                request.reply(status_codes::InternalError, error_msg);
                log::warning(error_msg);
                return;
            }

            const string&     data   = get<1>(reply.value);
            ofi::rdma_memory& memory = m_ofi_connections->get_rdma_memory(connection_id);
            if (!memory.is_registered())
            {
                m_ofi_connections->register_memory(connection_id, data.size());
            }
            if (memory.is_allocated() && memory.get_buffer_size() < data.size())
            {
                static const string error_msg = "allocated RDMA buffer is too small";
                log::warning(error_msg);
                request.reply(status_codes::InternalError, failure_json(error_msg));
                return;
            }

            std::memcpy(memory.get_buffer(), data.c_str(), data.size());

            web::json::value reply_json   = web::json::value::object();
            reply_json["status"]["type"]  = web::json::value::string("SUCCESS");
            reply_json["data"]["address"] = reinterpret_cast<uint64_t>(memory.get_buffer());
            reply_json["data"]["size"]    = memory.get_buffer_size();
            reply_json["data"]["key"]     = memory.get_key();
            request.reply(status_codes::OK, reply_json);
        }
#endif /* ENABLE_OPENFABRICS_CONNECTOR */

        // /////////////////////////////////////////////////////////////////////////////

        string loader_adapter::next()
        {
            unique_lock<mutex> lock(m_mutex);

            if (m_is_reset)
                m_is_reset = false;
            else
                m_loader.get_current_iter()++;

            if (m_loader.get_current_iter().positional_end())
            {
                return string("");
            }
            else
            {
                // stringstream initialization is very costly, so we have to avoid it by using thread_local.
                // It has a drawback though that of increased memory consumption.
                // If this is a problem, we can consider using sprintf and allocating memory on each request.
                thread_local std::ostringstream ss;
                auto                            iter = m_loader.get_current_iter();
                lock.unlock();
                ss.seekp(ios::beg);
                ss << *iter;
                return ss.str();
            }
        };

        void loader_adapter::reset()
        {
            lock_guard<mutex> lg(m_mutex);
            m_loader.reset();
            m_is_reset = true;
        };

        string loader_adapter::batch_size() const { return std::to_string(m_loader.batch_size()); };
        string loader_adapter::names_and_shapes() const
        {
            std::stringstream ss;
            ss << m_loader.get_names_and_shapes();
            return string(ss.str());
        };

        string loader_adapter::batch_count() const
        {
            return std::to_string(m_loader.batch_count());
        }

        string loader_adapter::record_count() const
        {
            return std::to_string(m_loader.record_count());
        }

// ////////////////////////////////////////////////////////////////////////////

#define REGISTER_SRV_FUNCTION(f)                                                                   \
    process_func[#f] = std::bind(&parser::f, this, std::placeholders::_1)

        parser::parser()
        {
            REGISTER_SRV_FUNCTION(batch_size);
            REGISTER_SRV_FUNCTION(reset);
            REGISTER_SRV_FUNCTION(names_and_shapes);
            REGISTER_SRV_FUNCTION(batch_count);
            REGISTER_SRV_FUNCTION(record_count);
        }

        json_response parser::post(const std::string& msg, const std::string& msg_body)
        {
            try
            {
                if (msg.substr(0, endpoint_prefix.length()) != endpoint_prefix)
                    throw std::invalid_argument("Invalid prefix");
                web::json::value reply_json  = web::json::value::object();
                reply_json["status"]["type"] = web::json::value::string("SUCCESS");
                reply_json["data"]["id"]     = web::json::value::string(
                    std::to_string(m_loader_manager.register_agent(json::parse(msg_body))));
                return json_response(status_codes::Accepted, reply_json);
            }
            catch (exception& ex)
            {
                string ex_msg(ex.what());
                if (ex_msg.find(not_found_loader) != string::npos)
                    return json_response(status_codes::NotFound, not_found_json());
                web::json::value reply_json         = web::json::value::object();
                reply_json["status"]["type"]        = web::json::value::string("FAILURE");
                reply_json["status"]["description"] = web::json::value::string(ex.what());
                if (ex_msg.find("error when parsing") != string::npos) // manifest cannot be parsed
                    return json_response(status_codes::BadRequest, reply_json);
                else
                    return json_response(status_codes::InternalError, reply_json);
            }
        }

        statused_response<next_tuple> parser::get(const std::string& msg)
        {
            static const auto notFound = statused_response<next_tuple>(
                status_codes::NotFound, tuple<web::json::value, std::string>(not_found_json(), ""));
            try
            {
                if (msg.substr(0, endpoint_prefix.length()) != endpoint_prefix)
                    return notFound;

                string path = msg;
                path.erase(0, endpoint_prefix.length() + 1);

                auto paths = web::uri::split_path(path);
                if (paths.size() != 2)
                    return notFound;

                int dataset_id = std::stoi(paths[0]);

                if (paths[1] == "next")
                {
                    return next(m_loader_manager.loader(dataset_id));
                }
                else
                {
                    auto it = process_func.find(paths[1]);
                    if (it == process_func.end())
                        return notFound;
                    else
                        return statused_response<next_tuple>(
                            status_codes::OK,
                            make_tuple((it->second)(m_loader_manager.loader(dataset_id)), ""));
                }
            }
            catch (exception& ex)
            {
                string ex_msg(ex.what());
                if (ex_msg.find(not_found_loader) != string::npos)
                    return notFound;
                web::json::value response_json         = web::json::value::object();
                response_json["status"]["type"]        = web::json::value::string("FAILURE");
                response_json["status"]["description"] = web::json::value::string(ex_msg);
                return statused_response<next_tuple>(status_codes::InternalError,
                                                     make_tuple(response_json, ""));
            }
        }

        json_response parser::del(const std::string& msg)
        {
            try
            {
                if (msg.substr(0, endpoint_prefix.length()) != endpoint_prefix)
                {
                    return {status_codes::NotFound, not_found_json()};
                }
                string query = msg;
                query.erase(0, endpoint_prefix.length() + 1);

                int dataset_id = std::stoi(query);
                m_loader_manager.unregister_agent(dataset_id);

                web::json::value reply_json{web::json::value::object()};
                reply_json["status"]["type"] = web::json::value::string("SUCCESS");
                return {status_codes::OK, reply_json};
            }
            catch (exception& ex)
            {
                string ex_msg(ex.what());
                if (ex_msg.find(not_found_loader) != string::npos)
                    return json_response(status_codes::NotFound, not_found_json());
                web::json::value response_json         = web::json::value::object();
                response_json["status"]["type"]        = web::json::value::string("FAILURE");
                response_json["status"]["description"] = web::json::value::string(ex.what());
                return {status_codes::InternalError, response_json};
            }
        }

        statused_response<next_tuple> parser::next(loader_adapter& loader)
        {
            web::json::value response_json = web::json::value::object();
            string           data          = loader.next();

            if (!data.empty())
            {
                response_json["status"]["type"] = web::json::value::string("SUCCESS");
                return statused_response<next_tuple>(status_codes::OK,
                                                     make_tuple(response_json, data));
            }
            else
            {
                response_json["status"]["type"] = web::json::value::string("END_OF_DATASET");
                return statused_response<next_tuple>(status_codes::NotFound,
                                                     make_tuple(response_json, ""));
            }
        }

        web::json::value parser::reset(loader_adapter& loader)
        {
            web::json::value response_json = web::json::value::object();

            loader.reset();
            response_json["status"]["type"] = web::json::value::string("SUCCESS");

            return response_json;
        }

        web::json::value parser::batch_size(loader_adapter& loader)
        {
            web::json::value response_json = web::json::value::object();

            response_json["status"]["type"]     = web::json::value::string("SUCCESS");
            response_json["data"]["batch_size"] = web::json::value::string(loader.batch_size());
            return response_json;
        }

        web::json::value parser::batch_count(loader_adapter& loader)
        {
            web::json::value response_json = web::json::value::object();

            response_json["status"]["type"]      = web::json::value::string("SUCCESS");
            response_json["data"]["batch_count"] = web::json::value::string(loader.batch_count());
            return response_json;
        }

        web::json::value parser::record_count(loader_adapter& loader)
        {
            web::json::value response_json = web::json::value::object();

            response_json["status"]["type"]       = web::json::value::string("SUCCESS");
            response_json["data"]["record_count"] = web::json::value::string(loader.record_count());
            return response_json;
        }

        web::json::value parser::names_and_shapes(loader_adapter& loader)
        {
            web::json::value response_json = web::json::value::object();

            response_json["status"]["type"] = web::json::value::string("SUCCESS");
            response_json["data"]["names_and_shapes"] =
                web::json::value::string(loader.names_and_shapes());
            return response_json;
        }

#if defined(ENABLE_OPENFABRICS_CONNECTOR)

        ofi_connection_pool::ofi_connection_pool(const string& address, unsigned int port)
        {
            m_ofi.bind_and_listen(port, address);
            m_thread =
                unique_ptr<std::thread>(new thread(&ofi_connection_pool::accept_connections, this));
            log::info("RDMA listening on %s:%d...", m_ofi.get_address(), m_ofi.get_port());
        }

        ofi_connection_pool::~ofi_connection_pool()
        {
            // stop thread waiting for new connections
            m_finish.store(true);
            m_thread->join();

            // unregister rdma memory
            for (auto& item : m_connections)
            {
                ofi_connection& connection = item.second;
                connection._ofi.unregister_memory(connection._rdma_memory);
            }
        }

        nervana::ofi::ofi& ofi_connection_pool::get_ofi(const string& connection_id)
        {
            lock_guard<recursive_mutex> lock(m_connections_mutex);
            return get_ofi_connection(connection_id)._ofi;
        }

        nervana::ofi::rdma_memory& ofi_connection_pool::get_rdma_memory(const string& connection_id)
        {
            lock_guard<recursive_mutex> lock(m_connections_mutex);
            return get_ofi_connection(connection_id)._rdma_memory;
        }

        void ofi_connection_pool::register_memory(const string& connection_id, size_t size)
        {
            lock_guard<recursive_mutex> lock(m_connections_mutex);
            ofi_connection&             connection = get_ofi_connection(connection_id);
            ofi::rdma_memory&           memory     = connection._rdma_memory;
            if (memory.is_allocated())
            {
                if (memory.get_buffer_size() < size)
                    throw std::runtime_error("allocated RDMA memory size is too small!");
            }
            else
            {
                memory.allocate(size);
            }
            if (!memory.is_registered())
            {
                connection._ofi.register_memory(memory);
            }
        }

        void ofi_connection_pool::unregister_memory(const string& connection_id)
        {
            lock_guard<recursive_mutex> lock(m_connections_mutex);
            ofi_connection&             connection = get_ofi_connection(connection_id);
            ofi::rdma_memory&           memory     = connection._rdma_memory;

            if (memory.is_registered())
                connection._ofi.unregister_memory(memory);
            memory.deallocate();
        }

        ofi_connection_pool::ofi_connection&
            ofi_connection_pool::get_ofi_connection(const string& connection_id)
        {
            try
            {
                return m_connections.at(connection_id);
            }
            catch (const std::out_of_range&)
            {
                throw invalid_argument("there is no such connection id");
            }
        }

        void ofi_connection_pool::remove_connection(const std::string& connection_id)
        {
            lock_guard<recursive_mutex> lock(m_connections_mutex);
            unregister_memory(connection_id);
            m_connections.erase(connection_id);
        }

        void ofi_connection_pool::accept_connections()
        {
            const int       timeout_ms      = 100;
            string          id_as_string    = std::to_string(m_connection_id_counter);
            ofi_connection& next_connection = m_connections[id_as_string];
            ofi::ofi*       ofi_connection  = &next_connection._ofi;
            while (!m_finish.load())
            {
                if (m_ofi.wait_for_connect(*ofi_connection, timeout_ms))
                {
                    log::info("new ofi connection has been established with ID %s", id_as_string);
                    ofi::message msg;
                    msg.allocate(id_as_string.size());
                    memcpy(msg.buffer(), id_as_string.c_str(), id_as_string.size());
                    ofi_connection->send(msg);
                    m_connection_id_counter++;
                    id_as_string = std::to_string(m_connection_id_counter);
                    m_connections_mutex.lock();
                    ofi_connection = &m_connections[id_as_string]._ofi;
                    m_connections_mutex.unlock();
                }
            }
        }

#endif
    }
}
