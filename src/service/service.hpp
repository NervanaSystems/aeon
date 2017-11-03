#pragma once

#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <mutex>
#include <tuple>

#include <cpprest/http_listener.h>

#include "json.hpp"
#include "loader.hpp"
#include "log.hpp"
#if defined(ENABLE_OPENFABRICS_CONNECTOR)
#include "../rdma/ofi.hpp"
#endif
#include "../client/http_connector.hpp"

namespace nervana
{
    namespace aeon
    {
        template <typename T>
        class statused_response
        {
        public:
            statused_response(web::http::status_code _status_code, const T& _value)
                : status_code(_status_code)
                , value(_value)
            {
            }

            web::http::status_code status_code;
            T                      value;
        };

        using json_response = statused_response<web::json::value>;
        using next_tuple    = std::tuple<web::json::value, std::string>;

        class loader_adapter
        {
        public:
            explicit loader_adapter(const nlohmann::json& config)
                : m_loader(config)
            {
            }

            void reset();

            std::string next();

            std::string batch_size() const;
            std::string names_and_shapes() const;
            std::string batch_count() const;
            std::string record_count() const;

        private:
            nervana::loader_local m_loader;
            std::mutex            m_mutex;
            bool                  m_is_reset{true};
        };

        class loader_manager
        {
        public:
            loader_manager()
                : m_id_generator(std::random_device{}())
            {
            }

            uint32_t register_agent(const nlohmann::json& config);
            void                  unregister_agent(uint32_t);
            aeon::loader_adapter& loader(uint32_t id);

        private:
            const uint32_t max_loader_number = 1000;
            std::mt19937   m_id_generator;
            std::mutex     m_mutex;
            std::map<uint32_t, std::unique_ptr<aeon::loader_adapter>> m_loaders;
        };

        class parser
        {
        public:
            parser();

            json_response post(const std::string& msg, const std::string& msg_body);
            json_response del(const std::string& msg);
            statused_response<next_tuple> get(const std::string& msg);

        private:
            const std::string version         = "v1";
            const std::string endpoint_prefix = "/" + version + "/dataset";

            using msg_process_func_t = std::function<web::json::value(loader_adapter&)>;

            std::map<std::string, msg_process_func_t> process_func;
            loader_manager m_loader_manager;

            statused_response<next_tuple> next(loader_adapter& loader);

            web::json::value batch_size(loader_adapter& loader);
            web::json::value reset(loader_adapter& loader);
            web::json::value names_and_shapes(loader_adapter& loader);
            web::json::value batch_count(loader_adapter& loader);
            web::json::value record_count(loader_adapter& loader);
        };

#if defined(ENABLE_OPENFABRICS_CONNECTOR)

        class ofi_connection_pool
        {
        public:
            ofi_connection_pool(const std::string& address, unsigned int port);
            ofi_connection_pool() = delete;
            ~ofi_connection_pool();

            ofi::ofi& get_ofi(const std::string& connection_id);
            ofi::rdma_memory& get_rdma_memory(const std::string& connection_id);

            void register_memory(const std::string& connection_id, size_t size);
            void unregister_memory(const std::string& connection_id);

            void remove_connection(const std::string& connection_id);

        private:
            struct ofi_connection
            {
                ofi::ofi         _ofi;
                ofi::rdma_memory _rdma_memory;
            };

            ofi_connection& get_ofi_connection(const std::string& connection_id);
            void accept_connections();

            std::atomic<bool> m_finish{false};
            int               m_connection_id_counter{1};
            std::map<std::string, ofi_connection> m_connections;
            std::unique_ptr<std::thread> m_thread;
            std::recursive_mutex         m_connections_mutex;
            ofi::ofi                     m_ofi;
        };
#endif

        class service
        {
        public:
            explicit service(const web::http::uri& uri
#if defined(ENABLE_OPENFABRICS_CONNECTOR)
                             ,
                             std::string rdma_addr = ""
#endif
                             )
                : m_listener{web::http::uri_builder{uri}.append_path(U("api")).to_uri()}
            {
#if defined(ENABLE_OPENFABRICS_CONNECTOR)
                if (rdma_addr.size() > 1)
                {
                    std::string::size_type delimiter{rdma_addr.find(':')};
                    std::string            rdma_address = rdma_addr.substr(0, delimiter);
                    unsigned int           rdma_port = std::stoi(rdma_addr.substr(delimiter + 1));
                    ofi_connection_pool*   connections{nullptr};
                    try
                    {
                        connections = new ofi_connection_pool(rdma_address, rdma_port);
                    }
                    catch (const std::exception& ex)
                    {
                        log::error(
                            "Couldn't start listening on %s:%d: %s. HTTP transport will be used.",
                            rdma_address,
                            rdma_port,
                            ex.what());
                    }
                    m_ofi_connections = std::unique_ptr<ofi_connection_pool>(connections);
                }
#endif
                m_listener.support(web::http::methods::POST,
                                   std::bind(&service::handle_post, this, std::placeholders::_1));
                m_listener.support(web::http::methods::GET,
                                   std::bind(&service::handle_get, this, std::placeholders::_1));
                m_listener.support(web::http::methods::DEL,
                                   std::bind(&service::handle_delete, this, std::placeholders::_1));

                m_listener.open().wait();
            }

            service() = delete;

            service(const service&) = delete;
            service& operator=(const service&) = delete;

            service(service&&) = delete;
            service& operator=(service&&) = delete;

            ~service() { m_listener.close().wait(); }
            const web::http::uri& uri() const { return m_listener.uri(); }
        private:
            web::http::experimental::listener::http_listener m_listener;
            parser                                           m_parser;

#if defined(ENABLE_OPENFABRICS_CONNECTOR)
            std::unique_ptr<ofi_connection_pool> m_ofi_connections;
            void reply_data_with_ofi(web::http::http_request&       request,
                                     const std::string&             query,
                                     statused_response<next_tuple>& reply);
#endif
            void handle_post(web::http::http_request r)
            {
                log::info("POST %s", r.relative_uri().path());
                auto answer = m_parser.post(web::uri::decode(r.relative_uri().path()),
                                            r.extract_string().get());
                r.reply(answer.status_code, answer.value);
            }

            void handle_get(web::http::http_request message);
            void handle_delete(web::http::http_request r);
        };
    }
}
