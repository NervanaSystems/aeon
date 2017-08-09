#pragma once

#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <mutex>

#include <cpprest/http_listener.h>

#include "json.hpp"
#include "loader.hpp"
#include "manifest_builder.hpp"
#include "gen_image.hpp"

namespace nervana
{
    struct default_config
    {
        size_t         height;
        size_t         width;
        size_t         batch_size;
        size_t         record_count;
        size_t         block_size;
        std::string    manifest_filename;
        nlohmann::json js_image;
        nlohmann::json label;
        nlohmann::json augmentation;
        nlohmann::json js;
        default_config();
    };

    class loader_adapter
    {
    public:
        loader_adapter(const nlohmann::json& config)
            : m_loader(config)
        {
        }
        void        reset();
        bool        next();
        std::string data();
        std::string position();
        std::string batch_size();
        std::string names_and_shapes();
        std::string batch_count();
        std::string record_count();

    private:
        nervana::loader_local m_loader;
    };

    class loader_manager
    {
    public:
        loader_manager()
            : m_id_generator(std::random_device{}())
        {
        }
        uint32_t                 register_agent(const nlohmann::json& config);
        void                     unregister_agent(uint32_t);
        nervana::loader_adapter& loader(uint32_t id);

    private:
        const uint32_t max_loader_number = 1000;
        std::mt19937   m_id_generator;
        std::mutex     m_mutex;
        std::map<uint32_t, std::unique_ptr<nervana::loader_adapter>> m_loaders;
    };

    class server_parser
    {
    public:
        server_parser();
        web::json::value post(std::string msg, std::string msg_body);
        web::json::value get(std::string msg);
        web::json::value del(std::string msg);

    private:
        const std::string version         = "v1";
        const std::string endpoint_prefix = "/" + version + "/dataset";
        typedef web::json::value (server_parser::*msg_process_func_t)(loader_adapter&);
        std::map<std::string, server_parser::msg_process_func_t> process_func;

        loader_manager m_loader_manager;

        web::json::value next(loader_adapter& loader);
        web::json::value batch_size(loader_adapter& loader);
        web::json::value reset(loader_adapter& loader);
        web::json::value names_and_shapes(loader_adapter& loader);
        web::json::value batch_count(loader_adapter& loader);
        web::json::value record_count(loader_adapter& loader);
    };

    class aeon_server
    {
    public:
        aeon_server(utility::string_t url);

        pplx::task<void> open();
        pplx::task<void> close();

    private:
        void handle_post(web::http::http_request message);
        void handle_get(web::http::http_request message);
        void handle_delete(web::http::http_request message);

        web::http::experimental::listener::http_listener m_listener;
        server_parser                                    m_server_parser;
    };
}
