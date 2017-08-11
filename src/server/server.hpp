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
            : m_loader(config), m_is_reset(true)
        {
        }
        void        reset();
        std::string next();

        std::string batch_size() const;
        std::string names_and_shapes() const;
        std::string batch_count() const;
        std::string record_count() const;

    private:
        nervana::loader_local m_loader;
        std::mutex            m_mutex;
        bool                  m_is_reset;
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
        web::json::value del(std::string msg);
        std::tuple<web::json::value, std::string> get(std::string msg);

    private:
        const std::string version         = "v1";
        const std::string endpoint_prefix = "/" + version + "/dataset";
        typedef web::json::value (server_parser::*msg_process_func_t)(loader_adapter&);
        std::map<std::string, server_parser::msg_process_func_t> process_func;

        loader_manager m_loader_manager;

        std::tuple<web::json::value, std::string> next(loader_adapter& loader);
        web::json::value batch_size(loader_adapter& loader);
        web::json::value reset(loader_adapter& loader);
        web::json::value names_and_shapes(loader_adapter& loader);
        web::json::value batch_count(loader_adapter& loader);
        web::json::value record_count(loader_adapter& loader);
    };

    class aeon_server
    {
    public:
        aeon_server(std::string http_addr);
        ~aeon_server();

    private:
        void handle_post(web::http::http_request message);
        void handle_get(web::http::http_request message);
        void handle_delete(web::http::http_request message);
        
        static utility::string_t path;

        std::unique_ptr<web::http::experimental::listener::http_listener> m_listener;
        server_parser                                                     m_server_parser;
    };
}
