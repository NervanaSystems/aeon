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
    static std::string create_manifest_file(size_t record_count, size_t width, size_t height);

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

    class loader_manager
    {
    public:
        loader_manager()
            : m_id_generator(std::random_device{}())
        {
        }
        uint32_t register_agent(const nlohmann::json& config);
        nervana::loader_local& loader(uint32_t id);

    private:
        const uint32_t max_loader_number = 1000;
        std::mt19937   m_id_generator;
        std::mutex     m_mutex;
        std::map<uint32_t, std::unique_ptr<nervana::loader_local>> m_loaders;
    };

    class server_message_process
    {
    public:
        std::string register_agent();

        std::string next(uint32_t id);
        std::string batch_size(uint32_t id);
        std::string reset(uint32_t id);
        std::string names_and_shapes(uint32_t id);
        std::string batch_count(uint32_t id);
        std::string record_count(uint32_t id);

    private:
        loader_manager m_loader_manager;
        default_config m_config;
    };

    class server_parser
    {
    public:
        server_parser();
        web::json::value post(std::string msg);
        web::json::value get(std::string msg);

    private:
        const std::string version         = "v1";
        const std::string endpoint_prefix = "/" + version + "/dataset";
        typedef web::json::value (server_parser::*msg_process_func_t)(uint32_t);
        std::map<std::string, server_parser::msg_process_func_t> process_func;
        server_message_process srv;

        web::json::value next(uint32_t id);
        web::json::value batch_size(uint32_t id);
        web::json::value reset(uint32_t id);
        web::json::value names_and_shapes(uint32_t id);
        web::json::value batch_count(uint32_t id);
        web::json::value record_count(uint32_t id);
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

        web::http::experimental::listener::http_listener m_listener;
        server_parser                                    m_server_parser;
    };
}
