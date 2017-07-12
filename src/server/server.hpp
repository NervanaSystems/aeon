#pragma once

#include <iostream>
#include <memory>
#include <chrono>
#include <random>

#include <cpprest/http_listener.h>

#include "json.hpp"
#include "loader.hpp"
#include "manifest_builder.hpp"
#include "gen_image.hpp"

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

static uint64_t get_dataset_seed()
{
    static uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    static std::mt19937 m_id_generator(seed);

    return m_id_generator();
}

class agents_database
{
public:
    agents_database(const nlohmann::json& config);
    uint64_t get_id() { return m_id; }

    const nervana::fixed_buffer_map& next();

private:
    uint64_t m_id;
    nlohmann::json m_config;

    std::shared_ptr<nervana::loader_local> m_dataloader;
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
    default_config m_config;

    std::vector<agents_database> m_datasets;
};

