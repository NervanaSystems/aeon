#include <iostream>
#include <memory>
#include <cpprest/http_listener.h>

#include "main.hpp"
#include "json.hpp"
#include "loader.hpp"
#include "manifest_builder.hpp"
#include "gen_image.hpp"

#include "aeonsvc.hpp"

using namespace web::http;

static std::string create_manifest_file(size_t record_count, size_t width, size_t height)
{
    std::string      manifest_filename = nervana::file_util::tmp_filename();
    manifest_builder mb;
    auto& ms = mb.record_count(record_count).image_width(width).image_height(height).create();
    std::ofstream f(manifest_filename);
    f << ms.str();
    return manifest_filename;
}

default_config::default_config()
    : height{16}
    , width{16}
    , batch_size{32}
    , record_count{1003}
    , block_size{300}
    , js_image{{"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}}
    , label{{"type", "label"}, {"binary", false}}
    , augmentation{{{"type", "image"}, {"flip_enable", true}}}
{
    manifest_filename = create_manifest_file(record_count, width, height);

    js = {{"manifest_filename", manifest_filename},
          {"batch_size", batch_size},
          {"block_size", block_size},
          {"etl", {js_image, label}},
          {"augmentation", augmentation}};
}

pplx::task<void> aeon_server::open()
{
    return m_listener.open();
}

pplx::task<void> aeon_server::close()
{
    return m_listener.close();
}

aeon_server::aeon_server(utility::string_t url)
    : m_listener(url)
{
    m_listener.support(methods::GET, std::bind(&aeon_server::handle_get, this, std::placeholders::_1));
}



void aeon_server::handle_get(http_request message)
{
    std::cout << "Message received" << std::endl;

    try
    {
        std::shared_ptr<nervana::loader> loader = std::make_shared<nervana::loader_local>(config.js);
        size_t                           idx    = m_hash_fn(loader);

        m_aeon_clients[idx] = loader;

        web::json::value json_idx = web::json::value::object();
        json_idx["idx"]           = web::json::value::number(idx);
        message.reply(status_codes::OK, json_idx);
    }
    catch (std::exception& e)
    {
        std::cout << e.what();
    }
}

struct shutdown_deamon
{
    void operator()(aeon_server* server)
    {
        server->close().wait();
    }
};

void start_deamon()
{
    utility::string_t port = U("34568");
    utility::string_t http_addr = U("http://127.0.0.1:");
    utility::string_t path = U("aeon");

    http_addr.append(port);
    uri_builder uri(http_addr);
    uri.append_path(path);

    auto addr = uri.to_uri().to_string();

    static std::unique_ptr<aeon_server, shutdown_deamon> server (new aeon_server(addr));
    server->open().wait();
}

