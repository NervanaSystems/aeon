#include <iostream>
#include <memory>
#include <sstream>

#include "server.hpp"
#include "json.hpp"
#include "loader.hpp"
#include "manifest_builder.hpp"
#include "gen_image.hpp"

#include "base64.hpp"
#include "aeonsvc.hpp"

using namespace web::http;
using nlohmann::json;
using namespace std;

namespace keywords
{
    static const std::string next = "next";
    static const std::string dataset = "dataset";
    static const std::string version = "v1";

    static const std::string id = "id";
    static const std::string data = "data";
    static const std::string name = "name";
}

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

uint32_t loader_manager::register_agent(const nlohmann::json& config)
{
    lock_guard<mutex> lg(m_mutex);
    
    if (m_loaders.size() >= max_loader_number - 1)
        throw std::runtime_error("the number of loaders exceeded");
    
    uint32_t id;
    while( m_loaders.find(id = (m_id_generator() % max_loader_number)) != m_loaders.end())
    {};
    
    m_loaders[id] = std::make_unique<nervana::loader_local>(config);
    
    return id;
}

const nervana::fixed_buffer_map& loader_manager::next(uint32_t id)
{
    auto it = m_loaders.find(id);
    if (it == m_loaders.end())
        throw invalid_argument("loader doesn't exist");
    
    auto& loader = *it->second;
    loader.get_current_iter()++;
    return *loader.get_current_iter();
}

aeon_server::aeon_server(utility::string_t url)
    : m_listener(url)
{
    m_listener.support(methods::POST, std::bind(&aeon_server::handle_post, this, std::placeholders::_1));
    m_listener.support(methods::GET, std::bind(&aeon_server::handle_get, this, std::placeholders::_1));
}

pplx::task<void> aeon_server::open()
{
    return m_listener.open();
}

pplx::task<void> aeon_server::close()
{
    return m_listener.close();
}

void aeon_server::handle_post(http_request message)
{
    auto path = web::uri::split_path(web::uri::decode(message.relative_uri().path()));

    //Path should have format v1/dataset
    if (path.size() != 2)
    {
        throw std::invalid_argument("Incorrect number of elements in path");
    }

    utility::string_t ver = path[0];
    utility::string_t dataset_kw = path[1];

    if (ver != keywords::version) 
    {
        throw std::invalid_argument("Incorrect version value");
    }

    if (dataset_kw != keywords::dataset)
    {
        throw std::invalid_argument("Incorrect dataset keyword");
    }

    web::json::value   reply_json  = web::json::value::object();
    
    uint32_t id = 0;
    try
    {
        id = m_loader_manager.register_agent(m_config.js);
    }
    catch (exception& ex)
    {
        reply_json["status"]["type"]        = web::json::value::string("FAILURE");
        reply_json["status"]["description"] = web::json::value::string(ex.what());
        message.reply(status_codes::Accepted, reply_json);
        return;
    }
    
    reply_json["status"]["type"] = web::json::value::string("SUCCESS");
    reply_json["data"]["id"]     = web::json::value::string(to_string(id));
    //reply_json["data"]["id"]     = web::json::value::number(id);
    message.reply(status_codes::Accepted, reply_json);
}

void aeon_server::handle_get(http_request message)
{
    auto path = web::uri::split_path(web::uri::decode(message.relative_uri().path()));

    //Path should have format v1/next/<id>
    if (path.size() != 3)
    {
        throw std::invalid_argument("Incorrect number of elements in path");
    }

    utility::string_t ver = path[0];
    utility::string_t next_kw = path[1];
    utility::string_t json_id = path[2];

    if (ver != keywords::version) 
    {
        throw std::invalid_argument("Incorrect version value");
    }

    if (next_kw != keywords::next)
    {
        throw std::invalid_argument("Incorrect dataset keyword");
    }

    uint32_t id = std::stoull(json_id);
    
    std::stringstream ss;
    ss << m_loader_manager.next(id);
    
    std::vector<char> encoded_data = nervana::base64::encode(ss.str().data(), ss.str().size());
    web::json::value value = web::json::value::object();
    value[keywords::data] = web::json::value::string(std::string(encoded_data.begin(), encoded_data.end()));
    
    message.reply(status_codes::OK, value);
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
    utility::string_t path = U("api");

    http_addr.append(port);
    uri_builder uri(http_addr);
    uri.append_path(path);

    auto addr = uri.to_uri().to_string();

    static std::unique_ptr<aeon_server, shutdown_deamon> server (new aeon_server(addr));
    server->open().wait();
}

