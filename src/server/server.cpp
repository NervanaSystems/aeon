#include <iostream>
#include <memory>

#include "server.hpp"
#include "json.hpp"
#include "loader.hpp"
#include "manifest_builder.hpp"
#include "gen_image.hpp"

#include "base64.hpp"
#include "aeonsvc.hpp"

using namespace web::http;

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

agents_database::agents_database(const nlohmann::json& config)
    : m_config(config)
{
    m_dataloader = std::make_shared<nervana::loader_local>(m_config);
    m_id = get_dataset_seed();
}

const nervana::fixed_buffer_map& agents_database::next()
{
    return *m_dataloader->get_current_iter();
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

    agents_database ds(m_config.js);

    m_datasets.push_back(ds);

    uint64_t idx = ds.get_id();

    web::json::value json_idx = web::json::value::object();
    json_idx[keywords::id]           = web::json::value::number(idx);
    message.reply(status_codes::OK, json_idx);
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

    uint64_t id = std::stoull(json_id);
    auto it = std::find_if(m_datasets.begin(), m_datasets.end(), [id] (auto& a) -> bool { return a.get_id() == id; });

    if (it == m_datasets.end()) 
    {
        throw std::invalid_argument("Incorrect id value in message");
    }
    
    auto& data = it->next();

    std::vector<web::json::value> values;

    for (auto name : data.get_names())
    {
        auto elements = data[name];

        web::json::value value = web::json::value::object();
        value[keywords::name] = web::json::value::string(name);

        std::vector<char> encoded_data = nervana::base64::encode(elements->data(), elements->size());
        value[keywords::data] = web::json::value::string(std::string(std::begin(encoded_data), std::end(encoded_data)));

        values.push_back(value);
    }

    message.reply(status_codes::OK, web::json::value::array(values));

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

