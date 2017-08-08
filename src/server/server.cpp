#include <iostream>
#include <memory>
#include <sstream>

#include "server.hpp"
#include "json.hpp"
#include "loader.hpp"
#include "manifest_builder.hpp"
#include "gen_image.hpp"
#include "typemap.hpp"

#include "base64.hpp"
#include "aeonsvc.hpp"

using namespace web::http;
using nlohmann::json;
using namespace std;
using namespace nervana;

static std::string nervana::create_manifest_file(size_t record_count, size_t width, size_t height)
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
    while (m_loaders.find(id = (m_id_generator() % max_loader_number)) != m_loaders.end())
    {
    };

    m_loaders[id] = std::make_unique<nervana::loader_adapter>(config);

    INFO << "Created new session " << id;

    return id;
}

void loader_manager::unregister_agent(uint32_t id)
{
    lock_guard<mutex> lg(m_mutex);

    auto it = m_loaders.find(id);
    if (it == m_loaders.end())
        throw invalid_argument("loader doesn't exist");
    m_loaders.erase(it);
}

nervana::loader_adapter& loader_manager::loader(uint32_t id)
{
    auto it = m_loaders.find(id);
    if (it == m_loaders.end())
        throw invalid_argument("loader doesn't exist");
    return *it->second.get();
}

aeon_server::aeon_server(utility::string_t url)
    : m_listener(url)
{
    m_listener.support(methods::POST,
                       std::bind(&aeon_server::handle_post, this, std::placeholders::_1));
    m_listener.support(methods::GET,
                       std::bind(&aeon_server::handle_get, this, std::placeholders::_1));
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
    INFO << "[POST] " << message.relative_uri().path();
    message.reply(status_codes::Accepted,
                  m_server_parser.post(web::uri::decode(message.relative_uri().path()),
                                       message.extract_string().get()));
}

void aeon_server::handle_get(http_request message)
{
    INFO << "[GET] " << message.relative_uri().path();
    message.reply(status_codes::OK,
                  m_server_parser.get(web::uri::decode(message.relative_uri().path())));
}

// /////////////////////////////////////////////////////////////////////////////
bool loader_adapter::next()
{
    m_loader.get_current_iter()++;
    return !m_loader.get_current_iter().positional_end();
};

string loader_adapter::data()
{
    std::stringstream ss;
    ss << *m_loader.get_current_iter();
    std::vector<char> encoded_data = nervana::base64::encode(ss.str().data(), ss.str().size());
    return std::string(encoded_data.begin(), encoded_data.end());
}

string loader_adapter::position()
{
    return to_string(m_loader.position());
}

string loader_adapter::batch_size()
{
    return to_string(m_loader.batch_size());
};

void loader_adapter::reset()
{
    m_loader.reset();
};

string loader_adapter::names_and_shapes()
{
    std::stringstream ss;
    ss << m_loader.get_names_and_shapes();
    return string(ss.str());
};

string loader_adapter::batch_count()
{
    return to_string(m_loader.batch_count());
}

string loader_adapter::record_count()
{
    return to_string(m_loader.record_count());
}

// ////////////////////////////////////////////////////////////////////////////

#define REGISTER_SRV_FUNCTION(f) process_func[#f] = &server_parser::f;

server_parser::server_parser()
{
    REGISTER_SRV_FUNCTION(next);
    REGISTER_SRV_FUNCTION(batch_size);
    REGISTER_SRV_FUNCTION(reset);
    REGISTER_SRV_FUNCTION(names_and_shapes);
    REGISTER_SRV_FUNCTION(batch_count);
    REGISTER_SRV_FUNCTION(record_count);
}

web::json::value server_parser::post(std::string msg, std::string msg_body)
{
    try
    {
        if (msg.substr(0, endpoint_prefix.length()) != endpoint_prefix)
            throw std::invalid_argument("Invalid prefix");
        web::json::value reply_json  = web::json::value::object();
        reply_json["status"]["type"] = web::json::value::string("SUCCESS");
        reply_json["data"]["id"]     = web::json::value::string(
            to_string(m_loader_manager.register_agent(json::parse(msg_body))));
        return reply_json;
    }
    catch (exception& ex)
    {
        web::json::value reply_json         = web::json::value::object();
        reply_json["status"]["type"]        = web::json::value::string("FAILURE");
        reply_json["status"]["description"] = web::json::value::string(ex.what());
        return reply_json;
    }
}

web::json::value server_parser::get(std::string msg)
{
    try
    {
        if (msg.substr(0, endpoint_prefix.length()) != endpoint_prefix)
            throw std::invalid_argument("Invalid prefix");

        msg.erase(0, endpoint_prefix.length() + 1);

        auto path = web::uri::split_path(msg);
        if (path.size() != 2)
            throw std::invalid_argument("Invalid command");

        int dataset_id = std::stoi(path[0]);

        if (path[1] == "close")
        {
            return close(dataset_id);
        }
        else
        {
            auto it = process_func.find(path[1]);
            if (it == process_func.end())
                throw std::invalid_argument("Invalid command");
            else
                return (this->*it->second)(m_loader_manager.loader(dataset_id));
        }
    }
    catch (exception& ex)
    {
        web::json::value response_json         = web::json::value::object();
        response_json["status"]["type"]        = web::json::value::string("FAILURE");
        response_json["status"]["description"] = web::json::value::string(ex.what());
        return response_json;
    }
}

web::json::value server_parser::next(loader_adapter& loader)
{
    web::json::value response_json = web::json::value::object();

    if (loader.next())
    {
        response_json["status"]["type"]           = web::json::value::string("SUCCESS");
        response_json["data"]["position"]         = web::json::value::string(loader.position());
        response_json["data"]["fixed_buffer_map"] = web::json::value::string(loader.data());
    }
    else
    {
        response_json["status"]["type"] = web::json::value::string("END_OF_DATASET");
    }

    return response_json;
}

web::json::value server_parser::reset(loader_adapter& loader)
{
    web::json::value response_json = web::json::value::object();

    loader.reset();

    response_json["status"]["type"]           = web::json::value::string("SUCCESS");
    response_json["data"]["fixed_buffer_map"] = web::json::value::string(loader.data());

    return response_json;
}

web::json::value server_parser::batch_size(loader_adapter& loader)
{
    web::json::value response_json = web::json::value::object();

    response_json["status"]["type"]     = web::json::value::string("SUCCESS");
    response_json["data"]["batch_size"] = web::json::value::string(loader.batch_size());
    return response_json;
}

web::json::value server_parser::batch_count(loader_adapter& loader)
{
    web::json::value response_json = web::json::value::object();

    response_json["status"]["type"]      = web::json::value::string("SUCCESS");
    response_json["data"]["batch_count"] = web::json::value::string(loader.batch_count());
    return response_json;
}

web::json::value server_parser::record_count(loader_adapter& loader)
{
    web::json::value response_json = web::json::value::object();

    response_json["status"]["type"]       = web::json::value::string("SUCCESS");
    response_json["data"]["record_count"] = web::json::value::string(loader.record_count());
    return response_json;
}

web::json::value server_parser::names_and_shapes(loader_adapter& loader)
{
    web::json::value response_json = web::json::value::object();

    response_json["status"]["type"]           = web::json::value::string("SUCCESS");
    response_json["data"]["names_and_shapes"] = web::json::value::string(loader.names_and_shapes());
    return response_json;
}

web::json::value server_parser::close(uint32_t id)
{
    m_loader_manager.unregister_agent(id);
    web::json::value response_json  = web::json::value::object();
    response_json["status"]["type"] = web::json::value::string("SUCCESS");
    return response_json;
}

// /////////////////////////////////////////////////////////////////////////////

struct shutdown_deamon
{
    void operator()(aeon_server* server) { server->close().wait(); }
};

void start_deamon()
{
    utility::string_t port      = U("34568");
    utility::string_t http_addr = U("http://127.0.0.1:");
    utility::string_t path      = U("api");

    http_addr.append(port);
    uri_builder uri(http_addr);
    uri.append_path(path);

    auto addr = uri.to_uri().to_string();

    static std::unique_ptr<aeon_server, shutdown_deamon> server(new aeon_server(addr));
    server->open().wait();
}
