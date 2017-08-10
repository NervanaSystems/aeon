#include <iostream>
#include <memory>
#include <sstream>

#include "server.hpp"
#include "json.hpp"
#include "loader.hpp"
#include "gen_image.hpp"
#include "typemap.hpp"

#include "base64.hpp"
#include "aeonsvc.hpp"

using namespace web::http;
using nlohmann::json;
using namespace std;
using namespace nervana;

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

utility::string_t aeon_server::path = U("api");

aeon_server::aeon_server(std::string http_addr)
{
    uri_builder uri(http_addr);
    uri.append_path(path);

    auto addr = uri.to_uri().to_string();
    
    m_listener = make_unique<web::http::experimental::listener::http_listener>(addr);
    
    m_listener->support(methods::POST,
                       std::bind(&aeon_server::handle_post, this, std::placeholders::_1));
    m_listener->support(methods::GET,
                       std::bind(&aeon_server::handle_get, this, std::placeholders::_1));
    m_listener->support(methods::DEL,
                       std::bind(&aeon_server::handle_delete, this, std::placeholders::_1));
    m_listener->open().wait();
}

aeon_server::~aeon_server()
{
   m_listener->close().wait(); 
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

void aeon_server::handle_delete(http_request message)
{
    INFO << "[DELETE] " << message.relative_uri().path();
    message.reply(status_codes::OK,
                  m_server_parser.del(web::uri::decode(message.relative_uri().path())));
}

// /////////////////////////////////////////////////////////////////////////////

string loader_adapter::next()
{
    lock_guard<mutex> lg(m_mutex);
    
    m_loader.get_current_iter()++;
    return m_loader.get_current_iter().positional_end() ? string(""): data();
};

string loader_adapter::reset() 
{
    lock_guard<mutex> lg(m_mutex);
    
    m_loader.reset();
    return data();
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

string loader_adapter::batch_size() const
{
    return to_string(m_loader.batch_size());
};

string loader_adapter::names_and_shapes() const
{
    std::stringstream ss;
    ss << m_loader.get_names_and_shapes();
    return string(ss.str());
};

string loader_adapter::batch_count() const
{
    return to_string(m_loader.batch_count());
}

string loader_adapter::record_count() const
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

        auto it = process_func.find(path[1]);
        if (it == process_func.end())
            throw std::invalid_argument("Invalid command");
        else
            return (this->*it->second)(m_loader_manager.loader(dataset_id));
    }
    catch (exception& ex)
    {
        web::json::value response_json         = web::json::value::object();
        response_json["status"]["type"]        = web::json::value::string("FAILURE");
        response_json["status"]["description"] = web::json::value::string(ex.what());
        return response_json;
    }
}

web::json::value server_parser::del(std::string msg)
{
      try
    {
        if (msg.substr(0, endpoint_prefix.length()) != endpoint_prefix)
            throw std::invalid_argument("Invalid prefix");

        msg.erase(0, endpoint_prefix.length() + 1);

        int dataset_id = std::stoi(msg);
        m_loader_manager.unregister_agent(dataset_id);
        web::json::value reply_json  = web::json::value::object();
        reply_json["status"]["type"] = web::json::value::string("SUCCESS");
        return reply_json;
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
    string data = loader.next();
    
    if (!data.empty())
    {
        response_json["status"]["type"]           = web::json::value::string("SUCCESS");
        response_json["data"]["position"]         = web::json::value::string(loader.position());
        response_json["data"]["fixed_buffer_map"] = web::json::value::string(data);
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
    
    response_json["status"]["type"]           = web::json::value::string("SUCCESS");
    response_json["data"]["fixed_buffer_map"] = web::json::value::string(loader.reset());

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
