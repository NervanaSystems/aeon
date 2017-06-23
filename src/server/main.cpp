#include <iostream>
#include <memory>
#include <cpprest/http_listener.h>

#include "json.hpp"
#include "loader.hpp"

using namespace web::http;

struct default_config
{
    const int    height            = 16;
    const int    width             = 16;
    const size_t batch_size        = 32;
    const size_t record_count      = 1003;
    const size_t block_size        = 300;
    
    std::string manifest_filename;

    const nlohmann::json js_image = {
            {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    const nlohmann::json label        = {{"type", "label"}, {"binary", false}};
    const nlohmann::json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    nlohmann::json js           = {{"manifest_filename", "dupa.txt"},
                                   {"batch_size", batch_size},
                                   {"block_size", block_size},
                                   {"etl", {js_image, label}},
                                   {"augmentation", augmentation}};
};

class aeon_server
{
public:
    aeon_server(utility::string_t url);

    pplx::task<void> open() 
    { 
        return m_listener.open(); 
    }

    pplx::task<void> close() 
    {
        return m_listener.close(); 
    }

private:

    void handle_get(http_request message);

    experimental::listener::http_listener m_listener;
    std::map<size_t, std::shared_ptr<nervana::loader>> m_aeon_clients;
    default_config config;

    std::hash<std::shared_ptr<nervana::loader>> m_hash_fn;
};

aeon_server::aeon_server(utility::string_t url) : m_listener(url)
{
    m_listener.support(methods::GET, std::bind(&aeon_server::handle_get, this, std::placeholders::_1));
}

void aeon_server::handle_get(http_request message)
{
    std::cout << "Message received" << std::endl;

    std::shared_ptr<nervana::loader> loader = std::make_shared<nervana::loader>(config.js);
    size_t idx = m_hash_fn(loader);

    std::cout << "Hash " << idx << std::endl;
    m_aeon_clients[idx] = loader;

    web::json::value json_idx = web::json::value::object();
    json_idx["idx"] = web::json::value::number(idx);
    message.reply(status_codes::OK, json_idx);
}

std::shared_ptr<aeon_server> initialize(const utility::string_t& address)
{
    uri_builder uri(address);
    uri.append_path(U("aeon"));

    auto addr = uri.to_uri().to_string();
    std::shared_ptr<aeon_server> server = std::make_shared<aeon_server>(addr);
    server->open().wait();
    
    std::cout << "Address " << addr << std::endl;

    return server;
}

void shutdown(std::shared_ptr<aeon_server> server)
{
    server->close().wait();
}

int main(int argc, char *argv[])
{
    utility::string_t port = U("34568");

    utility::string_t address = U("http://127.0.0.1:");
    address.append(port);

    auto server = initialize(address);
    std::cout << "Press ENTER to exit." << std::endl;

    std::string line;
    std::getline(std::cin, line);

    shutdown(server);
    return 0;
}
