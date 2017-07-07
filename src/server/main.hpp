#include <iostream>
#include <memory>
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

class aeon_server
{
public:
    aeon_server(utility::string_t url);

    pplx::task<void> open();
    pplx::task<void> close();

private:
    void handle_get(web::http::http_request message);

    web::http::experimental::listener::http_listener m_listener;
    std::map<size_t, std::shared_ptr<nervana::loader>> m_aeon_clients;
    default_config config;

    std::hash<std::shared_ptr<nervana::loader>> m_hash_fn;
};

