#include <cpprest/http_client.h>

using namespace web;
using namespace http;

class aeon_client
{
private:
    uint64_t m_idx;
    client::http_client m_client;

public:
    aeon_client(uint64_t idx, const client::http_client& client);
    void next();
};


aeon_client::aeon_client(uint64_t idx, const client::http_client& client)
    : m_idx(idx)
    , m_client(client)
{
}

void aeon_client::next()
{
    utility::ostringstream_t buf;
    buf << U("?idx=") << m_idx;


}

std::shared_ptr<aeon_client> create_client(const http::uri& uri)
{
    client::http_client client(http::uri_builder(uri).append_path(U("/aeon")).to_uri());
    http_response response = client.request(methods::GET).get();

    std::shared_ptr<aeon_client> c;

    if (response.status_code() == status_codes::OK)
    {
        if (response.headers().content_type() == U("application/json"))
        {
            web::json::value json_idx = response.extract_json().get();
            web::json::object idx_obj = json_idx.as_object();

            web::json::value idx = idx_obj["idx"];
            c = std::make_shared<aeon_client>(idx.as_number().to_uint64(), client);
        }
    }

    return c;
}

int main()
{
    utility::string_t port = U("34568");
    utility::string_t address = U("http://127.0.0.1:");
    address.append(port);

    http::uri uri = http::uri(address);

    std::shared_ptr<aeon_client> client = create_client(uri);

    while (true)
    {

    }

    return 0;
}
