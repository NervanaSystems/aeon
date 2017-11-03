/*
 Copyright 2016 Intel(R) Nervana(TM)
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <strings.h>
#include <memory.h>
#include <streambuf>
#include <iostream>
#include <map>

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>

#define MAX_ACTIVE_CONNECTIONS 3
#define HTTPD_PATH_LENGTH_MAX 256
#define MAX_ARGV 10

class ProtocolTCP;

namespace web
{
    class server;
    class page;
    namespace tcp
    {
        class connection;
    }
}

typedef std::function<void(web::page& p, const std::string& url)> page_request_handler;
typedef std::function<void(const std::string& message)> error_message_handler;

class web::tcp::connection : public std::streambuf
{
public:
    connection(uint16_t port);
    ~connection();
    void                        close();
    std::shared_ptr<connection> listen();

    void write(const std::string& s);
    void write(const char* data, size_t size);
    void flush();

    std::ostream& get_output_stream();
    std::istream& get_input_stream();

private:
    std::streamsize xsputn(const char* s, std::streamsize n) override;
    std::streambuf::int_type overflow(std::streambuf::int_type c) override;
    std::streambuf::int_type underflow() override;

    connection();

    int               m_socket;
    std::ostream      m_ostream;
    std::istream      m_istream;
    const std::size_t m_put_back;
    std::vector<char> m_char_buffer;
    bool              m_is_server;
    uint16_t          m_listening_port;
};

class web::server
{
    friend class HTTPPage;

public:
    server();
    ~server();

    void start(uint16_t port);
    void stop();

    void register_page_handler(page_request_handler);
    void register_error_handler(error_message_handler);

    void page_request(page& page);

    void wait_for_exit();

private:
    server(server&);

    static void connection_handler_entry(std::shared_ptr<page>);
    void        connection_handler(void*);

    void process_loop();

    static std::vector<std::string> split(const std::string& src, char delimiter);
    static std::string to_lower(const std::string& s);
    static std::string trim(const std::string& s);

    std::thread                           m_thread;
    std::shared_ptr<web::tcp::connection> m_listen_connection;
    std::shared_ptr<web::tcp::connection> m_current_connection;
    page_request_handler                  m_page_handler;
    error_message_handler                 m_error_handler;
    bool                                  m_active;
    bool                                  m_single_thread = true;
};

class web::page
{
    friend class server;

public:
    ~page();
    typedef std::function<void(web::page&)> marker_content;

    void initialize(std::shared_ptr<web::tcp::connection>);

    static std::string html_encode(const std::string& s);

    // Puts writes the string converting all newline characters to <br>
    bool puts(const std::string& string);
    bool send_string(const std::string& string);
    bool raw_send(const void* buffer, size_t length);
    // SendASCIIString converts all non-displayable characters
    // to '%XX' where XX is the hex values of the character
    void send_ascii_string(const std::string& string);
    void dump_data(const char* buffer, size_t length);
    bool send_file(const std::string& filename);
    bool send_as_file(const char* buffer, size_t length);

    void page_ok(const char* mimeType = "text/html");
    void page_not_found();
    void page_no_content();
    void page_unauthorized();

    void master_page_file(const std::string& path, const std::string& marker, marker_content);
    void master_page_string(const std::string& path, const std::string& marker, marker_content);

    void flush();

    const std::map<std::string, std::string>& args() const;
    const std::string& content_type() const;
    size_t             content_length() const;
    tcp::connection&   connection();

    page();

    std::ostream& output_stream();
    std::istream& input_stream();

private:
    page(page&);

    void   close_pending_open();
    size_t get_file_size(const std::string& filename);

    std::string m_url;
    std::string m_content_type;
    int         m_content_length;
    std::map<std::string, std::string> m_args;
    std::shared_ptr<web::tcp::connection> m_connection;
    std::thread                           m_thread;
    server*                               m_server;
    bool                                  m_http_header_sent;
};
