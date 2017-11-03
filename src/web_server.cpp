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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>

#include "web_server.hpp"

using namespace std;

vector<string> web::server::split(const string& src, char delimiter)
{
    size_t         pos;
    string         token;
    size_t         start = 0;
    vector<string> rc;
    while ((pos = src.find(delimiter, start)) != std::string::npos)
    {
        token = src.substr(start, pos - start);
        start = pos + 1;
        rc.push_back(token);
    }
    if (start <= src.size())
    {
        token = src.substr(start);
        rc.push_back(token);
    }
    return rc;
}

string web::server::to_lower(const string& s)
{
    std::locale  loc;
    stringstream ss;
    for (auto c : s)
        ss << char(tolower(c));
    return ss.str();
}

string web::server::trim(const string& s)
{
    string rc = s;
    while (!rc.empty() && (rc.back() == '\r' || rc.back() == '\n'))
    {
        rc.pop_back();
    }
    return rc;
}

web::server::server()
    : m_thread()
    , m_listen_connection(0)
    , m_current_connection(0)
    , m_page_handler(0)
    , m_error_handler(0)
{
}

web::server::~server()
{
    stop();
}

void web::server::start(uint16_t port)
{
    m_listen_connection      = std::make_shared<tcp::connection>(port);
    m_active                 = true;
    std::function<void()> fn = std::bind(&web::server::process_loop, this);
    m_thread                 = std::thread(fn);
}

void web::server::stop()
{
    m_active = false;
    m_listen_connection->close();
    if (m_thread.joinable())
    {
        m_thread.join();
    }
}

void web::server::wait_for_exit()
{
    if (m_thread.joinable())
    {
        m_thread.join();
    }
}

void web::server::register_page_handler(page_request_handler handler)
{
    m_page_handler = handler;
}

void web::server::register_error_handler(error_message_handler handler)
{
    m_error_handler = handler;
}

void web::server::page_request(web::page& current_page)
{
    string                                path;
    std::string                           url = "";
    string                                line;
    vector<string>                        lines;
    std::shared_ptr<web::tcp::connection> connection = current_page.m_connection;
    istream&                              input      = connection->get_input_stream();

    current_page.m_http_header_sent = false;

    while (getline(input, line))
    {
        line = trim(line);

        if (line.empty())
            break;
        lines.push_back(line);
    }

    if (lines.size() > 0)
    {
        line                  = lines[0];
        vector<string> tokens = split(line, ' ');

        if (tokens.size() > 1)
        {
            path = tokens[1];

            for (int i = 1; i < lines.size(); i++) // skip first line
            {
                line        = lines[i];
                auto offset = line.find_first_of(':');
                if (offset == string::npos)
                {
                    continue;
                }
                string tag   = to_lower(line.substr(0, offset));
                string value = line.substr(offset + 1);

                if (tag == "authorization")
                {
                }
                else if (tag == "content-type")
                {
                    current_page.m_content_type = value;
                }
                else if (tag == "content-length")
                {
                    current_page.m_content_length = stol(value);
                }
            }

            if (path.empty() == false)
            {
                vector<string> tok = split(path, '?');
                url                = tok[0];
                if (tok.size() > 1)
                {
                    auto           query = tok[1];
                    vector<string> qlist = split(query, '&');
                    for (const string& arg : qlist)
                    {
                        auto result = split(arg, '=');
                        if (result.size() == 1)
                        {
                            result.push_back("");
                        }
                        current_page.m_args.insert({result[0], result[1]});
                    }
                }

                // Page.Initialize( connection );
                if (m_page_handler)
                {
                    try
                    {
                        m_page_handler(current_page, url);
                    }
                    catch (exception)
                    {
                    }
                }

                connection->flush();
                connection->close();
            }
        }
    }
}

void web::server::connection_handler_entry(std::shared_ptr<page> page)
{
    page->m_server->page_request(*page);
    //    page->m_thread.detach();
}

void web::server::process_loop()
{
    std::shared_ptr<web::tcp::connection> connection;
    std::shared_ptr<page>                 current_page;

    while (m_active)
    {
        try
        {
            connection = m_listen_connection->listen();
        }
        catch (exception)
        {
            break;
        }

        // Spawn off a thread to handle this connection
        current_page = std::shared_ptr<page>(new page());
        if (current_page)
        {
            current_page->initialize(connection);
            current_page->m_server = this;
            if (m_single_thread)
            {
                current_page->m_server->page_request(*current_page);
            }
            else
            {
                auto fn = std::bind(&web::server::connection_handler_entry, current_page);
                current_page->m_thread = std::thread(fn);
            }
        }
        else
        {
            cout << "Error: Out of pages\n";
        }
    }
}

web::page::page()
{
}

web::page::~page()
{
    if (m_thread.joinable())
    {
        m_thread.join();
    }
}

std::ostream& web::page::output_stream()
{
    return m_connection->get_output_stream();
}

std::istream& web::page::input_stream()
{
    return m_connection->get_input_stream();
}

void web::page::initialize(std::shared_ptr<web::tcp::connection> connection)
{
    m_connection = connection;
}

string web::page::html_encode(const string& s)
{
    stringstream ss;
    for (char ch : s)
    {
        switch (ch)
        {
        case '&': ss << "&amp;"; break;
        case '<': ss << "&lt;"; break;
        case '>': ss << "&gt;"; break;
        case '"': ss << "&quot;"; break;
        case '\'': ss << "&apos;"; break;
        default: ss << ch; break;
        }
    }
    return ss.str();
}

bool web::page::puts(const string& s)
{
    for (char c : s)
    {
        switch (c)
        {
        case '\n': raw_send("<br>", 5); break;
        case '\r': break;
        default: raw_send(&c, 1); break;
        }
    }

    return true;
}

bool web::page::send_string(const string& s)
{
    return raw_send(s.data(), s.size());
}

bool web::page::raw_send(const void* p, size_t length)
{
    if (!m_http_header_sent)
    {
        page_ok();
    }
    m_connection->write((const char*)p, length);

    return true;
}

void web::page::send_ascii_string(const string& s)
{
    char buffer[4];

    for (char c : s)
    {
        if (c < 0x21 || c > 0x7E)
        {
            snprintf(buffer, sizeof(buffer), "%%%02X", c);
            raw_send(buffer, 3);
        }
        else
        {
            raw_send(&c, 1);
        }
    }
}

void web::page::dump_data(const char* buffer, size_t length)
{
    int i;
    int j;

    i = 0;
    j = 0;
    send_string("<code>\n");
    while (i + 1 <= length)
    {
        printf("%04X ", i);
        for (j = 0; j < 16; j++)
        {
            if (i + j < length)
            {
                printf("%02X ", buffer[i + j]);
            }
            else
            {
                printf("   ");
            }

            if (j == 7)
            {
                printf("- ");
            }
        }
        printf("  ");
        for (j = 0; j < 16; j++)
        {
            if (buffer[i + j] >= 0x20 && buffer[i + j] <= 0x7E)
            {
                printf("%c", buffer[i + j]);
            }
            else
            {
                printf(".");
            }
            if (i + j + 1 == length)
            {
                break;
            }
        }

        i += 16;

        send_string("<br>\n");
    }
    send_string("</code>\n");
}

void web::page::page_not_found()
{
    m_http_header_sent = true;
    send_string("HTTP/1.0 404 Not Found\r\n\r\n");
}

void web::page::page_ok(const char* mimeType)
{
    m_http_header_sent = true;
    send_string("HTTP/1.0 200 OK\r\nContent-type: ");
    send_string(mimeType);
    send_string("\r\n\r\n");
}

void web::page::page_no_content()
{
    m_http_header_sent = true;
    send_string("HTTP/1.0 204 No Content\r\nContent-type: text/html\r\n\r\n");
}

void web::page::page_unauthorized()
{
    m_http_header_sent = true;
    send_string("HTTP/1.0 401 Unauthorized\r\nContent-type: text/html\r\n\r\n");
}

bool web::page::send_file(const string& filename)
{
    bool rc = false;

    ifstream f(filename, ios::in | ios::binary);
    if (f)
    {
        f.seekg(0, f.end);
        size_t size = f.tellg();
        f.seekg(0, f.beg);

        char* buffer = new char[size];
        f.read(buffer, size);
        f.close();
        send_as_file(buffer, size);
        delete[] buffer;

        rc = true;
    }

    return rc;
}

bool web::page::send_as_file(const char* buffer, size_t size)
{
    stringstream ss;
    ss << "HTTP/1.0 200 OK\r\nContent-Type: application/octet-stream\r\n";
    ss << "Content-Length: " << size << "\r\n\r\n";
    m_http_header_sent = true;
    send_string(ss.str());

    raw_send(buffer, size);

    return true;
}

void web::page::flush()
{
    m_connection->flush();
}

const std::map<std::string, std::string>& web::page::args() const
{
    return m_args;
}

const std::string& web::page::content_type() const
{
    return m_content_type;
}

size_t web::page::content_length() const
{
    return m_content_length;
}

web::tcp::connection& web::page::connection()
{
    return *m_connection;
}

void web::page::master_page_file(const string& path, const string& marker, marker_content content)
{
    ifstream file(path);
    if (!file)
    {
        throw std::runtime_error("error opening file '" + path + "'");
    }
    string data((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    master_page_string(data, marker, content);
}

void web::page::master_page_string(const string&  source,
                                   const string&  marker,
                                   marker_content content)
{
    int markerIndex = 0;
    for (char c : source)
    {
        if (c == marker[markerIndex])
        {
            markerIndex++;
            if (markerIndex == marker.size())
            {
                // found the marker
                markerIndex = 0;
                send_string("<!-- ");
                send_string(marker);
                send_string(" start -->");
                content(*this);
                send_string("<!-- ");
                send_string(marker);
                send_string(" end -->");
            }
        }
        else if (markerIndex > 0)
        {
            // Send the part of the marker that matched so far
            raw_send(marker.data(), markerIndex);
            markerIndex = 0;
        }
        else
        {
            char ch = c;
            raw_send(&ch, 1);
        }
    }
}

web::tcp::connection::connection()
    : m_ostream{this}
    , m_istream{this}
    , m_put_back(5)
    , m_char_buffer(100 + m_put_back)
    , m_is_server{false}
    , m_listening_port{0}
{
}

web::tcp::connection::connection(uint16_t port)
    : m_ostream{this}
    , m_istream{this}
    , m_put_back(5)
    , m_char_buffer(100 + m_put_back)
    , m_is_server{true}
    , m_listening_port{port}
{
    char* end = &m_char_buffer.front() + m_char_buffer.size();
    setg(end, end, end);

    struct sockaddr_in serv_addr;
    m_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (m_socket < 0)
    {
        throw std::runtime_error("ERROR opening socket");
    }

    // set SO_REUSEADDR on a socket to true (1):
    int optval = 1;
    setsockopt(m_socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof optval);

    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family      = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port        = htons(port);
    if (::bind(m_socket, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0)
    {
        stringstream ss;
        ss << "error binding to port " << port;
        throw std::runtime_error(ss.str());
    }
    ::listen(m_socket, 5);
}

web::tcp::connection::~connection()
{
}

ostream& web::tcp::connection::get_output_stream()
{
    return m_ostream;
}

istream& web::tcp::connection::get_input_stream()
{
    return m_istream;
}

streamsize web::tcp::connection::xsputn(const char* s, streamsize n)
{
    write(s, n);
    return n;
}

int web::tcp::connection::overflow(int c)
{
    char ch = c;
    write(&ch, 1);
    return c;
}

std::streambuf::int_type web::tcp::connection::underflow()
{
    std::streambuf::int_type rc;
    if (gptr() < egptr()) // buffer not exhausted
    {
        rc = traits_type::to_int_type(*gptr());
    }
    else
    {
        char* base  = &m_char_buffer.front();
        char* start = base;

        if (eback() == base) // true when this isn't the first fill
        {
            // Make arrangements for putback characters
            std::memmove(base, egptr() - m_put_back, m_put_back);
            start += m_put_back;
        }

        // start is now the start of the buffer, proper.
        // Read from fptr_ in to the provided buffer
        auto n = ::read(m_socket, start, m_char_buffer.size() - (start - base));
        if (n == 0)
        {
            rc = traits_type::eof();
        }
        else
        {
            // Set buffer pointers
            setg(base, start, start + n);
            rc = traits_type::to_int_type(*gptr());
        }
    }

    return rc;
}

void web::tcp::connection::close()
{
    if (m_is_server)
    {
        int                sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in remote;
        remote.sin_family = AF_INET;
        inet_pton(AF_INET, "127.0.0.1", &remote.sin_addr);
        remote.sin_port = ntohs(m_listening_port);
        if (connect(sock, (struct sockaddr*)&remote, sizeof(remote)) < 0)
        {
            cout << __FILE__ << " " << __LINE__ << " error connecting to self" << endl;
        }
        usleep(100);
        ::close(sock);
        ::close(m_socket);
    }
    else
    {
        struct linger ling;
        ling.l_onoff  = 1;
        ling.l_linger = 30;
        setsockopt(m_socket, SOL_SOCKET, SO_LINGER, &ling, sizeof(ling));

        shutdown(m_socket, SHUT_WR);

        while (!m_is_server)
        {
            //        int flags = fcntl(m_socket, F_GETFL, 0);
            //        fcntl(m_socket, F_SETFL, flags | O_NONBLOCK);

            //        int pending_data;
            //        fcntl(m_socket, TIOCOUTQ, &pending_data);
            //        cout << __FILE__ << " " << __LINE__ << " pending_data " << pending_data << endl;

            char   buffer[32];
            size_t rc = read(m_socket, buffer, sizeof(buffer));
            if (rc == 0)
            {
                break;
            }
            else if (rc == -1 && errno != EAGAIN && errno != EWOULDBLOCK)
            {
                break;
            }
        }

        ::close(m_socket);
    }
}

std::shared_ptr<web::tcp::connection> web::tcp::connection::listen()
{
    int                newsockfd;
    struct sockaddr_in cli_addr;
    socklen_t          clilen;
    clilen    = sizeof(cli_addr);
    newsockfd = ::accept(m_socket, (struct sockaddr*)&cli_addr, &clilen);
    if (newsockfd < 0)
    {
        throw std::runtime_error("ERROR on accept");
    }
    auto rc      = std::shared_ptr<connection>(new connection());
    rc->m_socket = newsockfd;
    return rc;
}

void web::tcp::connection::write(const char* data, size_t size)
{
    string s{data, size};
    int    count = 0;
    int    flags = 0;
    while (count != size)
    {
        count = ::send(m_socket, data, size, flags);
        if (count == size)
        {
            break;
        }
        else if (count < 0)
        {
            break;
        }
        data += count;
        size -= count;
    }
}

void web::tcp::connection::write(const std::string& s)
{
    write(s.data(), s.size());
}

void web::tcp::connection::flush()
{
    fsync(m_socket);
}
