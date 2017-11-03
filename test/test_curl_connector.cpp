/*
 Copyright 2017 Intel(R) Nervana(TM)
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

#include "gtest/gtest.h"

#include "curl_connector.hpp"
#include "log.hpp"
#include "web_server.hpp"

using std::ostream;
using std::string;
using std::stringstream;
using nervana::curl_connector;
using nervana::http_response;
using nervana::http_query_t;

namespace
{
    const string test_address = "127.0.0.1";
    const int    test_port    = 34567;

    const string index_page               = "This is index page!";
    const string first_page_endpoint      = "/first_page";
    const string first_page               = "<H1> First Page <H1><br>this is content";
    const string query1_name              = "var1";
    const string query2_name              = "var2";
    const string query_page_endpoint      = "/query_page";
    const string not_existing_endpoint    = "/some_missing_endpoint";
    const string post_page_endpoint       = "/post_page";
    const string post_query_page_endpoint = "/post_query_page";

    void process_page_request(web::page& p, const string& url);
}

TEST(curl_connector, get)
{
    web::server          server;
    page_request_handler fn = process_page_request;
    server.register_page_handler(fn);
    server.start(test_port);
    auto connector = curl_connector(test_address, test_port);

    http_response response = connector.get("");
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, index_page);

    response = connector.get("/");
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, index_page);

    response = connector.get(first_page_endpoint);
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, first_page);

    response = connector.get(not_existing_endpoint);
    EXPECT_EQ(response.code, 404);
}

TEST(curl_connector, get_query)
{
    web::server          server;
    page_request_handler fn = process_page_request;
    server.register_page_handler(fn);
    server.start(test_port);
    auto connector = curl_connector(test_address + "/", test_port);

    string        var1     = "abc";
    string        var2     = "def";
    http_query_t  query    = {{query1_name, var1}, {query2_name, var2}};
    http_response response = connector.get(query_page_endpoint, query);
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, var1 + var2);

    var1     = "a b\"c!";
    query    = {{query1_name, var1}};
    response = connector.get(query_page_endpoint, query);
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, "a%20b%22c%21");

    query    = {};
    response = connector.get(query_page_endpoint, query);
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, "");
}

TEST(curl_connector, post)
{
    web::server          server;
    page_request_handler fn = process_page_request;
    server.register_page_handler(fn);
    server.start(test_port);
    auto connector = curl_connector(test_address + "/", test_port);

    string        post_body = "this is body of post request";
    http_response response  = connector.post(post_page_endpoint, post_body);
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, post_body);

    response = connector.post(not_existing_endpoint);
    EXPECT_EQ(response.code, 404);
}

TEST(curl_connector, post_query)
{
    web::server          server;
    page_request_handler fn = process_page_request;
    server.register_page_handler(fn);
    server.start(test_port);
    auto connector = curl_connector(test_address + "/", test_port);

    {
        string       var1  = "abc";
        string       var2  = "def";
        http_query_t query = {{query1_name, var1}, {query2_name, var2}};
        stringstream expected;
        expected << query1_name << "=" << var1 << "&" << query2_name << "=" << var2;

        http_response response = connector.post(post_query_page_endpoint, query);

        EXPECT_EQ(response.code, 200);
        EXPECT_EQ(response.data, expected.str());
    }

    {
        string       var1  = "a b\"c!";
        http_query_t query = {{query1_name, var1}};
        stringstream expected;
        expected << query1_name << "="
                 << "a%20b%22c%21";

        http_response response = connector.post(post_query_page_endpoint, query);

        EXPECT_EQ(response.code, 200);
        EXPECT_EQ(response.data, expected.str());
    }

    {
        http_query_t query = {};

        http_response response = connector.post(post_query_page_endpoint, query);

        EXPECT_EQ(response.code, 200);
        EXPECT_EQ(response.data, "");
    }
}

TEST(curl_connector, del)
{
    web::server          server;
    page_request_handler fn = process_page_request;
    server.register_page_handler(fn);
    server.start(test_port);
    auto connector = curl_connector(test_address, test_port);

    http_response response = connector.del("");
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, index_page);

    response = connector.del("/");
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, index_page);

    response = connector.del(first_page_endpoint);
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, first_page);

    response = connector.del(not_existing_endpoint);
    EXPECT_EQ(response.code, 404);
}

TEST(curl_connector, del_query)
{
    web::server          server;
    page_request_handler fn = process_page_request;
    server.register_page_handler(fn);
    server.start(test_port);
    auto connector = curl_connector(test_address + "/", test_port);

    string        var1     = "abc";
    string        var2     = "def";
    http_query_t  query    = {{query1_name, var1}, {query2_name, var2}};
    http_response response = connector.del(query_page_endpoint, query);
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, var1 + var2);

    var1     = "a b\"c!";
    query    = {{query1_name, var1}};
    response = connector.del(query_page_endpoint, query);
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, "a%20b%22c%21");

    query    = {};
    response = connector.del(query_page_endpoint, query);
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, "");
}

TEST(curl_connector, post_and_get)
{
    web::server          server;
    page_request_handler fn = process_page_request;
    server.register_page_handler(fn);
    server.start(test_port);
    auto connector = curl_connector(test_address + "/", test_port);

    string        post_body = "this is body of post request";
    http_response response  = connector.post(post_page_endpoint, post_body);
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, post_body);

    response = connector.post(not_existing_endpoint);
    EXPECT_EQ(response.code, 404);

    response = connector.get("");
    EXPECT_EQ(response.code, 200);
    EXPECT_EQ(response.data, index_page);
}

namespace
{
    void process_page_request(web::page& p, const string& url)
    {
        ostream& out = p.output_stream();
        (void)out;
        if (url == "/")
        {
            p.send_string(index_page);
        }
        else if (url == first_page_endpoint)
        {
            p.send_string(first_page);
        }
        else if (url == query_page_endpoint)
        {
            // send concatenated query
            string response;
            if (p.args().find(query1_name) != p.args().end())
            {
                response += p.args().at(query1_name);
            }
            if (p.args().find(query2_name) != p.args().end())
            {
                response += p.args().at(query2_name);
            }
            p.send_string(response);
        }
        else if (url == post_page_endpoint || url == post_query_page_endpoint)
        {
            // echo post body in the response
            // payload retrieve is not possible at this moment
            string response = "";
            for (int i = 0; i < p.content_length(); i++)
            {
                char c = p.input_stream().get();
                response += c;
            }
            p.send_string(response);
        }
        else
        {
            p.page_not_found();
        }
    }
}
