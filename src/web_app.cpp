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

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "web_app.hpp"
#include "async_manager.hpp"
#include "util.hpp"
#include "base64.hpp"
#include "loader.hpp"

using namespace std;

void web_app::register_loader(nervana::loader* l)
{
    m_loader_list.push_back(l);
}

void web_app::deregister_loader(const nervana::loader* l)
{
    auto f = find(m_loader_list.begin(), m_loader_list.end(), l);
    if (f != m_loader_list.end())
    {
        m_loader_list.erase(f);
    }
}

static string master_page = R"(
    <html>
    <head>
        <title>Aeon Debug</title>
        <script src="//ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js"></script>
        <script src="//netdna.bootstrapcdn.com/bootstrap/3.1.1/js/bootstrap.min.js"></script>
        <link rel="stylesheet" type="text/css" href="//netdna.bootstrapcdn.com/bootstrap/3.1.1/css/bootstrap.min.css" />
    </head>
    <body>
        <!-- Static navbar -->
        <nav class="navbar navbar-default navbar-static-top">
          <div class="container">
            <div class="navbar-header">
              <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar" aria-expanded="false" aria-controls="navbar">
                <span class="sr-only">Toggle navigation</span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
              </button>
              <a class="navbar-brand" href="https://github.com/NervanaSystems/aeon">Aeon</a>
            </div>
            <div id="navbar" class="navbar-collapse collapse">
              <ul class="nav navbar-nav">
                <li><a href="/">Home</a></li>
                <li class="dropdown">
                  <a href="#" class="dropdown-toggle" data-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false">Aeon Stats <span class="caret"></span></a>
                  <ul class="dropdown-menu">
                    <li><a href="/loader">Loader</a></li>
                  </ul>
                </li>
              </ul>
            </div><!--/.nav-collapse -->
          </div>
        </nav>

    <div class="container">
        $content
    </div> <!-- /container -->

    </body>
    </html>
)";

web_app::web_app(uint16_t port)
{
    page_request_handler fn =
        bind(&web_app::process_page_request, this, placeholders::_1, placeholders::_2);
    web_server.register_page_handler(fn);
    web_server.start(port);
}

web_app::~web_app()
{
    // web_server.stop();
}

void web_app::home_page(web::page& p)
{
    time_t     t   = time(0);
    struct tm* now = localtime(&t);
    ostream&   out = p.output_stream();

    out << "<span>Current time: " << asctime(now) << "</span>\n";

    out << "<table class=\"table table-striped\">\n";
    out << "  <thead>\n";
    out << "    <th>Name</th>\n";
    out << "    <th>State</th>\n";
    out << "  </thead>\n";
    out << "  <tbody>\n";
    for (auto info : nervana::async_manager_status)
    {
        out << "<tr>";
        out << "<td> " << info->get_name() << "</td>";
        out << "<td>";
        switch (info->get_state())
        {
        case nervana::async_state::idle: out << "idle"; break;
        case nervana::async_state::wait_for_buffer: out << "waiting for buffer"; break;
        case nervana::async_state::fetching_data: out << "fetching data"; break;
        case nervana::async_state::processing: out << "processing"; break;
        }
        out << "</td>";
        out << "</tr>";
    }
    out << "  </tbody>\n";
    out << "</table>\n";
}

void web_app::stopwatch(web::page& p)
{
}

void web_app::loader(web::page& p)
{
    ostream& out = p.output_stream();

    for (nervana::loader* current_loader : m_loader_list)
    {
        auto config = current_loader->get_current_config();
        out << "<pre>";
        out << config.dump(4);
        out << "</pre>";

        // Fetch next output buffer
        const nervana::fixed_buffer_map& fixed_buffer = *(current_loader->get_current_iter());
        const nervana::buffer_fixed_size_elements* buffer_ptr = fixed_buffer["image"];
        if (buffer_ptr)
        {
            // explicit copy the data
            nervana::buffer_fixed_size_elements image_buffer{*buffer_ptr};
            out << "<div class=\"container\">";
            for (size_t i = 0; i < image_buffer.get_item_count(); i++)
            {
                cv::Mat         mat = image_buffer.get_item_as_mat(i);
                vector<uint8_t> encoded;
                imencode(".jpg", mat, encoded);
                vector<char> b64 =
                    nervana::base64::encode((const char*)encoded.data(), encoded.size());
                out << "\n<img src=\"data:image/jpg;base64,";
                p.raw_send(b64.data(), b64.size());
                out << "\" style=\"padding-top:5px\"";
                out << "class=\"image col-lg-3\" ";
                out << "/>";
            }
            out << "</div>";
        }
    }
}

void web_app::page_404(web::page& p)
{
    ostream& out = p.output_stream();
    out << "<div class=\"jumbotron>";
    out << "<h1>Page Not Found</h1>";
    out << "</div>";
}

void web_app::process_page_request(web::page& p, const string& url)
{
    ostream& out = p.output_stream();
    (void)out;
    if (url == "/")
    {
        auto mc = bind(&web_app::home_page, this, placeholders::_1);
        p.master_page_string(master_page, "$content", mc);
    }
    else if (url == "/stopwatch")
    {
        auto mc = bind(&web_app::stopwatch, this, placeholders::_1);
        p.master_page_string(master_page, "$content", mc);
    }
    else if (url == "/loader")
    {
        auto mc = bind(&web_app::loader, this, placeholders::_1);
        p.master_page_string(master_page, "$content", mc);
    }
    else
    {
        p.page_not_found();
    }
}
