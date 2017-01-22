/*
 Copyright 2016 Nervana Systems Inc.
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

#include "web_app.hpp"
#include "async_manager.hpp"
#include "util.hpp"

using namespace std;

// class web_starter
// {
// public:
//     web_starter()
//     {
//         cout << __FILE__ << " " << __LINE__ << " web_starter" << endl;
//     }
//     virtual ~web_starter()
//     {
//         cout << __FILE__ << " " << __LINE__ << " ~web_starter" << endl;
//     }
// };

// static web_starter w_starter;

// void web_app::start()
// {
//     cout << __FILE__ << " " << __LINE__ << " test" << endl;
// }

static web_app s_web_app{};

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
                    <li><a href="/stopwatch">Stopwatch</a></li>
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

web_app::web_app()
{
    page_request_handler fn =
        bind(&web_app::process_page_request, this, placeholders::_1, placeholders::_2);
    web_server.register_page_handler(fn);
    web_server.start(8086);
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
    else
    {
        p.page_not_found();
    }
}
