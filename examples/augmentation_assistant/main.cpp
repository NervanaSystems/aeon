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

#include <iostream>

#include "aeon.hpp"
#include "web_server.hpp"

using namespace std;

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
                <li><a href="/exit">Exit</a></li>
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

static uint16_t port = 5000;

class augmentation_server
{
public:
    augmentation_server()
    {
        page_request_handler fn =
            bind(&augmentation_server::page_handler, this, placeholders::_1, placeholders::_2);
        m_server.register_page_handler(fn);
        m_server.start(port);
    }

    void page_handler(web::page& page, const std::string& url)
    {
        // ostream& out = page.output_stream();
        if (url == "/")
        {
            auto mc = bind(&augmentation_server::home_page, this, placeholders::_1);
            page.master_page_string(master_page, "$content", mc);
        }
        else if (url == "/exit")
        {
            // m_server.stop();
        }
    }

    void wait_for_exit()
    {
        m_server.wait_for_exit();
    }

    void home_page(web::page& p)
    {
        time_t     t   = time(0);
        struct tm* now = localtime(&t);
        ostream&   out = p.output_stream();
        nervana::image::config cfg;

        out << "<span>Current time: " << asctime(now) << "</span>\n";

        out << "<table class=\"table table-striped\">\n";
        out << "  <thead>\n";
        out << "    <th>Name</th>\n";
        out << "    <th>Type</th>\n";
        out << "    <th>Required</th>\n";
        out << "    <th>Default</th>\n";
        out << "  </thead>\n";
        out << "  <tbody>\n";
        for (auto x : cfg.get_config_list())                                                             \
        {
            out << "  <tr>\n";
            out << "    <td>" << x->name() << "</td>\n";
            out << "    <td>" << x->type()  << "</td>\n";
            out << "    <td>" << (x->required() ? "REQUIRED" : "OPTIONAL")  << "</td>\n";
            out << "    <td>" << (x->required() ? "" : x->get_default_value())  << "</td>\n";
            out << "  </tr>\n";
        }
        out << "  </tbody>\n";
        out << "</table>\n";



        out << "<form class=\"form-horizontal\">\n";
        for (auto x : cfg.get_config_list())                                                             \
        {
            out << "  <div class=\"form-group\">\n";
            if (x->type() == "unsigned int" || x->type() == "float")
            {
                out << "    <label for=\"" << x->name() << "\" class=\"col-sm-2 control-label\">" << x->name() << "</label>\n";
                out << "    <div class=\"col-sm-3\">\n";
                out << "      <input type=\"number\" class=\"form-control\" id=\"" << x->name() << "\">\n";
                out << "    </div>\n";
            } 
            else if (x->type() == "std::string")
            {
                out << "    <label for=\"" << x->name() << "\" class=\"col-sm-2 control-label\">" << x->name() << "</label>\n";
                out << "    <div class=\"col-sm-3\">\n";
                out << "      <input type=\"text\" class=\"form-control\" id=\"" << x->name() << "\">\n";
                out << "    </div>\n";
            }
            else if (x->type() == "bool")
            {
                out << "    <div class=\"col-sm-offset-2 col-sm-3\">\n";
                out << "      <div class=\"checkbox\">\n";
                out << "        <label>\n";
                out << "          <input type=\"checkbox\" " << (x->get_default_value() == "0" ? "" : "checked") << "> " << x->name() << "\n";
                out << "        </label>\n";
                out << "      </div>\n";
                out << "    </div>\n";
            }
            else if (x->type().find("distribution") != string::npos)
            {
                out << "    <label for=\"" << x->name() << "\" class=\"col-sm-2 control-label\">" << x->name() << "</label>\n";
                out << "    <div class=\"col-sm-3\">\n";
                out << "      <input type=\"text\" class=\"form-control\" id=\"" << x->name() << "\">\n";
                out << "    </div>\n";
            }
            else
            {
                out << "<!-- " << x->type() << " -->\n";
            }
            out << "  </div>\n";
        }
        out << "</form>\n";
    }

private:
    web::server m_server;
    vector<int> m_elements_size_list = {1024, 32};
};

int main(int argc, char** argv)
{
    std::cout << "server started on port " << port << std::endl;
    augmentation_server server;

    server.wait_for_exit();
}
