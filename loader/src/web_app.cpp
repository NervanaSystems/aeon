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

using namespace std;

//class web_starter
//{
//public:
//    web_starter()
//    {
//        cout << __FILE__ << " " << __LINE__ << " web_starter" << endl;
//    }
//    virtual ~web_starter() {}
//};

//static web_starter w_starter;

//void web_app::start()
//{
//    std::cout << __FILE__ << " " << __LINE__ << " test" << std::endl;
//}

string master_page = R"(
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
                  <a href="#" class="dropdown-toggle" data-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false">Demos <span class="caret"></span></a>
                  <ul class="dropdown-menu">
                    <li><a href="/forms">Forms Demo</a></li>
                  </ul>
                </li>
                <li class="dropdown">
                  <a href="#" class="dropdown-toggle" data-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false">Network Info <span class="caret"></span></a>
                  <ul class="dropdown-menu">
                    <li><a href="/show/mac">show MAC</a></li>
                    <li><a href="/show/ip">show IP</a></li>
                    <li><a href="/show/arp">show ARP</a></li>
                    <li><a href="/show/tcp">show TCP</a></li>
                  </ul>
                </li>
                <li class="dropdown">
                  <a href="#" class="dropdown-toggle" data-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false">OS Info <span class="caret"></span></a>
                  <ul class="dropdown-menu">
                    <li><a href="/show/thread">show threads</a></li>
                    <li><a href="/show/queue">show queues</a></li>
                    <li><a href="/show/event">show events</a></li>
                    <li><a href="/show/mutex">show mutexs</a></li>
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
    page_request_handler fn = bind(&web_app::process_page_request, this, placeholders::_1, placeholders::_2);
    web_server.register_page_handler(fn);
    web_server.start(8086);
}

web_app::~web_app()
{
    web_server.stop();
}

void web_app::home_page(web::page& p)
{
    time_t     t   = time(0);
    struct tm* now = localtime(&t);
    ostream& out = p.output_stream();

    out << "<span>Current time: " << asctime(now) << "</span>\n";

    out << "<table class=\"table table-striped\">\n";
    out << "  <thead>\n";
    out << "    <th>Protocol</th>\n";
    out << "    <th>Size of class</th>\n";
    out << "  </thead>\n";
    out << "  <tbody>\n";
//    out << "    <tr><td>MAC</td><td> " << sizeof(tcpStack.MAC)  << "</td></tr>\n";
//    out << "    <tr><td>IP</td><td>  " << sizeof(tcpStack.IP)   << "</td></tr>\n";
//    out << "    <tr><td>TCP</td><td> " << sizeof(tcpStack.TCP)  << "</td></tr>\n";
//    out << "    <tr><td>ARP</td><td> " << sizeof(tcpStack.ARP)  << "</td></tr>\n";
//    out << "    <tr><td>ICMP</td><td>" << sizeof(tcpStack.ICMP) << "</td></tr>\n";
//    out << "    <tr><td>DHCP</td><td>" << sizeof(tcpStack.DHCP) << "</td></tr>\n";
    out << "  </tbody>\n";
    out << "</table>\n";
}

void web_app::forms_demo(web::page& p)
{
    ostream& out = p.output_stream();
    out << "<form action=\"/formsresult\">";

    out << "<label for=\"FirstName\">First name:</label>";
    out << "<input type=\"text\" name=\"FirstName\" class=\"form-control\" value=\"Robert\"/>";
    out << "<br>";

    out << "<label for=\"LastName\">Last name:</label>";
    out << "<input type=\"text\" name=\"LastName\" class=\"form-control\" value=\"Kimball\"/>";
    out << "<br>";

    out << "<input type=\"submit\" value=\"submit\" />";

    out << R"(<a href="/files/test1.zip">test1.zip</a><br>)";
    out << "      <form action=\"/test/uploadfile\" method=\"POST\" ";
    out << "      enctype=\"multipart/form-data\" action=\"_URL_\">\n";
    out << "File: <input type=\"file\" name=\"file\" size=\"50\"><br>\n";
    out << "      <input type=\"submit\" value=\"Upload\">\n";
    out << "      </form><br>\n";
}

void web_app::forms_response(web::page& p)
{
    ostream& out = p.output_stream();
    for (auto arg : p.args())
    {
        string name, value;
        p.parse_arg(arg, name, value);
        out << "<span>" << name << " = " << value << "</span>";
        out << "<br>";
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
    else if (url == "/forms")
    {
        auto mc = bind(&web_app::forms_demo, this, placeholders::_1);
        p.master_page_string(master_page, "$content", mc);
    }
    else if (url == "/formsresult")
    {
        auto mc = bind(&web_app::forms_response, this, placeholders::_1);
        p.master_page_string(master_page, "$content", mc);
    }
    else if (url == "/files/test1.zip")
    {
        p.page_ok();
        p.send_file("c:\\test.rar");
    }
    else if (url == "/test/uploadfile")
    {
//        cout << "Reading " << p.content_length() << " bytes\n";
//        for (int i = 0; i < p.content_length(); i++)
//        {
//            p.connection().read();
//        }
//        cout << "Done reading\n";
//        p.page_ok();
//        out << "Upload " << p.content_length() << " bytes complete\n";
    }
    else
    {
        p.page_not_found();
    }
}
