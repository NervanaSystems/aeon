/*
 * Copyright 2017 Intel(R) Nervana(TM)
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <getopt.h>

#include <cstdlib>
#include <cctype>
#include <cerrno>
#include <csignal>

#include <limits>
#include <iostream>
#include <iomanip>

#include <boost/filesystem/path.hpp>

#include "version.hpp"
#include "log.hpp"
#include "pidfile.hpp"
#include "service.hpp"

using namespace nervana;

namespace
{
    namespace detail
    {
        constexpr int default_port_number{4586};

        void                  fini(int, void*) { pidfile::remove(); }
        volatile sig_atomic_t configure{0};
        volatile sig_atomic_t terminate{0};

        namespace configuration
        {
            class exception : public std::exception
            {
            };

            void read(const boost::filesystem::path& path)
            {
                configure = 0;
                log::info("configuration read successfully");
            }
        }

        void sighandler(int signum)
        {
            switch (signum)
            {
            case SIGINT:
            case SIGTERM:
                log::info("%s caught - shutting down the service!",
                          (signum == SIGINT) ? "SIGINT" : "SIGTERM");
                terminate = 1;
                break;
            case SIGHUP:
                log::info("SIGHUP caught - reloading configuration");
                configure = 1;
                break;
            default: break;
            }
        }

        inline std::ostream& titleinfo(std::ostream& outs)
        {
            outs << "Intel(R) Nervana(tm) AEON Service " << aeon::version::major << "."
                 << aeon::version::minor << " (build " << aeon::build::number << ")"
                 << "\nCopyright (C) 2017 by Intel Corporation. All rights reserved.\n\n";

            return outs;
        }

        void version()
        {
            std::stringstream ss;
            titleinfo(ss) << "This is free software; see the source for copying "
                             "conditions. There is NO warranty;\nnot even for MARCHANTABILITY"
                             " of FITNESS FOR A PARTICULAR PURPOSE.\n\n";

            std::cout << ss.str() << std::flush;
        }

        void help(const std::string& progname)
        {
            std::stringstream ss;
            titleinfo(ss)
                << "Usage: " << progname
                << " [OPTIONS]\n\n"
                   "Mandatory arguments for long options are mandatory for short "
                   "options, too.\n\n"
                   "    -d --daemon       Run the process in background as a daemon process.\n"
                   "                      By default the process runs in foreground.\n"
                   "    -R --respawn      Daemon only: respawn the process if it abnormally "
                   "terminates.\n"
                   "    -u --uri=uri      URI to listening interface in format "
                   "'protocol://host:port/'.\n"
                   "    -r --address=ip   IP address of RDMA interface.\n"
                   "    -p --port=port    Port number of RDMA interface.\n"
                   "    -c --config=path  Path to configuration file (default /etc/"
                << progname << ".cfg)\n"
                               "    -N --noconfig     Bypass the system configuration file, /etc/"
                << progname
                << ".cfg.\n"
                   "                      Only user's ~/.aeonrc will be read (if it exists).\n"
                   "    -l --log=path     Path to log file (daemon default /var/log/"
                << progname << ".log).\n"
                               "                      Foreground process has no default.\n"
                               "    -v --version      Display version.\n"
                               "    -h --help         Print this help.\n\n";

            std::cout << ss.str() << std::flush;
        }
    }
}

int main(int argc, char* argv[])
{
    std::string progname{boost::filesystem::path{argv[0]}.filename().c_str()};

    std::string             address;
    std::string             port;
    boost::filesystem::path log_path;
    boost::filesystem::path cfg_path{"/etc/aeon/service.cfg"};
    web::http::uri          uri;
    bool                    run_as_daemon{false};
    bool                    respawn{false};
    bool                    ignore_config{false};

    for (;;)
    {
        static struct option long_options[] = {{"daemon", no_argument, nullptr, 'd'},
                                               {"noconfig", required_argument, nullptr, 'N'},
                                               {"respawn", no_argument, nullptr, 'R'},
                                               {"help", no_argument, nullptr, 'h'},
                                               {"uri", required_argument, nullptr, 'u'},
                                               {"address", required_argument, nullptr, 'a'},
                                               {"port", required_argument, nullptr, 'p'},
                                               {"log", required_argument, nullptr, 'l'},
                                               {"version", no_argument, nullptr, 'V'},
                                               {"config", required_argument, nullptr, 'c'},
                                               {nullptr, no_argument, nullptr, 0}};

        int option_index{0};
        int c{getopt_long_only(argc, argv, "h", long_options, &option_index)};

        if (c == -1)
        {
            break;
        }
        switch (c)
        {
        case 'l': log_path = optarg; break;
        case 'u': uri      = optarg; break;
        case 'a': address  = optarg; break;
        case 'p': port     = optarg; break;
        case 'c': cfg_path = optarg; break;
        case 'h': detail::help(progname); return 0;
        case 'V': detail::version(); return 0;
        case 'R': respawn       = true; break;
        case 'd': run_as_daemon = true; break;
        case 'N': ignore_config = true; break;
        case '?': return -1;
        default: break;
        }
    }

    if (!run_as_daemon || (getuid() != 0))
    {
        log::add_terminal_sink();
    }
    if (getuid() == 0)
    {
        if (log_path.empty())
        {
            log_path = "/var/log/" + progname;
        }
    }

    try
    {
        if (!log_path.empty())
        {
            log::add_file_sink(log_path);
        }
    }
    catch (const std::exception& e)
    {
        log::error("%s - no logging to file available!", e.what());
    }

    if (run_as_daemon)
    {
        if (getuid() != 0)
        {
            log::error(
                "insufficient user privileges - only root can run AEON "
                "service as daemon!");
            exit(EXIT_FAILURE);
        }
        else
        {
            if (pidfile::check())
            {
                log::add_terminal_sink();
                log::alert("the serivce is already running, exiting...");
                exit(EXIT_SUCCESS);
            }
        }
    }
    if (uri.is_empty())
    {
        log::critical(
            "URI of listening interface has not been provided, "
            "terminating...");
        return -1;
    }
    if (uri.scheme() != "http")
    {
        log::critical(
            "invalid URI protocol, only HTTP is supported, "
            "terminating...");
        return -1;
    }
    web::http::uri_builder uri_builder{uri};
    if (uri.port() == 0)
    {
        log::warning("URI is missing port number - using default (%d)",
                     detail::default_port_number);
        uri_builder.set_port(detail::default_port_number);
    }
    if (uri.path() != "/")
    {
        log::warning("URI has path specified (%s), ignoring...", uri.path());
        uri_builder.set_path("");
    }
    if (!uri.fragment().empty())
    {
        log::warning("URI has fragments specified (%s), ignoring...", uri.fragment());
        uri_builder.set_fragment("");
    }
    if (!uri.query().empty())
    {
        log::warning("URI has queries specified (%s), ignoring...", uri.query());
        uri_builder.set_query("");
    }
    if (!uri.user_info().empty())
    {
        log::warning("URI has user info specified (%s), ignoring...", uri.user_info());
        uri_builder.set_user_info("");
    }

    if (address.empty() != port.empty())
    {
        log::critical("both RDMA address and port should be provided, terminating...");
        return -1;
    }

    try
    {
        if (run_as_daemon)
        {
            pid_t pid{fork()};
            if (pid < 0)
            {
                log::debug("main(): fork() failed (errno=%d)", errno);
                exit(EXIT_FAILURE);
            }
            if (pid > 0)
            {
                exit(EXIT_SUCCESS);
            }
            if (setsid() < 0)
            {
                log::debug("main(): setsid() failed (errno=%d)", errno);
                exit(EXIT_FAILURE);
            }
            signal(SIGCHLD, SIG_IGN);
            pid = fork();
            if (pid < 0)
            {
                log::debug("main(): fork() failed (errno=%d)", errno);
                exit(EXIT_FAILURE);
            }
            if (pid > 0)
            {
                exit(EXIT_SUCCESS);
            }
            umask(0);
            if (chdir("/") < 0)
            {
                log::debug("main(): chdir() failed (errno=%d)", errno);
                exit(EXIT_FAILURE);
            }
            for (int i{(int)sysconf(_SC_OPEN_MAX)}; i > 0; i--)
            {
                close(i);
            }
            int t{open("/dev/null", O_RDWR)};
            dup(t);
            dup(t);

            pidfile::create();
            if (on_exit(detail::fini, nullptr))
            {
                exit(EXIT_FAILURE);
            }
        }

        signal(SIGINT, detail::sighandler);
        signal(SIGHUP, detail::sighandler);
        signal(SIGTERM, detail::sighandler);

        log::info("signal handlers installed");
        detail::configuration::read(cfg_path);

        aeon::service service
        {
            uri_builder.to_uri()
#if defined(ENABLE_OPENFABRICS_CONNECTOR)
                ,
                address + ":" + port
#endif
        };
        log::info("the service has started, listening on %s...", service.uri().to_string());

        while (detail::terminate == 0)
        {
            if (detail::configure != 0)
            {
                detail::configuration::read(cfg_path);
            }
            sleep(5);
        }
        log::info("the service has stopped");
        exit(EXIT_SUCCESS);
    }
    catch (const pidfile::exception::open& e)
    {
        log::critical(
            "unable to create a pid file - %s,"
            "terminating service...",
            e.error_message());
        exit(EXIT_FAILURE);
    }
    catch (const pidfile::exception::lock& e)
    {
        log::critical(
            "failure to lock the pid file - %s, "
            "terminating...",
            e.error_message());
        exit(EXIT_FAILURE);
    }
    catch (const pidfile::exception::write& e)
    {
        log::critical(
            "failure writing to the pid file - %s, "
            "terminating...",
            e.error_message());
        exit(EXIT_FAILURE);
    }
    catch (const web::uri_exception& e)
    {
        log::critical("unable to start the service - %s, terminating...", e.what());
        exit(EXIT_FAILURE);
    }
    catch (const detail::configuration::exception& e)
    {
        log::error("fail to read confiuration - %s, terminating...", e.what());
        exit(EXIT_FAILURE);
    }
}
