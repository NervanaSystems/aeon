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

#include "log.hpp"
#include "pidfile.hpp"
#include "service.hpp"

using namespace nervana;

namespace {
  namespace detail {

    void fini(int, void *) {
      pidfile::remove();
    }

    volatile sig_atomic_t configure{ 0 };
    volatile sig_atomic_t terminate{ 0 };

    void read_configuration(const boost::filesystem::path& path) {
      configure = 0;
    }

    void sighandler(int signum) {
      switch (signum) {
      case SIGINT:
      case SIGTERM:
        log::info("%s caught - shutting down AEON service",
                  (signum == SIGINT) ? "SIGINT" : "SIGTERM");
        terminate = 1;
        break;
      case SIGHUP:
        log::info("SIGHUP caught - reloading AEON service "
                    "configuration...");
        configure = 1;
        break;
      default:
        log::info("%d caught!", signum);
      }
    }

    void help(const std::string& progname) {
      std::stringstream ss;
      ss << "Usage: " << progname
         << " [--daemon] [--address <address> [--port <port>]]\n";
      std::cerr << ss.str() << std::flush;
    }
  }
}

int main(int argc, char* argv[]) {
#if defined(program_invocation_short_name)
  std::string progname{ program_invocation_short_name };
#else
  std::string progname{
    boost::filesystem::path{ argv[0] }.filename().c_str()
  };
#endif

  std::string addr;
  std::string port;
  boost::filesystem::path log_path;
  boost::filesystem::path cfg_path{ "/etc/aeon/service.cfg" };
  bool run_as_daemon{ false };

  for (;;) {
    static struct option long_options[] = {
      { "daemon", no_argument, nullptr, 'd' },
      { "help", no_argument, nullptr, 'h' },
      { "address", required_argument, nullptr, 'a' },
      { "port", required_argument, nullptr, 'p' },
      { "log", required_argument, nullptr, 'l' },
      { "version", no_argument, nullptr, 'v'},
      { "config", required_argument, nullptr, 'c' },
      { nullptr, no_argument, nullptr, 0 }
    };

    int option_index{ 0 };
    int c{ getopt_long_only(argc, argv, "h", long_options, &option_index) };

    if (c == -1) {
      break;
    }
    switch (c) {
    case 'l':
      log_path = optarg;
      break;
    case 'a':
      addr = optarg;
      break;
    case 'p':
      port = optarg;
      break;
    case 'c':
      cfg_path = optarg;
      break;
    case 'h':
      detail::help(progname);
      return 0;
    case 'd':
      run_as_daemon = true;
      break;
    case '?':
      return -1;
    default:
      break;
    }
  }
  if (!run_as_daemon || (getuid() != 0)) {
    log::add_terminal_sink();
  }
  log::add_syslog_sink();
  if (getuid() == 0) {
    if (log_path.empty()) {
      log_path = "/var/log/" + progname;
    }
  }
  try {
    if (!log_path.empty()) {
      log::add_file_sink(log_path);
    }
  } catch(const std::exception& e) {
    log::error("%s - no logging to file available!", e.what());
  }

  if (run_as_daemon) {
    if (getuid() != 0) {
      log::error("insufficient user privileges - only root can run AEON "
                   "service as daemon!");
      exit(EXIT_FAILURE);
    } else {
      if (pidfile::check()) {
        log::add_terminal_sink();
        log::alert("AEON serivce is already running, exiting...");
        exit(EXIT_SUCCESS);
      }
    }
  }
  if (addr.empty()) {
    log::critical("address of network interface is not given, "
                    "terminating...");
    return -1;
  }
  if (port.empty()) {
    log::warning("port number is missing - using default port number");
    port = "4586";
  }
  if (run_as_daemon) {
    pid_t pid{ fork() };
    if (pid < 0) {
      log::debug("main(): fork() failed (errno=%d)", errno);
      exit(EXIT_FAILURE);
    }
    if (pid > 0) {
      log::info("child #1 process exited!");
      exit(EXIT_SUCCESS);
    }
    if (setsid() < 0) {
      log::debug("main(): setsid() failed (errno=%d)", errno);
      exit(EXIT_FAILURE);
    }
    signal(SIGCHLD, SIG_IGN);
    pid = fork();
    if (pid < 0) {
      log::debug("main(): fork() failed (errno=%d)", errno);
      exit(EXIT_FAILURE);
    }
    if (pid > 0) {
      log::info("child #2 process exited!");
      exit(EXIT_SUCCESS);
    }
    umask(0);
    if (chdir("/") < 0) {
      log::debug("main(): chdir() failed (errno=%d)", errno);
      exit(EXIT_FAILURE);
    }
    for (int i{ (int)sysconf(_SC_OPEN_MAX) }; i > 0; i--) {
      close(i);
    }
    int t{ open("/dev/null", O_RDWR) };
    dup(t);
    dup(t);
    try {
      pidfile::create();
    } catch (const pidfile::exception::open &e) {
      log::critical("unable to create a pid file - %s,"
                      "terminating service...", e.error_message());
      exit(EXIT_FAILURE);
    } catch (const pidfile::exception::lock &e) {
      log::critical("failure to lock the pid file - %s,"
                      "terminating service...", e.error_message());
      exit(EXIT_FAILURE);
    } catch (const pidfile::exception::write &e) {
      log::critical("failure writing to the pid file - %s,"
                      "terminating service...", e.error_message());
      exit(EXIT_FAILURE);
    }
  }
  if (on_exit(detail::fini, nullptr)) {
    exit(EXIT_FAILURE);
  }

  log::info("starting AEON service...");

  signal(SIGINT, detail::sighandler);
  signal(SIGHUP, detail::sighandler);
  signal(SIGTERM, detail::sighandler);

  log::info("signal handlers installed");
  try {
    detail::read_configuration(cfg_path);
  } catch(std::exception& e) {
    log::error("failure to read configuration - %s"
                "terminating...", e.what());
    exit(EXIT_FAILURE);
  }
  log::info("configuration read successfully");
  try {
    aeon::service service{ addr + ":" + port };
  } catch(const web::uri_exception& e) {
    log::critical("unable to start AEON service - %s, "
        "terminating...", e.what());
    exit(EXIT_FAILURE);
  }
  log::info("AEON service started");
  while (detail::terminate == 0) {
    if (detail::configure != 0) {
      log::info("configuration reloaded successfully");
      detail::read_configuration(cfg_path);
    }
    sleep(5);
  }
  log::info("AEON service stopped");
  exit(EXIT_SUCCESS);
}
