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

#if !defined(AEON_Service_PIDFILE_H_INCLUDED_)
#define AEON_Service_PIDFILE_H_INCLUDED_

#include <unistd.h>
#include <fcntl.h>

#include <fstream>
#include <string>
#include <stdexcept>

#include <boost/filesystem/path.hpp>

namespace nervana
{
    namespace pidfile
    {
        namespace detail
        {
            constexpr char path[] = "/var/run/aeon-service.pid";
        }

        namespace exception
        {
            namespace detail
            {
                class exception
                {
                public:
                    explicit exception(int error_code) noexcept : m_error_code{error_code} {}
                    std::string error_message() const { return std::strerror(m_error_code); }
                private:
                    int m_error_code{0};
                };
            }
            class open : public detail::exception
            {
            public:
                open()
                    : detail::exception{errno}
                {
                }
            };
            class lock : public detail::exception
            {
            public:
                lock()
                    : detail::exception{errno}
                {
                }
            };
            class write : public detail::exception
            {
            public:
                write()
                    : detail::exception{errno}
                {
                }
            };
        }

        inline void create()
        {
            int fd{open(detail::path, O_WRONLY | O_CREAT, 0640)};
            if (fd < 0)
            {
                throw pidfile::exception::open{};
            }
            if (lockf(fd, F_TLOCK, 0) < 0)
            {
                close(fd);
                throw pidfile::exception::lock{};
            }
            std::string pid{std::to_string(getpid())};
            ssize_t     count{write(fd, pid.c_str(), pid.size())};
            close(fd);
            if (count < pid.size())
            {
                throw pidfile::exception::write{};
            }
        }

        inline void remove() { unlink(detail::path); }
        inline bool check()
        {
            pid_t pid{0};
            {
                std::ifstream ifs{detail::path};
                if (ifs.is_open())
                {
                    ifs >> pid;
                }
            }
            if (pid > 0)
            {
                return ::system(("kill -0 " + std::to_string(pid) + " 2>/dev/null").c_str()) == 0;
            }
            return false;
        }
    }
}

#endif /* AEON_Service_PIDFILE_H_INCLUDED_ */
