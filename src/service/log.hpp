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

#if !defined(AEON_Service_LOG_HPP_INCLUDED_)
#define AEON_Service_LOG_HPP_INCLUDED_

#pragma once

#include <syslog.h>

#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <locale>
#include <fstream>
#include <sstream>
#include <iostream>

#include <boost/filesystem/path.hpp>
#include <boost/format.hpp>

namespace nervana
{
    namespace log
    {
        enum class level : int
        {
            Quiet = 0,
            Emerg,
            Alert,
            Critical,
            Error,
            Warning,
            Notice,
            Info,
            Debug
        };

        namespace detail
        {
            struct sink
            {
                virtual ~sink() = default;

                virtual void emergency(const std::string&) = 0;
                virtual void alert(const std::string&)     = 0;
                virtual void notice(const std::string&)    = 0;
                virtual void info(const std::string&)      = 0;
                virtual void warning(const std::string&)   = 0;
                virtual void error(const std::string&)     = 0;
                virtual void debug(const std::string&)     = 0;
                virtual void critical(const std::string&)  = 0;
            };

            class syslog : public sink
            {
            public:
                syslog() noexcept { ::openlog(nullptr, LOG_PID, LOG_DAEMON); }
                ~syslog() override { ::closelog(); }
                syslog(syslog&&) noexcept = delete;
                syslog& operator=(syslog&&) noexcept = delete;
                syslog(const syslog&)                = delete;
                syslog& operator=(const syslog&) = delete;

                void emergency(const std::string& msg) override
                {
                    ::syslog(LOG_EMERG, "%s", msg.c_str());
                }
                void alert(const std::string& msg) override
                {
                    ::syslog(LOG_ALERT, "%s", msg.c_str());
                }
                void notice(const std::string& msg) override
                {
                    ::syslog(LOG_NOTICE, "%s", msg.c_str());
                }
                void info(const std::string& msg) override
                {
                    ::syslog(LOG_INFO, "%s", msg.c_str());
                }
                void warning(const std::string& msg) override
                {
                    ::syslog(LOG_WARNING, "%s", msg.c_str());
                }
                void error(const std::string& msg) override
                {
                    ::syslog(LOG_ERR, "%s", msg.c_str());
                }
                void debug(const std::string& msg) override
                {
                    ::syslog(LOG_DEBUG, "%s", msg.c_str());
                }
                void critical(const std::string& msg) override
                {
                    ::syslog(LOG_CRIT, "%s", msg.c_str());
                }
            };

            class stamp : public std::streambuf
            {
            public:
                explicit stamp(std::basic_ios<char_type>&         out,
                               std::function<void(std::ostream&)> get = [](std::ostream&) {})
                    : m_out{out}
                    , m_get{std::move(get)}
                    , m_buffer{out.rdbuf(this)}
                {
                }
                ~stamp() override { m_out.rdbuf(m_buffer); }
            protected:
                int_type overflow(int_type ch) override
                {
                    if (traits_type::eq_int_type(ch, traits_type::eof()))
                    {
                        return m_buffer->pubsync() == -1 ? ch : traits_type::not_eof(ch);
                    }
                    if (m_newline)
                    {
                        std::ostream s{m_buffer};
                        m_get(s);
                        if (!s)
                        {
                            return traits_type::eof();
                        }
                    }
                    m_newline = traits_type::to_char_type(ch) == '\n';
                    return m_buffer->sputc(ch);
                }

            private:
                std::basic_ios<char_type>&         m_out;
                bool                               m_newline{true};
                std::function<void(std::ostream&)> m_get;
                std::streambuf*                    m_buffer;
            };

            class stream : public sink
            {
            public:
                void emergency(const std::string& msg) override { output_e("EMERGENCY", msg); }
                void alert(const std::string& msg) override { output_e("ALERT", msg); }
                void notice(const std::string& msg) override { output_s("NOTICE", msg); }
                void info(const std::string& msg) override { output_s("INFO", msg); }
                void warning(const std::string& msg) override { output_e("WARNING", msg); }
                void error(const std::string& msg) override { output_e("ERROR", msg); }
                void debug(const std::string& msg) override { output_e("DEBUG", msg); }
                void critical(const std::string& msg) override { output_e("CRITICAL", msg); }
            protected:
                virtual void output_s(const std::string& prefix, const std::string& msg) = 0;
                virtual void output_e(const std::string& prefix, const std::string& msg) = 0;
            };

            class file : public stream
            {
            public:
                explicit file(const boost::filesystem::path& path)
                    : m_path{path}
                {
                }

                file(file&&) noexcept = delete;
                file& operator=(file&&) noexcept = delete;

                file() = delete;

                file(const file&) = delete;
                file& operator=(const file&) = delete;

            protected:
                void output_s(const std::string& prefix, const std::string& msg) final
                {
                    _output(prefix, msg);
                }
                void output_e(const std::string& prefix, const std::string& msg) final
                {
                    _output(prefix, msg);
                }

            private:
                boost::filesystem::path m_path;

                void _output(const std::string& prefix, const std::string& msg)
                {
                    std::stringstream ss;
                    detail::stamp     stamp{
                        ss, [&prefix](std::ostream& os) {
                            std::chrono::system_clock::time_point tp{
                                std::chrono::system_clock::now()};
                            std::time_t tt{std::chrono::system_clock::to_time_t(tp)};
                            std::tm*    tm{std::localtime(&tt)};
                            std::chrono::duration<double> sd{
                                tp - std::chrono::system_clock::from_time_t(tt) +
                                std::chrono::seconds{tm->tm_sec}};
                            os << std::setw(4) << std::setfill('0') << tm->tm_year + 1900 << "-"
                               << std::setw(2) << std::setfill('0') << tm->tm_mon << "-"
                               << std::setw(2) << std::setfill('0') << tm->tm_mday << " "
                               << std::setw(2) << std::setfill('0') << tm->tm_hour << ":"
                               << std::setw(2) << std::setfill('0') << tm->tm_min << ":"
                               << (sd.count() < 10 ? "0" : "") << std::fixed << sd.count() << " ["
                               << prefix << "] ";
                        }};
                    ss << msg << "\n";
                    std::ofstream ofs{m_path.string(), std::ios::ate | std::ios::app};
                    if (ofs.is_open())
                    {
                        ofs << ss.str() << std::flush;
                    }
                }
            };

            class terminal : public stream
            {
            public:
                terminal(terminal&&) noexcept = delete;
                terminal& operator=(terminal&&) noexcept = delete;

                terminal() = default;

                terminal(const terminal&) = delete;
                terminal& operator=(const terminal&) = delete;

            protected:
                void output_s(const std::string& prefix, const std::string& msg) final
                {
                    _output(std::cout, prefix, msg);
                }
                void output_e(const std::string& prefix, const std::string& msg) final
                {
                    _output(std::cerr, prefix, msg);
                }

            private:
                void _output(std::ostream& outs, const std::string& prefix, const std::string& msg)
                {
                    std::stringstream ss;
                    stamp             t{ss,
                            [&prefix](std::ostream& os) { os << "AEON Log > " << prefix << ": "; }};
                    ss << msg << "\n";
                    outs << ss.str() << std::flush;
                }
            };

            template <log::level Level>
            inline void write(detail::sink&, const std::string& msg)
            {
            }
            template <>
            inline void write<level::Debug>(detail::sink& sink, const std::string& msg)
            {
                sink.debug(msg);
            }
            template <>
            inline void write<level::Error>(detail::sink& sink, const std::string& msg)
            {
                sink.error(msg);
            }
            template <>
            inline void write<level::Critical>(detail::sink& sink, const std::string& msg)
            {
                sink.critical(msg);
            }
            template <>
            inline void write<level::Info>(detail::sink& sink, const std::string& msg)
            {
                sink.info(msg);
            }
            template <>
            inline void write<level::Warning>(detail::sink& sink, const std::string& msg)
            {
                sink.warning(msg);
            }
            template <>
            inline void write<level::Notice>(detail::sink& sink, const std::string& msg)
            {
                sink.notice(msg);
            }
            template <>
            inline void write<level::Emerg>(detail::sink& sink, const std::string& msg)
            {
                sink.emergency(msg);
            }
            template <>
            inline void write<level::Alert>(detail::sink& sink, const std::string& msg)
            {
                sink.alert(msg);
            }

            class logger
            {
            public:
                logger(logger&&) noexcept = delete;
                logger& operator=(logger&&) noexcept = delete;

                logger(const logger&) = delete;
                logger& operator=(const logger&) = delete;

                static void set_level(log::level level) noexcept
                {
                    logger::_get()._set_level(level);
                }

                static log::level get_level() noexcept { return logger::_get()._get_level(); }
                static void register_sink(detail::sink* sink)
                {
                    logger::_get()._register_sink(sink);
                }

                template <level Level>
                static void write(const std::string& msg)
                {
                    logger::_get()._write<Level>(msg);
                }

            private:
                log::level                                 m_level{level::Info};
                std::vector<std::unique_ptr<detail::sink>> m_sinks;

                logger() try
                {
                    _register_sink(new detail::syslog{});
                }
                catch (const std::exception&)
                {
                    // The registration of syslog sink may fail,
                    // however the creation of logger object must
                    // complete. Other sinks may be registered
                    // later successfully.
                }

                static logger& _get()
                {
                    static logger instance;
                    return instance;
                }

                void _set_level(log::level level) noexcept { m_level = level; }
                log::level                 _get_level() const noexcept { return m_level; }
                void _register_sink(detail::sink* sink) { m_sinks.emplace_back(sink); }
                template <log::level              Level>
                void _write(const std::string& msg) const
                {
                    if (m_level >= Level)
                    {
                        for (const auto& sink : m_sinks)
                        {
                            detail::write<Level>(*sink, msg);
                        }
                    }
                }
            };
        }

        inline void set_level(log::level level) noexcept { detail::logger::set_level(level); }
        inline log::level get_level() noexcept { return detail::logger::get_level(); }
        inline void add_terminal_sink() { detail::logger::register_sink(new detail::terminal{}); }
        inline void add_file_sink(const boost::filesystem::path& path)
        {
            detail::logger::register_sink(new detail::file{path});
        }

        namespace detail
        {
            inline void format(boost::format&) {}
            template <typename Arg, typename... Args>
            inline void format(boost::format& log, Arg const& arg, Args&&... args)
            {
                log % arg;
                detail::format(log, std::forward<Args>(args)...);
            }

            template <log::level level, typename... Args>
            inline void log(const std::string& format, Args&&... args)
            {
                boost::format log{format};
                detail::format(log, std::forward<Args>(args)...);
                logger::write<level>(log.str());
            };
        }

        template <typename... T>
        void emergency(const std::string& format, T&&... t)
        {
            detail::log<level::Emerg>(format, std::forward<T>(t)...);
        }

        template <typename... T>
        void alert(const std::string& format, T&&... t)
        {
            detail::log<level::Alert>(format, std::forward<T>(t)...);
        }

        template <typename... T>
        void notice(const std::string& format, T&&... t)
        {
            detail::log<level::Notice>(format, std::forward<T>(t)...);
        }

        template <typename... T>
        void error(const std::string& format, T&&... t)
        {
            detail::log<level::Error>(format, std::forward<T>(t)...);
        }

        template <typename... T>
        void warning(const std::string& format, T&&... t)
        {
            detail::log<level::Warning>(format, std::forward<T>(t)...);
        }

        template <typename... T>
        void info(const std::string& format, T&&... t)
        {
            detail::log<level::Info>(format, std::forward<T>(t)...);
        }

        template <typename... T>
        void debug(const std::string& format, T&&... t)
        {
            detail::log<level::Debug>(format, std::forward<T>(t)...);
        }

        template <typename... T>
        void critical(const std::string& format, T&&... t)
        {
            detail::log<level::Critical>(format, std::forward<T>(t)...);
        }
    }
}

#endif /* AEON_Service_LOG_H_INCLUDED_ */
