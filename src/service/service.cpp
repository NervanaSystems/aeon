#include <iostream>
#include <memory>
#include <sstream>

#include "service.hpp"
#include "json.hpp"
#include "loader.hpp"
#include "typemap.hpp"

#include "base64.hpp"

using namespace web::http;
using nlohmann::json;
using namespace std;
using namespace nervana;

namespace nervana {
  namespace aeon {

    namespace {
      constexpr char notFoundLoader[] = "loader doesn't exist";
      web::json::value notFoundJson();
    }

    uint32_t loader_manager::register_agent(const nlohmann::json &config) {
        lock_guard<mutex> lg(m_mutex);

        if (m_loaders.size() >= max_loader_number - 1)
            throw std::runtime_error("the number of loaders exceeded max value");

        uint32_t id;
        while (m_loaders.find(id = (m_id_generator() % max_loader_number)) != m_loaders.end()) {
        };

        m_loaders[id] = std::unique_ptr<aeon::loader_adapter>(new aeon::loader_adapter(config));
        INFO << "Created new session " << id;

        return id;
    }

    void loader_manager::unregister_agent(uint32_t id) {
        lock_guard<mutex> lg(m_mutex);

        auto it = m_loaders.find(id);
        if (it == m_loaders.end())
            throw invalid_argument(notFoundLoader);
        m_loaders.erase(it);
    }

    aeon::loader_adapter &loader_manager::loader(uint32_t id) {
        auto it = m_loaders.find(id);
        if (it == m_loaders.end())
            throw invalid_argument(notFoundLoader);
        return *it->second.get();
    }

// /////////////////////////////////////////////////////////////////////////////

    string loader_adapter::next() {
        lock_guard<mutex> lg(m_mutex);

        if (m_is_reset)
            m_is_reset = false;
        else
            m_loader.get_current_iter()++;

        if (m_loader.get_current_iter().positional_end()) {
            return string("");
        } else {
            // stringstream initialization is very costly, so we have to avoid it by using thread_local.
            // It has a drawback though that of increased memory consumption.
            // If this is a problem, we can consider using sprintf and allocating memory on each request.
            thread_local std::ostringstream ss;
            ss.seekp(ios::beg);
            ss << *m_loader.get_current_iter();
            return ss.str();
        }
    };

    void loader_adapter::reset() {
        lock_guard<mutex> lg(m_mutex);
        m_loader.reset();
        m_is_reset = true;
    };

    string loader_adapter::batch_size() const {
        return std::to_string(m_loader.batch_size());
    };

    string loader_adapter::names_and_shapes() const {
        std::stringstream ss;
        ss << m_loader.get_names_and_shapes();
        return string(ss.str());
    };

    string loader_adapter::batch_count() const {
        return std::to_string(m_loader.batch_count());
    }

    string loader_adapter::record_count() const {
        return std::to_string(m_loader.record_count());
    }

// ////////////////////////////////////////////////////////////////////////////

#define REGISTER_SRV_FUNCTION(f) process_func[#f] = &parser::f;

    parser::parser() {
        REGISTER_SRV_FUNCTION(batch_size);
        REGISTER_SRV_FUNCTION(reset);
        REGISTER_SRV_FUNCTION(names_and_shapes);
        REGISTER_SRV_FUNCTION(batch_count);
        REGISTER_SRV_FUNCTION(record_count);
    }

    json_response parser::post(std::string msg, std::string msg_body) {
        try {
            if (msg.substr(0, endpoint_prefix.length()) != endpoint_prefix)
                throw std::invalid_argument("Invalid prefix");
            web::json::value reply_json = web::json::value::object();
            reply_json["status"]["type"] = web::json::value::string("SUCCESS");
            reply_json["data"]["id"] = web::json::value::string(
              std::to_string(m_loader_manager.register_agent(json::parse(msg_body))));
            return json_response(status_codes::Accepted, reply_json);
        }
        catch (exception &ex) {
            string ex_msg(ex.what());
            if (ex_msg.find(notFoundLoader) != string::npos)
                return json_response(status_codes::NotFound, notFoundJson());
            web::json::value reply_json = web::json::value::object();
            reply_json["status"]["type"] = web::json::value::string("FAILURE");
            reply_json["status"]["description"] = web::json::value::string(ex.what());
            if (ex_msg.find("error when parsing") != string::npos) // manifest cannot be parsed
                return json_response(status_codes::BadRequest, reply_json);
            else
                return json_response(status_codes::InternalError, reply_json);
        }
    }

    statused_response<next_tuple> parser::get(std::string msg) {
        static const auto notFound = statused_response<next_tuple>(
          status_codes::NotFound, tuple<web::json::value, std::string>(notFoundJson(), ""));
        try {
            if (msg.substr(0, endpoint_prefix.length()) != endpoint_prefix)
                return notFound;

            msg.erase(0, endpoint_prefix.length() + 1);

            auto path = web::uri::split_path(msg);
            if (path.size() != 2)
                return notFound;

            int dataset_id = std::stoi(path[0]);

            if (path[1] == "next") {
                return next(m_loader_manager.loader(dataset_id));
            } else {
                auto it = process_func.find(path[1]);
                if (it == process_func.end())
                    return notFound;
                else
                    return statused_response<next_tuple>(
                      status_codes::OK,
                      make_tuple((this->*it->second)(m_loader_manager.loader(dataset_id)), ""));
            }
        }
        catch (exception &ex) {
            string ex_msg(ex.what());
            if (ex_msg.find(notFoundLoader) != string::npos)
                return notFound;
            web::json::value response_json = web::json::value::object();
            response_json["status"]["type"] = web::json::value::string("FAILURE");
            response_json["status"]["description"] = web::json::value::string(ex_msg);
            return statused_response<next_tuple>(status_codes::InternalError, make_tuple(response_json, ""));
        }
    }

    json_response parser::del(std::string msg) {
        try {
            if (msg.substr(0, endpoint_prefix.length()) != endpoint_prefix) {
                return { status_codes::NotFound, notFoundJson() };
            }
            msg.erase(0, endpoint_prefix.length() + 1);

            INFO << msg;
            m_loader_manager.unregister_agent(std::stoul(msg));

            web::json::value reply_json{ web::json::value::object() };
            reply_json["status"]["type"] = web::json::value::string("SUCCESS");
            return { status_codes::OK, reply_json };
        }
        catch (exception &ex) {
            string ex_msg(ex.what());
            if (ex_msg.find(notFoundLoader) != string::npos)
                return json_response(status_codes::NotFound, notFoundJson());
            web::json::value response_json = web::json::value::object();
            response_json["status"]["type"] = web::json::value::string("FAILURE");
            response_json["status"]["description"] = web::json::value::string(ex.what());
            return { status_codes::InternalError, response_json };
        }
    }

    statused_response<next_tuple> parser::next(loader_adapter &loader) {
        web::json::value response_json = web::json::value::object();
        string data = loader.next();

        if (!data.empty()) {
            response_json["status"]["type"] = web::json::value::string("SUCCESS");
            return statused_response<next_tuple>(status_codes::OK, make_tuple(response_json, data));
        } else {
            response_json["status"]["type"] = web::json::value::string("END_OF_DATASET");
            return statused_response<next_tuple>(status_codes::NotFound, make_tuple(response_json, ""));
        }
    }

    web::json::value parser::reset(loader_adapter &loader) {
        web::json::value response_json = web::json::value::object();

        loader.reset();
        response_json["status"]["type"] = web::json::value::string("SUCCESS");

        return response_json;
    }

    web::json::value parser::batch_size(loader_adapter &loader) {
        web::json::value response_json = web::json::value::object();

        response_json["status"]["type"] = web::json::value::string("SUCCESS");
        response_json["data"]["batch_size"] = web::json::value::string(loader.batch_size());
        return response_json;
    }

    web::json::value parser::batch_count(loader_adapter &loader) {
        web::json::value response_json = web::json::value::object();

        response_json["status"]["type"] = web::json::value::string("SUCCESS");
        response_json["data"]["batch_count"] = web::json::value::string(loader.batch_count());
        return response_json;
    }

    web::json::value parser::record_count(loader_adapter &loader) {
        web::json::value response_json = web::json::value::object();

        response_json["status"]["type"] = web::json::value::string("SUCCESS");
        response_json["data"]["record_count"] = web::json::value::string(loader.record_count());
        return response_json;
    }

    web::json::value parser::names_and_shapes(loader_adapter &loader) {
        web::json::value response_json = web::json::value::object();

        response_json["status"]["type"] = web::json::value::string("SUCCESS");
        response_json["data"]["names_and_shapes"] = web::json::value::string(loader.names_and_shapes());
        return response_json;
    }

    namespace {
      web::json::value notFoundJson() {
          web::json::value response_json = web::json::value::object();
          response_json["status"]["type"] = web::json::value::string("FAILURE");
          response_json["status"]["description"] = web::json::value::string("not found");
          return response_json;
      }
    }

  }
}
