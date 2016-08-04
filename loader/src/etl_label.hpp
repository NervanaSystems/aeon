#pragma once

#include <sstream>

#include "interface.hpp"
#include "util.hpp"

namespace nervana {

    namespace label {
        class config;
        class decoded;

        class extractor;
        class loader;
    }

    class label::config : public interface::config {
    public:
        bool        binary = true;
        std::string type_string{"uint32_t"};

        config(nlohmann::json js)
        {
//            if(js.is_null()) {
//                throw std::runtime_error("missing label config in json config");
//            }

            for(auto& info : config_list) {
                info->parse(js);
            }
            verify_config(config_list, js);

            add_shape_type({1}, type_string);
        }

    private:
        config() {}
        std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
            ADD_SCALAR(binary, mode::OPTIONAL),
            ADD_SCALAR(type_string, mode::OPTIONAL)
        };
    };

    class label::decoded : public interface::decoded_media {
    public:
        decoded(int index) :
            _index{index} {}
        virtual ~decoded() override {}

        int get_index() { return _index; }

    private:
        decoded() = delete;
        int _index;
    };


    class label::extractor : public interface::extractor<label::decoded> {
    public:
        extractor(const label::config& cfg)
        {
            _binary = cfg.binary;
        }

        ~extractor() {}

        std::shared_ptr<label::decoded> extract(const char* buf, int bufSize) override
        {
            int lbl;
            if (_binary) {
                if (bufSize != 4) {
                    std::stringstream ss;
                    ss << "Only 4 byte buffers can be loaded as int32.  ";
                    ss << "label_extractor::extract received " << bufSize << " bytes";
                    throw std::runtime_error(ss.str());
                }
                lbl = unpack_le<int>(buf);
            } else {
                lbl = std::stoi(std::string(buf, (size_t) bufSize));
            }
            return std::make_shared<label::decoded>(lbl);
        }

    private:
        bool _binary = true;
    };

    class label::loader : public interface::loader<label::decoded> {
    public:
        loader(const label::config& cfg) : _cfg{cfg} {}
        ~loader() {}

        void load(const std::vector<void*>& buflist, std::shared_ptr<label::decoded> mp) override
        {
            char* buf = (char*)buflist[0];
            int index = mp->get_index();
            memcpy(buf, &index, _cfg.get_shape_type().get_otype().size);
        }
    private:
        const label::config& _cfg;
    };
}
