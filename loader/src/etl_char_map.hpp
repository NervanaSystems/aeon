#pragma once

#include <string>
#include <vector>
#include <istream>
#include <unordered_map>
#include <cstdint>

#include "interface.hpp"

namespace nervana {
    namespace char_map {
        class decoded;
        class extractor;
        class loader;
        class config;
    }

    class char_map::config : public interface::config {
        friend class extractor;
    public:
        uint32_t        max_length;
        std::string     alphabet;
        bool            pack_for_ctc = false;
        std::string     type_string{"uint8_t"};

        config(nlohmann::json js) {
            if(js.is_null()) {
                throw std::runtime_error("missing char_map config in json config");
            }

            for(auto& info : config_list) {
                info->parse(js);
            }
            verify_config(config_list, js);

            // Now fill in derived
            otype = nervana::output_type(type_string);
            shape = std::vector<uint32_t> {1, max_length};

            uint8_t index = 0;
            for (auto c: alphabet)
            {
                _cmap.insert({std::toupper(c), index++});
            }
            validate();
        }

        const std::unordered_map<char, uint8_t>& get_cmap() const {return _cmap;}

    private:
        std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
            ADD_SCALAR(max_length, mode::REQUIRED),
            ADD_SCALAR(alphabet, mode::REQUIRED),
            ADD_SCALAR(pack_for_ctc, mode::OPTIONAL),
            ADD_SCALAR(type_string, mode::OPTIONAL),
        };
        std::unordered_map<char, uint8_t> _cmap;

        config() {}
        void validate() {
            if (type_string != "uint8_t") {
                throw std::runtime_error("Invalid load type for char map " + type_string);
            }
            if (!unique_chars(alphabet)) {
                throw std::runtime_error("alphabet does not consist of unique chars " + alphabet);
            }
            base_validate();
        }

        bool unique_chars(std::string test_string)
        {
            if (test_string.size() > UINT8_MAX)
            {
                return false;
            }

            std::sort(test_string.begin(), test_string.end());

            for (uint i=1; i<test_string.size(); i++)
            {
                if (test_string[i-1] == test_string[i])
                {
                    return false;
                }
            }
            return true;
        }
    };

    class char_map::decoded : public interface::decoded_media {
    public:
        decoded(std::vector<uint8_t> char_ints, uint32_t nvalid)
        : _labels{char_ints}, _nvalid{nvalid}
        {}

        virtual ~decoded() {}

        std::vector<uint8_t> get_data() const { return _labels; }
        uint32_t get_length() const { return _nvalid; }

    private:
        std::vector<uint8_t>    _labels;
        uint32_t                _nvalid;
    };

    class char_map::extractor : public interface::extractor<char_map::decoded> {
    public:
        extractor( const char_map::config& cfg)
        : _cmap{cfg.get_cmap()}, _max_length{cfg.max_length}
        {}
        virtual ~extractor(){}
        virtual std::shared_ptr<char_map::decoded> extract(const char*, int) override;
    private:
        const std::unordered_map<char, uint8_t>& _cmap;  // This comes from config
        uint32_t  _max_length;
    };

    class char_map::loader : public interface::loader<char_map::decoded> {
    public:
        loader( const char_map::config& cfg) {}
        virtual ~loader(){}
        virtual void load(char*, std::shared_ptr<char_map::decoded>) override;
    private:
    };
}
