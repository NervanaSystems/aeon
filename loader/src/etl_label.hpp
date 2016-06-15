#pragma once
#include "etl_interface.hpp"
#include "params.hpp"
#include "util.hpp"

using namespace std;

namespace nervana {

    namespace label {
        class config;
        class decoded;

        class extractor;
        class transformer;
        class loader;
    }

    class label::config : public json_config_parser {
    public:
        config(std::string argString) {}
    };


    class label::decoded : public decoded_media {
    public:
        decoded(int index) :
            _index{index} {}
        virtual ~decoded() override {}

        inline MediaType get_type() override { return MediaType::TARGET; }
        inline int get_index() { return _index; }

    private:
        decoded() = delete;
        int _index;
    };


    class label::extractor : public interface::extractor<label::decoded> {
    public:
        extractor(shared_ptr<const label::config> cfg) {}

        ~extractor() {}

        std::shared_ptr<label::decoded> extract(const char* buf, int bufSize) override
        {
            if (bufSize != 4) {
                throw runtime_error("Only 4 byte buffers can be loaded as int32");
            }
            return make_shared<label::decoded>(unpack_le<int>(buf));
        }
    };

    class label::transformer : public interface::transformer<label::decoded, nervana::params> {
    public:
        transformer(shared_ptr<const label::config>) {}

        ~transformer() {}

        std::shared_ptr<label::decoded> transform(
                            std::shared_ptr<nervana::params> txs,
                            std::shared_ptr<label::decoded> mp) override { return mp; }
    };

    class label::loader : public interface::loader<label::decoded> {
    public:
        loader(shared_ptr<const label::config> cfg) {}

        ~loader() {}

        void load(char* buf, int bufSize, std::shared_ptr<label::decoded> mp) override
        {
            int index = static_pointer_cast<label::decoded>(mp)->get_index();
            memcpy(buf, &index, bufSize);
        }
    };
}
