#pragma once
#include <random>
#include "etl_interface.hpp"
#include "params.hpp"

using namespace std;

namespace nervana {

    class label_settings : public parameter_collection {
    public:
        int scale;
        int shift;
        label_settings() {}
    };


    class label_params : public parameter_collection {
    public:
        int ex_offset;

        int tx_scale;
        int tx_shift;

        float ld_offset;
        bool ld_dofloat;

        label_params() {
            // Optionals with some standard defaults
            ADD_ARG(ex_offset, "offset to add on extract", "eo", "extract_offset", 0, -100, 100);
            ADD_ARG(tx_scale, "scale to multiply by on transform", "tsc", "transform_scale", 1, -5, 5);
            ADD_ARG(tx_shift, "shift to multiply by on transform", "tsh", "transform_shift", 0, -200, 200);
            ADD_ARG(ld_offset, "offset to add on load if loading as float", "lo", "load_offset", 0.0, -0.9, 0.9);
            ADD_ARG(ld_dofloat, "load as a float?", "lf", "load_dofloat", false);
        }

        void fill_settings(settings_ptr stg, default_random_engine eng) {
            uniform_int_distribution<int> scale_rng(-tx_scale, tx_scale);
            uniform_int_distribution<int> shift_rng(-tx_shift, tx_shift);

            auto lstg = static_pointer_cast<label_settings>(stg);
            lstg->scale = scale_rng(eng);
            lstg->shift = shift_rng(eng);
        }
    };


    class decoded_label : public decoded_media {
    public:
        decoded_label(int index) :
            _index{index} {}
        virtual ~decoded_label() override {}

        inline MediaType get_type() override { return MediaType::TARGET; }
        inline int get_index() { return _index; }

    private:
        decoded_label() = delete;
        int _index;
    };


    class label_extractor : public extractor_interface {
    public:
        label_extractor(param_ptr pptr) {
            _ex_offset = static_pointer_cast<label_params>(pptr)->ex_offset;
        }

        ~label_extractor() {}

        media_ptr extract(char* buf, int bufSize) override {
            if (bufSize != 4) {
                throw runtime_error("Only 4 byte buffers can be loaded as int32");
            }
            return make_shared<decoded_label>(*reinterpret_cast<int *>(buf) + _ex_offset);
        }

    private:
        int _ex_offset;
    };


    class label_transformer : public transformer_interface {
    public:
        label_transformer(param_ptr pptr) {}

        ~label_transformer() {}

        media_ptr transform(settings_ptr tx, const media_ptr& mp) override {
            int old_index = static_pointer_cast<decoded_label>(mp)->get_index();
            shared_ptr<label_settings> txs = static_pointer_cast<label_settings>(tx);

            return make_shared<decoded_label>( old_index * txs->scale + txs->shift );
        }

        // Filling settings is done by the relevant params
        void fill_settings(settings_ptr tx) override {}

    };


    class label_loader : public loader_interface {
    public:
        label_loader(param_ptr pptr) {
            shared_ptr<label_params> lbl_ptr = static_pointer_cast<label_params>(pptr);
            _ld_offset = lbl_ptr->ld_offset;
            _ld_dofloat = lbl_ptr->ld_dofloat;
        }
        ~label_loader() {}
        void load(char* buf, int bufSize, const media_ptr& mp) override {
            int index = static_pointer_cast<decoded_label>(mp)->get_index();
            if (_ld_dofloat) {
                float ld_index = index + _ld_offset;
                memcpy(buf, &ld_index, bufSize);
            } else {
                memcpy(buf, &index, bufSize);
            }

        }

    private:
        float _ld_offset;
        bool _ld_dofloat;
    };

}
