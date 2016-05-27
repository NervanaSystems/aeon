#pragma once
#include "etl_interface.hpp"
#include "params.hpp"

namespace nervana {
    class label_params : public parameter_collection {
    public:
        int ex_offset;

        int tx_scale;
        int tx_shift;

        int ld_width;

        float scale_pct;
        int channels;
        int angle;
        float cbs_range;
        float lighting_range;
        float aspect_ratio;
        bool area_scaling;
        bool flip;
        bool center;

        image_params() {
            // Required Params
            ADD_ARG(height, "image height", "h", "height");
            ADD_ARG(width, "image width", "w", "width");
            ADD_ARG(scale_pct, "percentage of original image to scale crop", "s1", "scale_pct");

            // Optionals with some standard defaults
            ADD_ARG(channels, "number of channels", "ch", "channels", 3, 1, 3);
            ADD_ARG(angle, "rotation angle", "angle", "rotate-angle", 0, 0, 90);
            ADD_ARG(cbs_range, "augmentation range in pct to jitter contrast, brightness, saturation", "c2", "cbs_range", 0.0, 0.0, 1.0);
            ADD_ARG(lighting_range, "augmentation range in pct to jitter lighting", "c2", "lighting_range", 0.0, 0.0, 0.2);
            ADD_ARG(aspect_ratio, "aspect ratio to jitter", "a1", "aspect_ratio", 1.0, 1.0, 2.0);
            ADD_ARG(area_scaling, "whether to use area based scaling", "a2", "area_scaling", false);
            ADD_ARG(flip, "randomly flip?", "f1", "flip", false);
            ADD_ARG(center, "always center?", "c1", "center", true);
        }
    };


    class label_settings : public parameter_collection {
    public:
        int scale;
        int shift;
    };


    class decoded_label : public decoded_media {
    public:
        decoded_label() {}
        decoded_label(int index) { _index = index; }
        virtual ~decoded_label() override {}

        inline MediaType get_type() override { return MediaType::TARGET; }
        inline int get_index() { return _index; }

    private:
        int _index;
    };


    class label_extractor : public extractor_interface {
    public:
        label_extractor(param_ptr pptr) :
            _ex_offset(pptr->ex_offset) {
        }
        ~label_extractor() {}
        media_ptr extract(char* buf, int bufSize) override {
            if (bufSize != 4) {
                throw std::runtime_error("Only 4 byte buffers can be loaded as int32")
            }
            return make_shared<decoded_label>(*reinterpret_cast<int *>(buf) + _ex_offset);
        }

    private:
        int _ex_offset;
    };

    class label_transformer : public transformer_interface {
    public:
        label_transformer(param_ptr pptr) :
            _tx_scale{pptr->tx_scale},
            _tx_shift{pptr->tx_shift} {
        }
        ~label_transformer() {}

        media_ptr transform(settings_ptr tx, const media_ptr& mp) override {
            int old_index = static_pointer_cast<decoded_label>(mp)->get_index();
            return make_shared<decoded_label>( old_index * _tx_scale + _tx_shift );
        }
        void fill_settings(settings_ptr tx) override {}

    private:
        int _tx_scale;
        int _tx_shift;
    };

    class bit_label_loader : public loader_interface {
    public:
        bit_label_loader(param_ptr pptr) :
            _ld_width{pptr->ld_width} {
        }
        ~bit_label_loader() {}
        void load(char* buf, int bufSize, const media_ptr& mp) override {
            if (_ld_width > bufSize) {
                throw std::runtime_error("load failed - buffer too small");
            }
        }

    private:
        int _ld_width;
    };

    class normal_label_loader : public loader_interface {
    public:
        normal_label_loader(param_ptr) {}
        ~normal_label_loader() {}
        void load(char* buf, int bufSize, const media_ptr& mp) override {

        }

    }

}
