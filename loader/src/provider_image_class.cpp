#include "provider_image_class.hpp"

using namespace nervana;

const nlohmann::json json_get(const nlohmann::json js, const std::string key, const nlohmann::json default_value) {
    // get the value in js at key.  If the value is null, return
    // default_value instead.
    auto value = js[key];
    if(value.is_null()) {
        return default_value;
    } else {
        return value;
    }
}

image_inference::image_inference(const nlohmann::json js) :
    image_config(js["image"]),
    image_extractor(image_config),
    image_transformer(image_config),
    image_loader(image_config),
    image_factory(image_config)
{
    num_inputs = 1;
    oshapes.push_back(image_config.get_shape_type());
}

void image_inference::provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) {
    std::vector<char>& datum_in  = in_buf[0]->getItem(idx);
    char* datum_out  = out_buf[0]->getItem(idx);

    if (datum_in.size() == 0) {
        std::stringstream ss;
        ss << "received encoded image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    // Process image data
    auto image_dec = image_extractor.extract(datum_in.data(), datum_in.size());
    auto image_params = image_factory.make_params(image_dec);
    image_loader.load(datum_out, image_transformer.transform(image_params, image_dec));
}

image_classifier::image_classifier(const nlohmann::json js) :
    image_config(js["image"]),
    // must use a default value {} otherwise, segfault ...
    label_config(json_get(js, "label", nlohmann::json("{}"))),
    image_extractor(image_config),
    image_transformer(image_config),
    image_loader(image_config),
    image_factory(image_config),
    label_extractor(label_config),
    label_loader(label_config)
{
    num_inputs = 2;
    oshapes.push_back(image_config.get_shape_type());
    oshapes.push_back(label_config.get_shape_type());
}

void image_classifier::provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) {
    std::vector<char>& datum_in  = in_buf[0]->getItem(idx);
    std::vector<char>& target_in = in_buf[1]->getItem(idx);
    char* datum_out  = out_buf[0]->getItem(idx);
    char* target_out = out_buf[1]->getItem(idx);

    if (datum_in.size() == 0) {
        std::stringstream ss;
        ss << "received encoded image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    // Process image data
    auto image_dec = image_extractor.extract(datum_in.data(), datum_in.size());
    auto image_params = image_factory.make_params(image_dec);
    image_loader.load(datum_out, image_transformer.transform(image_params, image_dec));

    // Process target data
    auto label_dec = label_extractor.extract(target_in.data(), target_in.size());
    label_loader.load(target_out, label_dec);
}

localization_decoder::localization_decoder(nlohmann::json js) :
    image_config(js["data_config"]["config"]),
    localization_config(js["target_config"]["config"]),
    image_extractor(image_config),
    image_transformer(image_config),
    image_loader(image_config),
    image_factory(image_config),
    localization_extractor(localization_config),
    localization_transformer(localization_config),
    localization_loader(localization_config)
{
    num_inputs = 2;
    oshapes.push_back(image_config.get_shape_type());
    oshapes.push_back(localization_config.get_shape_type());
}

void localization_decoder::provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) {
    std::vector<char>& datum_in  = in_buf[0]->getItem(idx);
    std::vector<char>& target_in = in_buf[1]->getItem(idx);

    char* datum_out  = out_buf[0]->getItem(idx);
    char* target_out = out_buf[1]->getItem(idx);

    if (datum_in.size() == 0) {
        std::stringstream ss;
        ss << "received encoded image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto image_dec = image_extractor.extract(datum_in.data(), datum_in.size());
    auto image_params = image_factory.make_params(image_dec);
    image_loader.load(datum_out, image_transformer.transform(image_params, image_dec));

    // Process target data
    auto target_dec = localization_extractor.extract(target_in.data(), target_in.size());
    localization_loader.load(target_out, localization_transformer.transform(image_params, target_dec));
}

bbox_provider::bbox_provider(nlohmann::json js) :
    image_config(js["data_config"]["config"]),
    bbox_config(js["target_config"]["config"]),
    image_extractor(image_config),
    image_transformer(image_config),
    image_loader(image_config),
    image_factory(image_config),
    bbox_extractor(bbox_config),
    bbox_transformer(bbox_config),
    bbox_loader(bbox_config)
{
    num_inputs = 2;
    oshapes.push_back(image_config.get_shape_type());
    oshapes.push_back(bbox_config.get_shape_type());
}

void bbox_provider::provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) {
    std::vector<char>& datum_in  = in_buf[0]->getItem(idx);
    std::vector<char>& target_in = in_buf[1]->getItem(idx);

    char* datum_out  = out_buf[0]->getItem(idx);
    char* target_out = out_buf[1]->getItem(idx);

    if (datum_in.size() == 0) {
        std::stringstream ss;
        ss << "received encoded image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto image_dec = image_extractor.extract(datum_in.data(), datum_in.size());
    auto image_params = image_factory.make_params(image_dec);
    image_loader.load(datum_out, image_transformer.transform(image_params, image_dec));

    // Process target data
    auto target_dec = bbox_extractor.extract(target_in.data(), target_in.size());
    bbox_loader.load(target_out, bbox_transformer.transform(image_params, target_dec));
}

provider_pixel_mask::provider_pixel_mask(nlohmann::json js) :
    image_config(js["data_config"]["config"]),
    target_config(js["target_config"]["config"]),
    image_extractor(image_config),
    image_transformer(image_config),
    image_loader(image_config),
    image_factory(image_config),
    target_extractor(target_config),
    target_transformer(target_config),
    target_loader(target_config)
{
}

void provider_pixel_mask::provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) {
    std::vector<char>& datum_in  = in_buf[0]->getItem(idx);
    std::vector<char>& target_in = in_buf[1]->getItem(idx);
    char* datum_out  = out_buf[0]->getItem(idx);
    char* target_out = out_buf[1]->getItem(idx);

    if (datum_in.size() == 0) {
        std::stringstream ss;
        ss << "received encoded image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto image_dec = image_extractor.extract(datum_in.data(), datum_in.size());
    auto image_params = image_factory.make_params(image_dec);
    auto image_transformed = image_transformer.transform(image_params, image_dec);
    image_loader.load(datum_out, image_transformed);

    // Process target data
    auto target_dec = image_extractor.extract(target_in.data(), target_in.size());
    auto target_transformed = target_transformer.transform(image_params, target_dec);
    image_loader.load(target_out, target_transformed);
}
