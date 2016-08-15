#include "provider_video_classifier.hpp"

using namespace nervana;
using namespace std;

video_classifier::video_classifier(nlohmann::json js) :
    video_config(js["video"]),
    label_config(js["label"]),
    video_extractor(video_config),
    video_transformer(video_config),
    video_loader(video_config),
    video_factory(video_config),
    label_extractor(label_config),
    label_loader(label_config)
{
    num_inputs = 2;
    oshapes.push_back(video_config.get_shape_type());
    oshapes.push_back(label_config.get_shape_type());
}

void video_classifier::provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf)
{
    std::vector<char>& datum_in  = in_buf[0]->get_item(idx);
    std::vector<char>& target_in = in_buf[1]->get_item(idx);
    char* datum_out  = out_buf[0]->get_item(idx);
    char* target_out = out_buf[1]->get_item(idx);

    if (datum_in.size() == 0) {
        std::stringstream ss;
        ss << "received encoded video with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    // Process video data
    auto video_dec = video_extractor.extract(datum_in.data(), datum_in.size());
    auto video_params = video_factory.make_params(video_dec);
    video_loader.load({datum_out}, video_transformer.transform(video_params, video_dec));

    // Process target data
    auto label_dec = label_extractor.extract(target_in.data(), target_in.size());
    label_loader.load({target_out}, label_dec);
}

