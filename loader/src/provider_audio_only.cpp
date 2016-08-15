#include "provider_audio_only.hpp"

using namespace nervana;
using namespace std;

audio_only::audio_only(nlohmann::json js) :
    audio_config(js["audio"]),
    audio_extractor(),
    audio_transformer(audio_config),
    audio_loader(audio_config),
    audio_factory(audio_config)
{
    num_inputs = 1;
    oshapes.push_back(audio_config.get_shape_type());
}

void audio_only::provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf)
{
    vector<char>& datum_in  = in_buf[0]->get_item(idx);
    char* datum_out  = out_buf[0]->get_item(idx);

    if (datum_in.size() == 0) {
        cout << "no data " << idx << endl;
        return;
    }

    // Process audio data
    auto audio_dec = audio_extractor.extract(datum_in.data(), datum_in.size());
    auto audio_params = audio_factory.make_params(audio_dec);
    audio_loader.load({datum_out}, audio_transformer.transform(audio_params, audio_dec));
}
