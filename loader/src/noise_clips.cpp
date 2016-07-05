#include "noise_clips.hpp"
#include "audio.hpp"

using namespace std;

IndexElement::IndexElement() {
}

Index::Index() : _maxTargetSize(0) {
}

Index::~Index() {
    for (auto elem : _elements) {
        delete elem;
    }
}

void Index::load(std::string& fileName, bool shuf) {
    ifstream ifs(fileName);
    if (!ifs) {
        stringstream ss;
        ss << "Could not open " << fileName;
        throw std::ios_base::failure(ss.str());
    }

    std::string line;
    // Ignore the header line.
    std::getline(ifs, line);
    while (std::getline(ifs, line)) {
        if (line[0] == '#') {
            // Ignore comments.
            continue;
        }
        addElement(line);
    }

    if (shuf == true) {
        shuffle();
    }

    if (_elements.size() == 0) {
        stringstream ss;
        ss << "Could not load index from " << fileName;
        throw std::runtime_error(ss.str());
    }
}

IndexElement* Index::operator[] (int idx) {
    return _elements[idx];
}

uint Index::size() {
    return _elements.size();
}

void Index::addElement(std::string& line) {
    IndexElement* elem = new IndexElement();
    std::istringstream ss(line);
    std::string token;
    std::getline(ss, token, ',');
    elem->_fileName = token;
    while (std::getline(ss, token, ',')) {
        elem->_targets.push_back(token);
    }

    // For now, restrict to a single target.
    assert(elem->_targets.size() <= 1);
    _elements.push_back(elem);
    if (elem->_targets.size() == 0) {
        return;
    }
    if (elem->_targets[0].size() > _maxTargetSize) {
        _maxTargetSize = elem->_targets[0].size();
    }
}

void Index::shuffle() {
    std::srand(0);
    std::random_shuffle(_elements.begin(), _elements.end());
}

NoiseClips::NoiseClips(char* _noiseIndexFile, char* _noiseDir, Codec* codec)
: _indexFile(_noiseIndexFile), _indexDir(_noiseDir),
  _buf(0), _bufLen(0) {
    loadIndex(_indexFile);
    loadData(codec);
}

NoiseClips::~NoiseClips() {
    delete[] _buf;
}

void NoiseClips::addNoise(shared_ptr<RawMedia> media, NoiseClipsState* state) {
    if (state->_rng(2) == 0) {
        // Augment half of the data examples.
        return;
    }
    // Assume a single channel with 16 bit samples for now.
    assert(media->size() == 1);
    assert(media->bytesPerSample() == 2);
    int bytesPerSample = media->bytesPerSample();
    int numSamples = media->numSamples();
    cv::Mat data(1, numSamples, CV_16S, media->getBuf(0));
    cv::Mat noise(1, numSamples, CV_16S);
    int left = numSamples;
    int offset = 0;
    // Collect enough noise data to cover the entire input clip.
    while (left > 0) {
        std::shared_ptr<RawMedia> clipData = _data[state->_index];
        assert(clipData->bytesPerSample() == bytesPerSample);
        int clipSize = clipData->numSamples() - state->_offset;
        cv::Mat clip(1, clipSize , CV_16S,
                 clipData->getBuf(0) + bytesPerSample * state->_offset);
        if (clipSize > left) {
            const cv::Mat& src = clip(cv::Range::all(), cv::Range(0, left));
            const cv::Mat& dst = noise(cv::Range::all(), cv::Range(offset, offset + left));
            src.copyTo(dst);
            left = 0;
            state->_offset += left;
        } else {
            const cv::Mat& dst = noise(cv::Range::all(),
                                       cv::Range(offset, offset + clipSize));
            clip.copyTo(dst);
            left -= clipSize;
            offset += clipSize;
            next(state);
        }
    }
    // Superimpose noise without overflowing.
    cv::Mat convData;
    data.convertTo(convData, CV_32F);
    cv::Mat convNoise;
    noise.convertTo(convNoise, CV_32F);
    float noiseLevel = state->_rng.uniform(0.f, 2.0f);
    convNoise *= noiseLevel;
    convData += convNoise;
    double min, max;
    cv::minMaxLoc(convData, &min, &max);
    if (-min > 0x8000) {
        convData *= 0x8000 / -min;
        cv::minMaxLoc(convData, &min, &max);
    }
    if (max > 0x7FFF) {
        convData *= 0x7FFF / max;
    }
    convData.convertTo(data, CV_16S);
}

void NoiseClips::next(NoiseClipsState* state) {
    state->_index++;
    if (state->_index == _data.size()) {
        // Wrap around.
        state->_index = 0;
        // Start at a random offset.
        state->_offset = state->_rng(_data[0]->numSamples());
    } else {
        state->_offset = 0;
    }
}

void NoiseClips::loadIndex(std::string& indexFile) {
    _index.load(indexFile, true);
}

void NoiseClips::loadData(Codec* codec) {
    for (uint i = 0; i < _index.size(); i++) {
        std::string& fileName = _index[i]->_fileName;
        int len = 0;
        readFile(fileName, &len);
        if (len == 0) {
            stringstream ss;
            ss << "Could not read " << fileName;
            throw std::runtime_error(ss.str());
        }
        _data.push_back(codec->decode(_buf, len));
    }
}

void NoiseClips::readFile(std::string& fileName, int* dataLen) {
    std::string path;
    if (fileName[0] == '/') {
        path = fileName;
    } else {
        path = _indexDir + '/' + fileName;
    }
    struct stat stats;
    int result = stat(path.c_str(), &stats);
    if (result == -1) {
        stringstream ss;
        ss << "Could not find " << path;
        throw std::runtime_error(ss.str());
    }
    off_t size = stats.st_size;
    if (_bufLen < size) {
        resize(size + size / 8);
    }
    std::ifstream ifs(path, std::ios::binary);
    ifs.read(_buf, size);
    *dataLen = size;
}

void NoiseClips::resize(int newLen) {
    delete[] _buf;
    _buf = new char[newLen];
    _bufLen = newLen;
}
