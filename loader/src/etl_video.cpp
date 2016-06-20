#include "etl_video.hpp"

using namespace std;
using namespace nervana;

void video::params::dump(ostream & ostr) {
    ostr << "FrameParams: ";
    _frameParams.dump(ostr);

    ostr << "Frames Per Clip: " << _framesPerClip << " ";
}
