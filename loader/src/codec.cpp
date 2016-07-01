#include "codec.hpp"

int Codec::_init = 0;

void raise_averror(const char* prefix, int errnum) {
    static char errbuf[512];
    av_strerror(errnum, &errbuf[0], 512);

    std::stringstream ss;
    ss << prefix << ": " << errbuf;
    throw std::runtime_error(ss.str());
}

int lockmgr(void **p, enum AVLockOp op) {
   mutex** mx = (mutex**) p;
   switch (op) {
   case AV_LOCK_CREATE:
      *mx = new mutex;
       break;
   case AV_LOCK_OBTAIN:
       (*mx)->lock();
       break;
   case AV_LOCK_RELEASE:
       (*mx)->unlock();
       break;
   case AV_LOCK_DESTROY:
       delete *mx;
       break;
   }
   return 0;
}
