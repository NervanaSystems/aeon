import _chromicorn
import numpy as np


class BBex(object):
    def consume(self, ary):
        self.ary = np.zeros(ary.T.shape)
        print ary
        self.ary[:] = ary.T

be = BBex()
_chromicorn.chromicorn(3, 2, be)

print be.ary
