# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""
This must be kept in sync with loader/media.hpp.
"""

import logging
import json
import sys

class ParamDict(dict):
    def __setitem__(self, key, val):
        if key not in self:
            raise ValueError("Bad key: {}".format(key))
        _val_ = dict.__getitem__(self, key)
        if not isinstance(_val_, dict):
            if type(val) != type(_val_):
                raise ValueError("Types don't match {} vs {}".format(val, _val_))
            return dict.__setitem__(self, key, val)
        if type(val) != type(_val_['value']):
            raise ValueError("Types don't match {} vs {}".format(val, _val_['value']))
        vrange = _val_['valid']
        if isinstance(vrange, list):
            assert val in vrange, "{} not in list {}".format(val, vrange)
        elif isinstance(vrange, dict):
            _min, _max = vrange.get('min', -float('inf')), vrange.get('max', float('inf'))
            assert _min <= val <= _max, "{} not between {} and {}".format(val, _min, _max)
        _val_['value'] = val

    def __getitem__(self, key):
        _val_ = dict.__getitem__(self, key)
        if not isinstance(_val_, dict):
            return _val_
        else:
            return _val_['value']

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).iteritems():
            self[k] = v

with open(sys.argv[1], 'r') as f:
    a = json.load(f)

a = ParamDict(**a['imagenet'])
a.update(**dict(channels=1, height=2, subtract_mean=True))

for k in a:
    print("{}:  {}".format(k, a[k]))

