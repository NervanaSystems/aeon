import os
from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming
from neon.layers import Conv, Pooling, GeneralizedCost, Activation, Affine
from neon.layers import MergeSum, SkipNode, BatchNorm
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import DataLoader, ImageParams
from neon.callbacks.callbacks import Callbacks, BatchNormTuneCallback
import pdb
# setup data provider
parser = NeonArgparser(__doc__)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

shape = dict(channel_count=3, height=32, width=32)
train_params = ImageParams(center=False, aspect_ratio=110, **shape)
common = dict(target_size=1, nclasses=10)

train = DataLoader(set_name='train', repo_dir=args.data_dir, media_params=train_params,
                   shuffle=True, subset_percent=args.subset_pct, **common)

for x, t in train:
	#print x.shape
	#pdb.set_trace()
	pass
