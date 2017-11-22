import sys
import numpy as np

from config import FEATURE_NAMES

feat = np.load("{}.npy".format(sys.argv[1]))

feat = np.array(FEATURE_NAMES)[feat[0]]

with open(sys.argv[1], 'w') as a_file:
    for f in feat:
        a_file.write("{}\n".format(f))
