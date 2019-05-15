# pylint: disable=invalid-name,missing-docstring,exec-used,too-many-arguments,too-few-public-methods,no-self-use
from __future__ import print_function

import sys
from src.color import Color
from src.daisy import Daisy
from src.DB import Database
from src.edge import Edge
from src.evaluate import infer
from src.gabor import Gabor
from src.HOG import HOG
from src.resnet import ResNetFeat
from src.vggnet import VGGNetFeat

depth = 5
d_type = 'd1'
query_idx = 0

if __name__ == '__main__':
    db = Database()

    # methods to call
    methods = {
        "color": Color,
        "daisy": Daisy,
        "edge": Edge,
        "hog": HOG,
        "gabor": Gabor,
        "vgg": VGGNetFeat,
        "resnet": ResNetFeat
    }

    try:
        mthd = sys.argv[1].lower()
    except IndexError:
        print("usage: {} <method>".format(sys.argv[0]))
        print("supported methods:\ncolor, daisy, edge, gabor, HOG, vgg, resnet")
        sys.exit(1)

    # call make_samples(db) accordingly
    samples = getattr(methods[mthd](), "make_samples")(db)

    # query the first img in data.csv
    query = samples[query_idx]

    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)
