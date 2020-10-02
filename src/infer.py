# pylint: disable=invalid-name,missing-docstring,exec-used,too-many-arguments,too-few-public-methods,no-self-use
from __future__ import print_function

from color import Color
from daisy import Daisy
from DB import Database
from edge import Edge
from evaluate import infer
from gabor import Gabor
from HOG import HOG
from resnet import ResNetFeat
from vggnet import VGGNetFeat

depth = 5
d_type = 'd1'
query_idx = 0

if __name__ == '__main__':
    db = Database()

    query = samples[query_idx]

    # retrieve by color
    method = Color()
    samples = method.make_samples(db)
    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)

    # retrieve by daisy
    method = Daisy()
    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)

    # retrieve by edge
    method = Edge()
    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)

    # retrieve by gabor
    method = Gabor()
    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)

    # retrieve by HOG
    method = HOG()
    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)

    # retrieve by VGG
    method = VGGNetFeat()
    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)

    # retrieve by resnet
    method = ResNetFeat()
    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)
