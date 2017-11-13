![MIT](https://badges.frapsoft.com/os/mit/mit.svg?v=102)

# CBIR 
__This repo implements a CBIR (content-based image retrieval) system__
<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='https://github.com/brianhuang1019/CBIR/blob/img/CBIR.png' padding='5px' height="400px"></img>
<a href='https://winstonhsu.info/2017f-mmai/'>Image src</a>


## Part1: feature extraction

Implement several popular image features:
- color-based
  - [RGB histogram](https://github.com/brianhuang1019/CBIR/blob/master/src/color.py)
- texture-based
  - [gabor filter](https://github.com/brianhuang1019/CBIR/blob/master/src/gabor.py)
- shape-based
  - [daisy](https://github.com/brianhuang1019/CBIR/blob/master/src/daisy.py)
  - [edge histogram](https://github.com/brianhuang1019/CBIR/blob/master/src/edge.py)
  - [HOG (histogram of gradient)](https://github.com/brianhuang1019/CBIR/blob/master/src/HOG.py)
- deep methods
  - [VGG net](https://github.com/brianhuang1019/CBIR/blob/master/src/vggnet.py)
  - [Residual net](https://github.com/brianhuang1019/CBIR/blob/master/src/resnet.py)

##### *all features are modulized*

### Feature Fusion
Some features are not robust enough, and turn to feature fusion
- [fusion.py](https://github.com/brianhuang1019/CBIR/blob/master/src/fusion.py)

### Dimension Reduction
The curse of dimensionality told that vectors in high dimension will sometime lose distance property
- [Random Projection](https://github.com/brianhuang1019/CBIR/blob/master/src/random_projection.py)


## Part2: image retrieval

CBIR system retrieval k images based on features similarity (L1 distance)

Robustness of system is evaluated by MMAP (mean MAP)

- image AP   : mean of precision at each hit
- class1 MAP = (class1.img[0].AP + class1.img[1].AP + ... + class1.img[M].AP) / M
- MMAP       = (class1.MAP + class2.MAP + ... + classN.MAP) / N

<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='https://github.com/brianhuang1019/CBIR/blob/img/AP.png' padding='5px' height="400px"></img>
<a href='http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf'>Image src</a>

<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='https://github.com/brianhuang1019/CBIR/blob/img/MAP.png' padding='5px' height="400px"></img>
<a href='http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf'>Image src</a>

implementation of this part can found at [evaluate.py](https://github.com/brianhuang1019/CBIR/blob/master/src/evaluate.py)
