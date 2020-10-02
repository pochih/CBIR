## How to run the code?

Let me show you how to use the code.

It can divided to two parts.

### Part1: make your image database
When you clone the repository, it will look like this:

    ├── src/            # Source files
    ├── result/         # Results
    ├── USAGE.md        # How to use the code
    └── README.md       # Intro to the repo

you need to add your images into a directory called __database/__, so it will look like this:

    ├── src/            # Source files
    ├── result/         # Results
    ├── USAGE.md        # How to use the code
    ├── README.md       # Intro to the repo
    └── database/       # Directory of all your images

__all your image should put into database/__

In this directory, each image class should have its own directory

see the picture for details:

<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='https://github.com/pochih/CBIR/blob/img/database.png' padding='5px'></img>

In my database, there are 25 classes, each class has its own directory,

and the images belong to this class should put into this directory.

### Part2: run the code

#### For image retrieval test

I implement several algorithm, you can run it with python3.
If you need to test with one of the algorithms, please use `query.py <method to test>`, by default the image being queried is the first one in `data.csv`

result looks like

```text
$ .venv/bin/python query.py resnet
Using cache..., config=resnet152-avg, distance=d1, depth=3

[+] query: database/vim/vim-02.jpg

database/vim/vim-01.jpg:        0.4892213046550751,     Class vim
database/vim/vim-03.jpg:        0.5926028490066528,     Class vim
database/emacs/emacs-02.jpg:    0.7468970417976379,     Class emacs
database/emacs/emacs-01.jpg:    0.7663252353668213,     Class emacs
database/store/qingfeng-1.jpg:  0.818065881729126,      Class store

$ cat data.csv
img,cls
database/vim/vim-02.jpg,vim
database/store/qingfeng-1.jpg,store
database/vim/vim-04.jpg,vim
database/vim/vim-01.jpg,vim
database/vim/vim-03.jpg,vim
database/store/qingfeng-2.jpg,store
database/store/qingfeng.jpg,store
database/emacs/emacs-01.jpg,emacs
database/emacs/emacs-02.jpg,emacs
```

#### For RGB histogram
```python
python3 src/color.py
```

#### For daisy image descriptor
```python
python3 src/daisy.py
```

#### For gabor filter
```python
python3 src/gabor.py
```

#### For edge histogram
```python
python3 src/edge.py
```

#### For histogram of gradient (HOG)
```python
python3 src/HOG.py
```

#### For VGG19
You need to install pytorch0.2 to run the code
```python
python3 src/vggnet.py
```

#### For ResNet152
You need to install pytorch0.2 to run the code
```python
python3 src/resnet.py
```

Above are basic usage of my codes.

There are some advanced issue such as features fusion and dimension reduction,

the intro of these parts will be written further in the future :D

### Appendix: feature fusion
I implement the basic feature fusion method -- concatenation.

Codes for feature fusion is written in [fusion.py](https://github.com/pochih/CBIR/blob/master/src/fusion.py)

In fusion.py, there is a class called *FeatureFusion*.

You can create a *FeatureFusion* instance with an argument called **features**.

For example, in [fusion.py line140](https://github.com/pochih/CBIR/blob/master/src/fusion.py#L140)
```python
fusion = FeatureFusion(features=['color', 'daisy'])
APs = evaluate_class(db, f_instance=fusion, d_type=d_type, depth=depth)
```
- The first line means to concatenate color featrue and daisy feature.
- The second line means to evaluate with the concatenated feature.

If you want to know the performance of all possible feature combination, look at [fusion.py line122](https://github.com/pochih/CBIR/blob/master/src/fusion.py#L122) for example
```python
evaluate_feats(db, N=2, d_type='d1')
```
- Parameter *N* means how many feature you want to concatenate.
- Parameter *d_type* means the distance metric you want to use.
- Function *evaluate_feats* will generate a result file that record performances for all feature concatenation.

## Author
Po-Chih Huang / [@pochih](http://pochih.github.io/)
