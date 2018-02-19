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

<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='https://github.com/brianhuang1019/CBIR/blob/img/database.png' padding='5px'></img>

In my database, there are 25 classes, each class has its own directory,

and the images belong to this class should put into this directory.

### Part2: run the code
I implement several algorithm, you can run it with python3.

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
```python
python3 src/vggnet.py
```

#### For ResNet152
```python
python3 src/resnet.py
```

Above are basic usage of my codes.

There are some advanced issue such as features fusion and dimension reduction,

the intro of these parts will be written further in the future :D


## Author
Po-Chih Huang / [@brianhuang1019](http://brianhuang1019.github.io/)
