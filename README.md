# Factorization model for recommendation

This project is my first attempt to create a recommendation system using tensorflow. My first idea was to contribute to https://github.com/songgc/TF-recomm . But since my code took its own direction I decided to create this repository instead. Nevertherless the core model implemented here is the same as the one from that repository. In the future I want to implement new recommendation models and have a more robust test framework. This is a one week project, so if it seems sloppy, it's not an accident.


### Requirements
* Tensorflow 
* Numpy
* Pandas 

## Usage

```
$ 
python3 recommender.py --help

optional arguments:
  -h, --help            show this help message and exit
  -d DIMENSION, --dimension DIMENSION
                        embedding vector size (default=15)
  -r REG, --reg REG     regularizer constant for the loss function
                        (default=0.05)
  -l LEARNING, --learning LEARNING
                        learning rate (default=0.001)
  -b BATCH, --batch BATCH
                        batch size (default=1000)
  -s STEPS, --steps STEPS
                        number of training (default=5000)
  -p PATH, --path PATH  ratings path (default=pwd/movielens/ml-1m/ratings.dat)

```


## Example

```
$
bash download_data.sh
python3 recommender.py -s 20000

>> step batch_error test_error elapsed_time
  0 2.671705 2.537921* 0.343948(s)
1000 1.188460 1.142280* 1.973038(s)
2000 1.013221 1.061189* 1.945483(s)
3000 0.965705 0.959811* 1.962791(s)
4000 0.963447 0.957588* 1.954662(s)
5000 0.896872 0.915500* 1.929358(s)
6000 0.883620 0.906923* 1.968918(s)
7000 0.887137 0.858354* 1.970129(s)
8000 0.879952 0.923793 1.563606(s)
9000 0.893490 0.909151 1.583156(s)
10000 0.883148 0.906680 1.580827(s)
11000 0.887621 0.921126 1.520585(s)
12000 0.861341 0.924831 1.525608(s)
13000 0.840560 0.893960 1.585923(s)
14000 0.839716 0.877921 1.573293(s)
15000 0.832154 0.917722 1.579683(s)
16000 0.821299 0.925236 1.570992(s)
17000 0.851868 0.867002 1.593472(s)
18000 0.857696 0.868844 1.577381(s)
19000 0.853078 0.841532* 1.917003(s)
 
>> The duration of the whole training with 20000 steps is 34.79
seconds,which is equal to: 0:0:0:34 (DAYS:HOURS:MIN:SEC)
>> The mean square error the whole valid dataset is  0.867328
```
