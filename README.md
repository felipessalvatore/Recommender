# Recommender

This project is my first attempt to create a recommendation system using tensorflow. My first idea was to contribute to [TF-recomm](https://github.com/songgc/TF-recomm). But since my code took its own direction I decided to create this repository instead. Like that repository I am trying to implement the models presented in [Factorization Meets the Neighborhood](http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf) using the dataset [Movielens](http://grouplens.org/datasets/movielens/). The only model implemented so far (SVDmodel) is the one described in section 2.3 of that paper. The equation for the loss function of this model is:

![equation](http://www.sciweavers.org/tex2img.php?eq=min_%7Bp_%7B%2A%7D%2Cq_%7B%2A%7D%2Cb_%7B%2A%7D%7D%20%5Csum_%7B%28u%2Ci%29%20%5Cin%20K%7D%28r_%7Bui%7D%20-%5Cmu%20-b_%7Bu%7D%20-b_%7Bi%7D%20-p_%7Bu%7D%5E%7BT%7Dq_%7Bi%7D%29%5E%7B2%7D%20%2B%20%5Clambda_%7B3%7D%28%7C%7Cp_%7Bu%7D%7C%7C%5E%7B2%7D%20%2B%20%7C%7Cq_%7Bi%7D%7C%7C%5E%7B2%7D%20%2B%20b_%7Bu%7D%5E%7B2%7D%20%2B%20b_%7Bi%7D%5E%7B2%7D%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0[/img])


This was a one week project. So it is all very sloppy.


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
  -m MOMENTUM, --momentum MOMENTUM
                        momentum factor (default=0.9)
```


## Example

```
$
bash download_data.sh
python3 recommender.py -s 20000

>> step batch_error test_error elapsed_time
  0 4.073770 4.101110* 0.237425(s)
1000 0.931412 0.973595* 1.529423(s)
2000 0.913438 0.925822* 1.576900(s)
3000 0.926782 0.918592* 1.524989(s)
4000 0.901126 0.916897* 1.535359(s)
5000 0.899008 0.992615 1.285739(s)
6000 0.930125 0.930230 1.283590(s)
7000 0.938554 0.932051 1.291614(s)
8000 0.899618 0.914168* 1.566624(s)
9000 0.894170 0.933366 1.286368(s)
10000 0.925143 0.950335 1.295246(s)
11000 0.902990 0.901682* 1.518931(s)
12000 0.945003 0.945029 1.283664(s)
13000 0.922796 0.904910 1.291076(s)
14000 0.881359 0.926066 1.282692(s)
15000 0.926298 0.932974 1.283417(s)
16000 0.889660 0.911729 1.279145(s)
17000 0.927255 0.954901 1.300359(s)
18000 0.910293 0.937727 1.285614(s)
19000 0.954403 0.937461 1.293204(s)
 
>> The duration of the whole training with 20000 steps is 27.50 seconds,
which is equal to:  0:0:0:27 (DAYS:HOURS:MIN:SEC)

>> The mean square error of the whole valid dataset is  0.919379

>> Using our model for one specific user we predicted the score of 10 movies as:
[ 4.63046455  3.73141932  4.4607563   4.1839838   4.10344076  4.51128387
  4.15068197  4.45693159  4.20441914  4.53234148]

>> And in reality the scores are:
[5 3 3 4 5 3 5 5 4 4]

```
