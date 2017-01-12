# Recommender

This project is my first attempt to create a recommendation system using tensorflow. My first idea was to contribute to [TF-recomm](https://github.com/songgc/TF-recomm). But since my code took its own direction I decided to create this repository instead. Like that repository I am trying to implement the models presented in [Factorization Meets the Neighborhood](http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf). The only model implemented so far (SVDmodel) is the one described in section 2.3 of that paper. The equation is:

![equation](http://www.sciweavers.org/tex2img.php?eq=1%2Bsin%28mc%5E2%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)





This was a one week project. So it is very sloppy.


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
  0 5.084445 4.942362* 0.239430(s)
1000 0.929015 0.903099* 1.532874(s)
2000 0.906508 0.870875* 1.558963(s)
3000 0.912761 0.920425 1.288688(s)
4000 0.921081 0.935999 1.289051(s)
5000 0.938996 0.896651 1.294045(s)
6000 0.901377 0.940087 1.289458(s)
7000 0.921445 0.898008 1.293325(s)
8000 0.906284 0.914094 1.291086(s)
9000 0.935321 0.928757 1.290218(s)
10000 0.964433 0.891465 1.281327(s)
11000 0.920245 0.881295 1.294385(s)
12000 0.931613 0.923730 1.282514(s)
13000 0.933952 0.908893 1.278047(s)
14000 0.900219 0.944106 1.279870(s)
15000 0.910001 0.900789 1.294415(s)
16000 0.950454 0.902835 1.294333(s)
17000 0.922296 0.939806 1.279068(s)
18000 0.877817 0.888710 1.280873(s)
19000 0.904908 0.908056 1.277388(s)
 
>> The duration of the whole training with 20000 steps is 26.50 seconds,
which is equal to:  0:0:0:26 (DAYS:HOURS:MIN:SEC)
>> The mean square error of the whole valid dataset is  1.07996
>> Using our model for one specific user we predicted the score of 10 movies as:
[ 5.  4.  5.  5.  5.  5.  5.  5.  5.  5.]
And in reality the scores are:
[5 3 3 4 5 3 5 5 4 4]
```
