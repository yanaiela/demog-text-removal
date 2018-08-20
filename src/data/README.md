# Preparing the datasets

## Race data
* Download the data from Su Lin Blodgett [dataset](https://sites.google.com/site/sulinblodgett/), described in "Demographic dialectal variation in social
media: A case study of african-american english." 
```sh
wget http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip

# or directly from the site: http://slanglab.cs.umass.edu/TwitterAAE/
```
* Then run the make_data.py script by:
```sh
python make_data.py /path/to/downloaded/twitteraae_all /path/to/project/data/processed/X_race X race
# where X is the main task - 'sentiment' or 'mention'
```


## Gender & Age data

* Download the data from the PAN16 challenge
```sh
wget https://www.uni-weimar.de/medien/webis/corpora/corpus-pan-labs-09-today/pan-16/pan16-data/pan16-author-profiling-training-dataset-2016-04-25.zip

# or directly from their site: https://pan.webis.de/clef16/pan16-web/author-profiling.html
```
* Extract the English dataset from the downloaded zip
* In the script `extract_pan16_data.py` fill in your twitter keys,
and the path to this project
* run the script from this directory:
```sh
python extract_pan16_data.py
```
* It will create a file containing all the tweets in the 
data/interim/author-profiling.tsv folder
* In the script make_author_data.py change the project variable 
to your project dir and then run it simply by
```sh
python make_author_data.py
```

* Due to the nature of twitters' data, some of the tweets might have been removed,
therefore, the resulting file might be different from the one used in this work.


## Unseen data (on sentiment/race, but can be easily extended to other)
* In the file make_unseen_data.py change the project with your path
* run that scipt with:
```sh
python make_unseen_data.py
``` 


