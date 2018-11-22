# Adv-Demog-Text



## Adversarial Removal of Demographic Attributes from Text Data

This is the code used in the paper:

**"Adversarial Removal of Demographic Attributes from Text Data"**
Yanai Elazar and Yoav Goldberg. EMNLP 2018. [Paper](https://arxiv.org/pdf/1808.06640.pdf)


### Prerequisites
* python 2.7

```sh
# create a clean conda env
conda create -n adv-demog-text python==2.7 anaconda
source activate adv-demog-text

# additional python pacakges from requirements.txt
pip install -r requirements.txt

# install dynet version out of my branch
git clone git@github.com:yanaiela/dynet.git

# checkout to the relevant commit
git checkout 5c2a49f595a2a1e286609f348977b714b2db12a1

# now follow the manual source installation of dynet in:
# https://dynet.readthedocs.io/en/latest/python.html
```



### Acquiring the Data:
read the [README.md](src/data/README.md) in the src/data folder


### Running the Models:
The main components of each experiment are the `trainer.py` 
and the `attacker.py` which correspond to the *Adversarial Training*
and *Attacker Network* respectively. 

NOTE: The first class is also used to train the baseline models.

Each script has its parameters to run with, and the overall experiments
used for this work can be found in the [runs.md](runs.md) file.

For running these models: 
* First, update the [consts.py](src/models/consts.py)
file and change the `tensorboard_dir` parameter to the tensorboard
directory
* install dynet from the source code. you'll need to use a certain
fork: https://github.com/yanaiela/dynet. clone it and then
follow the manual instructions from dynet: http://dynet.readthedocs.io/en/latest/python.html
* now each line in the [runs.md](runs.md) should be working

### Citing
If you find this work relevant to yours, please consider citing us:
```
@InProceedings{elazar:2018,
 author    = {Elazar, Yanai and Goldberg, Yoav},
 title     = {Adversarial Removal of Demographic Attributes from Text Data},
 booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
 year      = {2018}
}
```


### Contact
If you have any question, issue or suggestions, feel free to contact 
us with the emails listed in the paper.




<p><small>This project is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

