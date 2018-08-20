- [x] Baselines
```bash
python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro -1 --task sentiment --type 1

python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task race --model sentiment-type:1-ro:-1/best_model --init 1

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task race --type 2

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task gender --type 2

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_gender --type 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --task gender --model sent_gender-type:1-ro:-1/best_model --init 1


python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --type 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --model mention_race-type:1-ro:-1/best_model --init 1


python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro -1 --task mention2_gender --type 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention2_gender --model mention2_gender-type:1-ro:-1/best_model --init 1

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro -1 --task mention2_gender --type 2

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro -1 --task mention_age --type 2

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro -1 --task mention_age --type 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_age --model mention_age-type:1-ro:-1/best_model --init 1
```

- [x] Unbalanced Sentiment
```bash
python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task unbalanced_race --type 2

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task unbalanced_race --type 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task unbalanced_race --model unbalanced_race-type:1-ro:-1/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-type:1-ro:-1/best_model --init 1

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:1.0/best_model --init 1


python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --adv_size 500
?python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:500/epoch_50 --adv_size 500

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --adv_size 1000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:1000/best_model --adv_size 1000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:1000/epoch_50 --adv_size 1000

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --adv_size 2000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:2000/epoch_50 --adv_size 2000

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --adv_size 5000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:5000/epoch_50 --adv_size 5000

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --adv_size 8000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:8000/epoch_50 --adv_size 8000


python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 2 --model unbalanced_race-n_adv:2-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 3
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 3 --model unbalanced_race-n_adv:3-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 5
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 5 --model unbalanced_race-n_adv:5-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 0.5 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:0.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.5 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:1.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 2.0 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:2.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 3.0 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:3.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 5.0 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:5.0/epoch_50

``` 

- [x] Adversarials
* Race Adversarial
```bash
python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/epoch_10
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/epoch_50 --init 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/epoch_60
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task unseen_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task unseen_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/epoch_50 --init 1

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 2 --model sent_race-n_adv:2-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 2 --model sent_race-n_adv:2-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 3
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 3 --model sent_race-n_adv:3-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 3 --model sent_race-n_adv:3-ro:1.0/epoch_50

python trainer.py --dynet-seed 123 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 5
python attacker.py --dynet-seed 123 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 5 --model sent_race-n_adv:5-ro:1.0/best_model
python attacker.py --dynet-seed 123 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 5 --model sent_race-n_adv:5-ro:1.0/epoch_50

```

* Other Adversarials
```bash

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_race --num_adv 1 --model mention_race-n_adv:1-ro:1.0/best_model
```

- [x] Age Adversarials
```bash
python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 2 --model mention_age-n_adv:2-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 2 --model mention_age-n_adv:2-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 3
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 3 --model mention_age-n_adv:3-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 3 --model mention_age-n_adv:3-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 5
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 5 --model mention_age-n_adv:5-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 5 --model mention_age-n_adv:5-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 500
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 500 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:500/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 500 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:500/epoch_100

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 1000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 1000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:1000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 2000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 2000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:2000/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 2000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:2000/epoch_100

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 5000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 5000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:5000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 8000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 8000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:8000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 15000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 15000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:15000/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 15000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:15000/epoch_100


python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 8000 --num_adv 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 2 --adv_size 8000 --model mention_age-n_adv:2-ro:1.0-adv_hid_size:8000/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 2 --adv_size 8000 --model mention_age-n_adv:2-ro:1.0-adv_hid_size:8000/best_model


python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 0.5 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:0.5/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:0.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.5 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:1.5/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:1.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 2.0 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:2.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:2.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 3.0 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:3.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:3.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 5.0 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:5.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:5.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:5.0/epoch_60
```

- [x] Gender2 Adversarials
```bash
python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 2 --model mention2_gender-n_adv:2-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 2 --model mention2_gender-n_adv:2-ro:1.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 2 --model mention2_gender-n_adv:2-ro:1.0/epoch_60

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 3
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 3 --model mention2_gender-n_adv:3-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 3 --model mention2_gender-n_adv:3-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 5
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 5 --model mention2_gender-n_adv:5-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 5 --model mention2_gender-n_adv:5-ro:1.0/epoch_50


python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --adv_size 500
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 500 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:500/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 500 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:500/epoch_60

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --adv_size 1000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 1000 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:1000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --adv_size 2000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 2000 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:2000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --adv_size 5000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 5000 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:5000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --adv_size 8000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 8000 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:8000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 0.5 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:0.5/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:0.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.5 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:1.5/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:1.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 2.0 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:2.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:2.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:2.0/epoch_60

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 3.0 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:3.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:3.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 5.0 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:5.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:5.0/epoch_50
```

- [x] lstm parameters reduction - Sentiment-Race
```bash
python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 200
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 200 --model sent_race-type:1-hid:200-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 100
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 100 --model sent_race-type:1-hid:100-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 50 --model sent_race-type:1-hid:50-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 10
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 10 --model sent_race-type:1-hid:10-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 2 --model sent_race-type:1-hid:2-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 1 --model sent_race-type:1-hid:1-ro:-1/best_model
```

- [x] lstm parameters reduction - Mention-Race
```bash
python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 200
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 200 --model mention_race-type:1-hid:200-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 100
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 100 --model mention_race-type:1-hid:100-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 50 --model mention_race-type:1-hid:50-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 10
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 10 --model mention_race-type:1-hid:10-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 2 --model mention_race-type:1-hid:2-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 1 --model mention_race-type:1-hid:1-ro:-1/best_model
```

- [x] Bigger Adversarial
```bash
python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --adv_size 500
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 500 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:500/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 500 --att_hid_size 500 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:500/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 500 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:500/epoch_50

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --adv_size 1000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 1000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:1000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 1000 --att_hid_size 1000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:1000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 1000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:1000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 2000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 2000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:2000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 2000 --att_hid_size 2000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:2000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 2000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:2000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 3000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 3000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:3000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 3000 --att_hid_size 3000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:3000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 3000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:3000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 5000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 5000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:5000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 5000 --att_hid_size 5000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:5000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 5000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:5000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 6000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 6000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:6000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 6000 --att_hid_size 6000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:6000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 6000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:6000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 8000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 8000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:8000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 8000 --att_hid_size 8000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:8000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 8000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:8000/epoch_50
```

- [x] Lambda Change
```bash
python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task sent_race
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task sent_race --model sent_race-n_adv:1-ro:0.5/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task sent_race --model sent_race-n_adv:1-ro:0.5/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task sent_race
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task sent_race --model sent_race-n_adv:1-ro:1.5/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task sent_race --model sent_race-n_adv:1-ro:1.5/best_model


python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task sent_race
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task sent_race --model sent_race-n_adv:1-ro:2.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task sent_race --model sent_race-n_adv:1-ro:2.0/best_model

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 3.0 --task sent_race
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task sent_race --model sent_race-n_adv:1-ro:3.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task sent_race --model sent_race-n_adv:1-ro:3.0/epoch_60
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task sent_race --model sent_race-n_adv:1-ro:3.0/best_model

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 5.0 --task sent_race
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task sent_race --model sent_race-n_adv:1-ro:5.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task sent_race --model sent_race-n_adv:1-ro:5.0/best_model
```

- [x] Race-Sentiment
```bash
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task race_sent --model race-type:2-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task race_sent --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task race_sent --num_adv 1 --model race_sent-n_adv:1-ro:1.0/best_model
```
