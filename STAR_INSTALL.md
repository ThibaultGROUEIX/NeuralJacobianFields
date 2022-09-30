# Install STAR

```
git submodule update --init --recursive
```
You need to download the STAR models, and of part the SURREAL database for a set of shape and pose parameters.

### STAR models

*  Download the models from our website <https://star.is.tue.mpg.de/> and place it under `dataset_generation/human_db/star_models`.
* Download the SURREAL database (2.5GB) from this link [drive](https://drive.google.com/file/d/1-Sinmau9KYwWj-9ANN7IKR8mm1FCBfvp/view?usp=sharing)  and place it under `dataset_generation/human_db/surreal_database/`. Make sure to respect SURREAL's [license](https://www.di.ens.fr/willow/research/surreal/data/license.html).




Your folder should look like

```
dataset_generation
|-human_db
    |-STAR
    |-star_generator.py
    |-surreal_database
        |-smpl_data.npz 
    |-star_models
        |-female
            |-model.npz
        |-male
            |-model.npz
        |-neutral
            |-model.npz
```

You should be all set to train while generating STAR data on the fly.