# InceptionPC

This repo contains the implementation for the AAAI 2025 paper "On the Relationship between Monotone and Squared Probabilistic Circuits".

To run the experiments from the 20 binary datasets, run e.g. the following command:
```
python run_exp.py -ds nltcs -w 8 -u 8 -p complex 
```

To run the experiments for ImageNet32/ImageNet64, download and extract the ImageNet32/ImageNet64 npz files [here](https://www.image-net.org/), place them in the data directoru, and run e.g. the following command:
```
python run_exp.py -ds imagenet32 -w 4 -u 8 -ep 200
```

