# Revisiting Pseudo-Label for Single-Positive Multi-Label Learning

Code for "Revisiting Pseudo-Label for Single-Positive Multi-Label Learning" (ICML'23)

## Setting up environment

```
conda env create -f environment.yaml
```

## Install datasets
See the `README.md` file in the `data` directory for instructions on downloading and setting up the datasets.

## Running example

For example, training the model on dataset pascal, loss VIB, learning rate 0.0001, feature dimension 256, batch size 32

```
python train_vib.py -d pascal
```

## Citing this work
```latex

@InProceedings{pmlr-v202-liu23ar,
  title = 	 {Revisiting Pseudo-Label for Single-Positive Multi-Label Learning},
  author =       {Liu, Biao and Xu, Ning and Lv, Jiaqi and Geng, Xin},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {22249--22265},
  year = 	 {2023},
  volume = 	 {202},
  address = {Honolulu, Hawaii},
}

```
