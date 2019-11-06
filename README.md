# DRDSC

This is a TensorFlow implementation of the DRDSC model as described in our paper:

>Y. Jiang, Z. Yang, Q. Xu, X. Cao and Q. Huang. Duet Robust Deep Subspace Clustering. ACM MM 2019.

## Acknowledge
The implementation is based on [Deep Subspace Clustering Network](https://github.com/panji1990/Deep-subspace-clustering-networks). We sincerely thank the authors for their work.

## Dependencies
- Tensorflow
- numpy
- sklearn
- munkres
- scipy

## Data
The clean data and pre-trained auto-encoder model can be found [here](https://github.com/panji1990/Deep-subspace-clustering-networks/tree/master/Data). You can add any type of corruptions (e.g., Gaussian noise or random pixel corruption) on these data. Please put the data in `data/` and the pre-trained model in `models/`.

## Citation
Please cite our paper if you use this code in your own work:

```
@inproceedings{jiang2019duet,
  title={Duet Robust Deep Subspace Clustering},
  author={Jiang, Yangbangyan and Xu, Qianqian and Yang, Zhiyong and Cao, Xiaochun and Huang, Qingming},
  booktitle={{ACM} International Conference on Multimedia},
  pages={1596--1604},
  year={2019}
}
```

