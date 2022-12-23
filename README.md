# HiBo-UA

### HiBo-UA Dataset
![avatar](https://github.com/jiwei0921/HiBo-UA/blob/main/HiBo.png)  

+ The HiBo-UA dataset can be approached in this link.
+ Meanwhile, we also provide state-of-the-art RGB-D methods' results on HiBo-UA dataset, and you can directly download their results.



### DCBF Code
This code is mainly based on our previous project ([DCF, CVPR21](https://github.com/jiwei0921/DCF)).

Stage 1: Run ```python demo_train_pre.py```, which performs the **D**epth **C**alibration Strategy.

Stage 2: Run ```python demo_train.py```, which performs the **F**usion Strategy.


### > Evaluation/Training Setup

+ The related all test datasets in this paper can be found in [this link (fetch code is **b2p2**)](https://pan.baidu.com/s/1sx1En1ecNyDf12jNGFeYZQ).
+ [This evaluation tool](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) is used to evaluate the above saliency maps in this paper.

+ The training set used in this paper can be accessed in [(NJUD+NLPR), code is **76gu**](https://pan.baidu.com/s/1sNxe3Szu7O_Qci1OGmKIKQ) and [(NJUD+NLPR+DUT), code is **201p**](https://pan.baidu.com/s/19aiosd_73VGMg7PB7HJzww).



### Acknowledgement

We thank all reviewers for their valuable suggestions. At the same time, thanks to the large number of researchers contributing to the development of open source in this field, particularly, [Deng-ping Fan](http://dpfan.net), [Runmin Cong](https://rmcong.github.io), [Tao Zhou](https://taozh2017.github.io), etc.

Our feature extraction network is based on [CPD backbone](https://github.com/wuzhe71/CPD).

### Bibtex
```
@article{Li_2022_DCBF,
    author    = {Li, Jingjing and Ji, Wei and Zhang, Miao and Piao, Yongri and Lu, Huchuan and Cheng, Li},
    title     = {Delving into Calibrated Depth for Accurate RGB-D Salient Object Detection},
    journal = {International Journal of Computer Vision},
    doi = {10.1007/s11263-022-01734-1},
    year      = {2022},
}
```

#### Contact Us
If you have any questions, please contact us ( wji3@ualberta.ca ).
