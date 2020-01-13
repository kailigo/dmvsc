# Self-guided Deep Multi-view Subspace Clustering via Consensus Affinity Regularization
Tensorflow implementation for the following paper

[Kai Li](http://kailigo.github.io/), [Hongfu Liu](http://hongfuliu.com/), [Yulun Zhang](http://yulunzhang.com/), [Kunpeng Li](https://kunpengli1994.github.io/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/). "Self-guided Deep Multi-view Subspace Clustering via Consensus Affinity Regularization", TNNLS submission.

## Introduction
Multi-view subspace clustering (MVSC) takes advantage of the complementary information among multi-view data and seeks to get the subspace clustering result agreed on all available views. Though proved to be effective in some cases, existing MVSC methods are limited by the fact they perform subspace analysis on the raw features which are often of high dimensions and with noises. The performance is thus often not satisfactory. To remedy this, we propose a Self-guided Deep Multi-view Subspace Clustering (SDMSC) model which performs joint deep feature embedding and subspace analysis.  SDMSC comprehensively explores the multi-view data and seeks a consensus data affinity relationship not only agreed on all views but also all intermediate embedding spaces. With more constraints being cast, the data affinity relationship is supposed to be more reliably recovered. Besides, to secure effective deep feature embedding without label supervision, we propose to use the data affinity relationship obtained in the raw feature space as the supervision signals to self-guide the embedding process. With this strategy, the risk that our deep clustering model being trapped in bad local minimal is reduced, bringing us satisfactory clustering results in high possibility. Experiments on seven widely used datasets show the proposed method significantly outperforms the state-of-the-art clustering methods.

## Environment
We recommended the following dependencies.

* Python 2.7
* Tensorflow (1.2)


## Commands
```bash
python2 demo_bbcsports.py
```