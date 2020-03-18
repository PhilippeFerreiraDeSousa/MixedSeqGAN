# SeqGAN

## Requirements: 
* **Tensorflow r1.0.1**
* Python 2.7
* CUDA 7.5+ (For GPU)

## Introduction
Apply Generative Adversarial Nets to generating sequences of discrete tokens.

![](https://github.com/LantaoYu/SeqGAN/blob/master/figures/seqgan.png)

The illustration of SeqGAN. Left: D is trained over the real data and the generated data by G. Right: G is trained by policy gradient where the final reward signal is provided by D and is passed back to the intermediate action value via Monte Carlo search.  

The research paper [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](http://arxiv.org/abs/1609.05473) has been accepted at the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17).

We provide example codes to repeat the synthetic data experiments with oracle evaluation mechanisms.
To run the experiment with default parameters:
```
$ python sequence_gan.py
```
You can change the all the parameters in `sequence_gan.py`.

Note: this code is based on the [work from Lantao Yu](https://github.com/LantaoYu/SeqGAN), itself inspired by the [previous work by ofirnachum](https://github.com/ofirnachum/sequence_gan). Many thanks to both of them.
