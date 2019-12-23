# Deep Neural Networks that Proves Geometry Theorem

## Introduction

We present the first deep neural networks that learn to prove simple Geometry theorems, producing proof steps of both deductive reasoning and creative construction. This contrasts with prior work where reasoning or construction follows hand-designed rules and heuristics. Our models learn from simply observing random sketches of points and lines on a blank canvas, i.e. no example proof designed by human is used. We control the random sketches to verify that the neural nets can indeed solve an unseen problem, as well as problems that require more reasoning steps than the ones they are trained on.


Project write-up: [Overleaf](https://www.overleaf.com/read/jjcgbdqkmzcz) (**Please do not share with anyone who does not have access to this Github repository.**)

![img](https://imgur.com/yJVCotO.png)

For the uninitiated in synthetic geometry theorem proof, we suggest having a look at [this report](https://www.imo-register.org.uk/2018-report-dominic.pdf) on the International Mathematical Olympiad 2018. Both solutions for Problem 1 in this report are perfect examples of elegant and simple geometry proofs. We should pursue to build a system that can attain this level of competence (and hence capable of competing at the IMO). Some nice results in Euclidean geometry includes the [Simson Line](https://en.wikipedia.org/wiki/Simson_line), or the [Butterfly Theorem](http://www.cut-the-knot.org/pythagoras/Butterfly.shtml) (Look for proof #14). The most famous theorem includes [Morley Trisector](https://en.wikipedia.org/wiki/Morley%27s_trisector_theorem), [Nine Point Circle](http://mathworld.wolfram.com/Nine-PointCircle.html), [Euler Line](https://en.wikipedia.org/wiki/Euler_line), or [Feuerbach point
](https://en.wikipedia.org/wiki/Feuerbach_point). Seeing these theorems can help readers appreciate the beauty of Geometry proof.

## Requirements

* python 2.x
* tensor2tensor
* sympy
* numpy


## Play with a trained model

First download the model trained with 20 steps of reasoning from here:



```bash
python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=path/to/model.ckpt-350000 \
--problem=geo_all20
```

