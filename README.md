# Deep Neural Networks that Proves Geometry Theorem

## Introduction

We present the first deep neural networks that learn to prove simple Geometry theorems, producing proof steps of both deductive reasoning and creative construction. This contrasts with prior work where reasoning or construction follows hand-designed rules and heuristics. Our models learn from simply observing random sketches of points and lines on a blank canvas, i.e. no example proof designed by human is used. We control the random sketches to verify that the neural nets can indeed solve an unseen problem, as well as problems that require more reasoning steps than the ones they are trained on.


First please read [the write-up](https://www.overleaf.com/read/jjcgbdqkmzcz) for an introduction of some concepts used this project. **Please do not share with anyone who does not have access to this Github repository.**

![img](https://imgur.com/yJVCotO.png)

For the uninitiated in synthetic geometry theorem proof, we suggest having a look at [this report](https://www.imo-register.org.uk/2018-report-dominic.pdf) on the International Mathematical Olympiad 2018. Both solutions for Problem 1 in this report are perfect examples of elegant and simple geometry proofs. We should pursue to build a system that can attain this level of competence (and hence capable of competing at the IMO). Some nice results in Euclidean geometry includes the [Simson Line](https://en.wikipedia.org/wiki/Simson_line), or the [Butterfly Theorem](http://www.cut-the-knot.org/pythagoras/Butterfly.shtml) (Look for proof #14). The most famous theorem includes [Morley Trisector](https://en.wikipedia.org/wiki/Morley%27s_trisector_theorem), [Nine Point Circle](http://mathworld.wolfram.com/Nine-PointCircle.html), [Euler Line](https://en.wikipedia.org/wiki/Euler_line), or [Feuerbach point
](https://en.wikipedia.org/wiki/Feuerbach_point). Seeing these theorems can help readers appreciate the beauty of Geometry.

This project is at its very early stage where we identify all the key challenges in problem formulation, knowledge representation, modeling, and learning. We then propose a set of solutions to these challenges and built an infrastructure around it. The project is a sanity check if the whole pipeline we imagined can work at all. In short we made 0 to 1. The current result on a simple theorem is promising, and we are optimistic about going from 1 to 100: scaling model, data, compute up to a greater capacity.

## Requirements

* python 2.x
* tensor2tensor
* sympy
* numpy


## Try a trained model

First get the model trained with 20 steps of reasoning by downloading all files with name `model.ckpt-350000*` from [here](https://console.cloud.google.com/storage/browser/geo_reasoning/all20_modelv1_lr0d05/avg/?project=optimal-buffer-256200&pli=1).

Next run the interactive decode script:


```bash
python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=path/to/model.ckpt-350000 \
--problem=geo_all20
```

After loading the model the script will display a prompt for you to enter the problem in text with a particular syntax, everything starts with a random triangle ABC:

```bash
>>> Given triangle ABC. 
```

Try the following problem (about parallelogram or the Thales theorem). Currently the syntax to enter the question is a bit weird but we hope it is not too hard to understand.

```bash
# Let l1 go through A parallel to BC, let l2 go through C parallel to ab, D is the intersection of l1 and l2. Prove that DA=BC:
l1=parallel: A=A, l=bc. l2=parallel: A=C, l=ab. D=line_line:l1=l1,l2=l2. DA=BC

# Let M be mid point of AB. Let l go through M and parallel to BC. Let N be the intersection of l and AC. Prove that N is midpoint of AC:
M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. AN=CN
```

An example run look like this:

```bash
>>> Given triangle ABC. M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. AN=CN

 Working on it ..
Applied Construct Line: A=M B=C => ab_1=l2
Applied Construct Parallel Line: A=C l=ab => l2_0=l3
Applied Equal Angles Because Parallel: l=l2 l1=ab l2=l3 =>
Applied Equal Angles Because Parallel: l=ca l1=l3 l2=ab =>
Applied Equal Angles Because Parallel: l=l2 l1=bc l2=l =>
Applied Equal Triangles: Angle-Side-Angle: A=C B=B C=M D=M F=C de=l ef=l3 => E_0=P3
Applied Equal Angles Because Parallel: l=l l1=ab l2=l3 =>
Applied Equal Triangles: Angle-Side-Angle: A=A B=N C=M D=C F=P3 de=ca ef=l =>
Found goal!
Save sketch to file name: thales
Saving to thales/thales.png
```

The script will save all the proof steps in `*.png` format to the `thales/` sub-folder. This usually takes a while since there is a lot to render. However visualizing the proof is better than reading the description with a weird syntax in text format. To skip this time consuming step leave the file name empty.

## Play with generating random sketches

To generate random sketches and collect training examples, run:

```bash
python explore.py \
--mode=datagen \
--out_dir=output_numpy \
--max_construction=7 \
--max_depth=45 \
--max_line=6 \
--max_point=8 \
--explore_worker_id=1234
```

With the above (default) flags, we wish to start the exploration with 7 random constructions, followed by a series of deductions up to max depth 45 (another 38 steps). We also limit the total number of lines and points to 6 and 8 during construction. Notice during deduction, new lines and points can still be created. `explore_worker_id` will be used to random seed the current worker if several processes are running in parallel. The output of this process will be saved to directory `output_numpy/`. Here the state, goal and action will be converted into `numpy` matrices and serialized into binary files with name:

```bash
res.<worker_id>.depth.<depth>.part.<part>
```

For example, `res.002.depth.07.part.00031` contains training examples number 31000 to 31999 (read `part.00031`) that are collected by worker number 2 (read `res.002`), because each file contains exactly 1000 training examples. Each training example in this file is the correct action to take starting from a state that is 7 steps away from the goal (read `depth.07`). 

**Possible room to improve:** Our current perception is that this random sketch generation is the bottleneck of the whole pipeline. The most time consuming component is implemented here: the subgraph isomorphism matching algorithm implemented in `trieu_graph_match.py`. Improving the speed of this algorithm might be crucial to scaling up. For example, with the default `max_construction=7` and `max_depth=45` above, it takes 4 processes ran in nearly a day to collect a few millions examples. One can argue that using the same amount of resource and targeting for the same amount of training examples, exploration at, say, `max_construction=50` and `max_depth=100` will give drastically higher quality training examples. The improvement in speed can be coming from a better algorithm, better parallelization or lower level language implementation.

To have a better look into this random generation process, try `interactive` mode:

```bash
python explore.py --mode=interactive
```

An example can be seen [here](https://github.com/thtrieu/deepgeo/blob/master/interactive_example.md)

## Generate TFrecords

To train the model using `tensor2tensor`, we have to convert the `numpy` arrays generated above into tfrecords. We do this using the `t2t_datagen` utility in `tensor2tensor` with customized Problem classes defined in `problem.py`. For example, the below script will generate tfrecords for training examples of depth 1 upto depth 20 into director `data_all20`, using problem `geo_all20`:


```bash
t2t_datagen.py \
--problem=geo_all20 \
--tmp_dir=output_numpy \
--data_dir=data_all20 \
--alsologtostderr
```

We have already generated tfrecords for several different problems, stored in `*_tfrecords/` directories in this project's [Google Cloud Storage](https://console.cloud.google.com/storage/browser/geo_reasoning/?project=optimal-buffer-256200).

## Training

To train a model (specified in `model.py`), we use the `t2t_trainer` utility as follow:

```bash
!python t2t_trainer.py \
--problem=geo_all20 \
--data_dir=path/to/geo_all20_tfrecords \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--hparams="batch_size=32,learning_rate=0.05" \
--output_dir=path/to/ckpt_dir \
--train_steps=350000 \
--alsologtostderr \
--schedule=train
```

To make use of Cloud TPUs, please refer to [this Colab](https://colab.research.google.com/drive/1kJ3nI6-EYy38mDbbQWBEg8rEpbOuL0MX). One can also run this [second Colab](https://colab.research.google.com/drive/1N55bMyX_p_NTskhRdRN8M0bsTLmFvCmK) in parallel to continously pick up checkpoints for validation set evaluation as well as monitoring training through Tensorboard.


## What's next?

We look forward to the following:

* Faster subgraph isomorphism matching for deeper exploration.
* Add more eligible actions for wider exploration.
* Solving the two technical issues in modelling detailed in the project report, Section 5.


