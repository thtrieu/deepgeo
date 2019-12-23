# Deep Neural Networks that Proves Geometry Theorem

## Introduction

We present the first deep neural networks that learn to prove simple Geometry theorems, producing proof steps of both deductive reasoning and creative construction. This contrasts with prior work where reasoning or construction follows hand-designed rules and heuristics. Our models learn from simply observing random sketches of points and lines on a blank canvas, i.e. no example proof designed by human is used. We control the random sketches to verify that the neural nets can indeed solve an unseen problem, as well as problems that require more reasoning steps than the ones they are trained on.


Project write-up: [Overleaf](https://www.overleaf.com/read/jjcgbdqkmzcz) (**Please do not share with anyone who does not have access to this Github repository.**)

![img](https://imgur.com/yJVCotO.png)

For the uninitiated in synthetic geometry theorem proof, we suggest having a look at [this report](https://www.imo-register.org.uk/2018-report-dominic.pdf) on the International Mathematical Olympiad 2018. Both solutions for Problem 1 in this report are perfect examples of elegant and simple geometry proofs. We should pursue to build a system that can attain this level of competence (and hence capable of competing at the IMO). Some nice results in Euclidean geometry includes the [Simson Line](https://en.wikipedia.org/wiki/Simson_line), or the [Butterfly Theorem](http://www.cut-the-knot.org/pythagoras/Butterfly.shtml) (Look for proof #14). The most famous theorem includes [Morley Trisector](https://en.wikipedia.org/wiki/Morley%27s_trisector_theorem), [Nine Point Circle](http://mathworld.wolfram.com/Nine-PointCircle.html), [Euler Line](https://en.wikipedia.org/wiki/Euler_line), or [Feuerbach point
](https://en.wikipedia.org/wiki/Feuerbach_point). Seeing these theorems can help readers appreciate the beauty of Geometry.

This project is at its very early stage where we identify all the key challenges in problem formulation, knowledge representation, modeling, and learning. We then propose a set of solutions to these challenges and built an infrastructure around it. The project is a sanity check if the whole pipeline we imagined can work at all. In short we made 0 to 1. The current result on a simple theorem is promising, and we are optimistic about going from 1 to 100: scaling it up to a greater capacity.

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

After loading the model the script will display a prompt for you to enter the problem, everything starts with a random triangle ABC:

```bash
>>> Given triangle ABC. 
```

Try the following problem and parallelogram or the Thales theorem. Currently the syntax to enter the theorem is a bit weird but we hope it is not too hard to understand.

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

The script will save all the proof steps in `*.png` format to the `thales/` sub-folder. This usually takes a while but good for visualizing the proof since the description printed in text format is hard to read due to its weird syntax. To skip this time consuming step leave the file name empty. 
