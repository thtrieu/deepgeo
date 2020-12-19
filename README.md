# Deep Neural Networks that Proves Geometry Theorem

## Introduction

We present the first deep neural networks that learn to prove simple Geometry theorems, producing proof steps of both deductive reasoning and creative construction. This contrasts with prior work where reasoning or construction follows hand-designed rules and heuristics. Our models learn from simply observing random sketches of points and lines on a blank canvas, i.e. no example proof designed by human is used. We control the random sketches to verify that the neural nets can indeed solve an unseen problem, as well as problems that require more reasoning steps than the ones they are trained on.


Here is [the project write-up](https://www.overleaf.com/read/jjcgbdqkmzcz) as a rough description of some concepts used throughout this README and the codebase. **Please do not share the report with anyone who does not have access to this Github repository.**

![img](https://imgur.com/yJVCotO.png)

For the uninitiated in synthetic geometry theorem proof, we suggest having a look at [this report](https://www.imo-register.org.uk/2018-report-dominic.pdf) on the International Mathematical Olympiad 2018. Both solutions for Problem 1 in this report are perfect examples of elegant and simple geometry proofs. We should pursue to build a system that can attain this level of competence (and hence capable of competing at the IMO). Some nice results in Euclidean geometry includes the [Simson Line](https://en.wikipedia.org/wiki/Simson_line), or the [Butterfly Theorem](http://www.cut-the-knot.org/pythagoras/Butterfly.shtml) (Look for proof #14). The most famous theorem includes [Morley Trisector](https://en.wikipedia.org/wiki/Morley%27s_trisector_theorem), [Nine Point Circle](http://mathworld.wolfram.com/Nine-PointCircle.html), [Euler Line](https://en.wikipedia.org/wiki/Euler_line), or [Feuerbach point
](https://en.wikipedia.org/wiki/Feuerbach_point). Seeing these theorems can help readers appreciate the beauty of Geometry.

This project is at its very early stage where we identify all the key challenges in (1) problem formulation, (2) knowledge representation, (3) modeling, and learning. We then propose a set of solutions to these 3 challenges and built an infrastructure around it. The project is a sanity check if the whole pipeline we imagined as such can work at all.

## Requirements

* python 2.x
* tensor2tensor
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

**Possible room to improve:** Our current perception is that this random sketch generation is the bottleneck of the whole pipeline. The most time consuming component is implemented here: the subgraph isomorphism matching algorithm implemented in function **`trieu_graph_match.recursively_match(..)`**. Improving the speed of this algorithm might be crucial to scaling up. For example, with the default `max_construction=7` and `max_depth=45` above, it takes 4 processes ran in nearly a day to collect a few millions examples. One can argue that using the same amount of resource and targeting for the same amount of training examples, exploration at, say, `max_construction=50` and `max_depth=100` will give drastically higher quality training examples. The improvement in speed can be coming from a better algorithm, better parallelization or lower level language implementation.

To have a better look into this random generation process, try `interactive` mode:

```bash
python explore.py --mode=interactive
```

An example can be seen [here](https://github.com/thtrieu/deepgeo/blob/master/interactive_example.md)

## Generate TFrecords

To train the model using `tensor2tensor`, we have to convert the `numpy` arrays generated above into tfrecords. We do this using the `t2t_datagen` utility in `tensor2tensor` with customized Problem classes defined in `problem.py`. For example, the below script will generate tfrecords for training examples of depth 1 upto depth 20 into director `data_all20`, using problem `geo_all20`:


```bash
python t2t_datagen.py \
--problem=geo_all20 \
--tmp_dir=output_numpy \
--data_dir=data_all20 \
--alsologtostderr
```

We have already generated tfrecords for several different problems, stored in `*_tfrecords/` directories in this project's [Google Cloud Storage](https://console.cloud.google.com/storage/browser/geo_reasoning/?project=optimal-buffer-256200).

## Training

Tensorflow code to build model and default hparams is in `model.py`. To train a model, we use the `t2t_trainer` utility as follow:

```bash
python t2t_trainer.py \
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

To make use of Cloud TPUs and Google Cloud Storage, please refer to [this Colab](https://colab.research.google.com/drive/1kJ3nI6-EYy38mDbbQWBEg8rEpbOuL0MX) for an example. One can also run this [second Colab](https://colab.research.google.com/drive/1N55bMyX_p_NTskhRdRN8M0bsTLmFvCmK) in parallel to continously pick up checkpoints for validation set evaluation as well as monitoring training through Tensorboard.


# A rough guide to the algorithm & code

## Problem formulation
Take the following example of a theorem: 
Given triangle ABC with sides AB=AC. Prove that angle B and C are equal.

Any problem like such consists of 
* a premise ("triangle ABC with sides AB=AC")
* a goal ("angle B and C are equal") and 
* the proof is a series of steps. 
  
We introduce the notion of a **state**, which is what we know about the problem **after** each proof step. The premise is therefore the initial state, and after each step the state got updated and enlarged. After that last step, the state contains the goal. 

The proof of our example consists of 2 steps:

1. Create Bisector AD of angle A. Here State = {Triangle ABC, AB=AC}, Goal = "angle B = angle C" and Action = "Create Bisector AD".

2. Triangle ABD = Triangle ACD, therefore angle B = angle C. Here State = {Triangle ABC, AB=AC, AD, angle BAD = angle CAD}, Goal = "angle B = angle C" and Action = "Triangle ABC = Triangle ACD"

Since each step adds to the state, we'll call each step an action. The idea here is to collect data in the form of ((State, Goal), Action) then use them to train neural networks, where Input = (State, Goal) and Output = Action. Once we collect data and train a neural net, we can produce the above proof by

1. Apply the neural network on input (State = {Triangle ABC, AB=AC}, Goal="angle B = angle C"), and get the output "Create bisector AD". Add the consequence of this action to the State, which grows into State={Triangle ABC, AB=AC, AD, angle BAD = angle CAD}.

2. Apply the same neural network on new input (State = {Triangle ABC, AB=AC, AD, angle BAD = angle CAD}, Goal="angle B = angle C"), and get the output "Triangle ABC = Triangle ACD". Add the consequence of this action to State, which grows into {Triangle ABC, AB=AC, AD, angle BAD = angle CAD, angle B = angle C, ...}

After such two application of the same neural network, the State now contains the Goal, so the proof concludes. For more challenging problems, the idea is to apply the neural network as many times as necessary (maybe 20-30 steps) until the State contains the Goal.

Notice the two actions (1) Create Bisector and (2) Equal Triangles, are rules to enlarge a given State. They are the **axioms** of Euclidean Geometry: (1) Any angle has a bisector and (2) If two triangles have two pair of equal sides and equal angle in the middle, they are equal. 

There is only a limited number of such axioms (20 according to Hilbert/Tarski). Not only do we pick actions from such a limited set, but also pick from an extended set of axioms plus fundamental theorems, so that the proofs get shorter and therefore less challenging to learn.

## How are we presenting (Input, Output) to the neural network?

There are a two consideration:

1. (Input, Output) can only be in a few forms that are known to be processable by neural networks of today: Vectors that are aranged into grid/sequence/graph/tables. For each of these modality, there is a type of neural net that is good at learning it.
   
2. This representation should capture the symmetries of the task, i.e., if we represent the Input as text, then the following two input texts: (1) "Given triangle ABC with AB=AC" (2) "Given triangle XYZ with two equal sides XY=ZX" should give the same output. We do not know of any text-processing neural nets that guarantee this property (so text shouldn't be the choice). Examples of other symmetries: if we rearrange the premise clauses, changing name of lines/points, changing the language, etc. the output shouldn't change as well.

Once we pick a way to represent data that satisfies the two consideration, the corresponding neural net will have less to learn from scratch, because the symmetries of the task are built into this design choice.

Our choice here is that Input = (State, Goal) be presented as a Graph, and Output = Action be presented as a Sequence. The task is Graph2Seq, and the neural net has Graph Neural Net as encoder, and a Transformer as its decoder.

## How are we presenting Euclidean Geometry into graphs?

Graphs consist of Nodes and Edges. 

1. Nodes are either Point, Line, Segment, Angle, HalfPlane, Circle, SegmentLength, AngleMeasure, LineDirection.
   
2. Edges connects two nodes: PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane, LineContainsPoint, HalfPlaneContainsPoint, SegmentHasLength, AngleHasMeasure, LineHasDirection, Distinct.

Next section explain these node/edge types in more detail. This representation captures the symmetries in previous section, e.g. there is no need for naming points/lines (A, B, C, etc). In the context of our problem formulation, there are a few things to be represented into such graph:

1. The pair (State, Goal)

2. The list of all Actions (axioms/theorem) to pick from.

Here State can be converted into such Graph in a straightforward manner. The Goal is a new part of the State Graph: a new node of type AngleMeasure that are connected to two nodes of type Angle (correspond to angle B and C) in the Graph. So the Input (State, Goal) to the neural net is a single Graph where the State is its subgraph, and the Goal is the remainder subgraph.

Each axiom/theorem consists of two parts: a Premise and a Conclusion. Both are graphs as well. For each action, if the Premise is a subgraph of the State graph, and the conclusion is **not** a subgraph of the State graph, the action is applicable.

## Specific choices in Euclidean Geometry representations

As seen above, there are nodes of type SegmentLength, AngleMeasure, and LineDirection. These are "transitive value" types, any number of Segments with the same length will be connected to the same SegmentLength, so neural nets doesn't have to learn about transitivity inferences. Same for Angle - AngleMeasure, and Line - LineDirection.

For every Line, two Halfplanes are created. Angles are defined as the intersection of two Halfplanes (i.e. in the graph, each Angle node is connected to two Halfplane nodes by HalfplaneCoversAngle edges). All topological relationships are captured through HalfPlaneContainsPoint edges.

Distinct is the only type of edge that connects two nodes of the same type (Point and Point, Line and Line). It says the two objects are **for sure** distinct. Similar to topological relations, this relation is rarely mentioned and often implicitly assumed when human does Geometry, but is quite important. Most axiom/theorem's premise contains this type of relation. For example: the middle point M of BC and D coincides, so we cannot consider AMD to be a triangle and therefore theorems about equal triangles is not applicable, unless there is a Distinct edge between M and D.


## Input and Output to the neural network

Input: as mentioned before the input is a single Graph where the State is its subgraph, and the Goal is the remainder subgraph. For the neural net to differentiate between the two parts, we can simply add an indicator embedding vector to each node in the graph.

Output: This should be a sequence of symbols, where the first one being the type of action to perform, and the remaining symbols point to relevant parts in the State graph that correspond to the premise of the action.






### `action_chain_lib.py`

Implements 
