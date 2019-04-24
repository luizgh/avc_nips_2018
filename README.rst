NIPS 2018 Adversarial Vision Challenge
======================================

Code to reproduce the attacks and defenses for the entries "JeromeR" in the `NIPS 2018 Adversarial Vision Challenge`_ (1st place on Untargeted attacks, 3rd place on Robust models and Targeted attacks)

Team name: LIVIA - ETS Montreal

Team members: `Jérôme Rony`_, Luiz Gustavo Hafemann

Overview
========

**Defense**: We trained a robust model with a new iterative gradient-based L2 attack that we propose 
(Decoupled Direction and Norm — DDN), that is fast enough to be used during training. 
In each training step, we find an adversarial example (using DDN) that is close to the decision 
boundary, and minimize the cross-entropy of this example. There is no change to the model architecture, 
nor any impact on inference time.

**Attacks**: Our attack is based on a collection of surrogate models (including robust models trained with DDN). 
For each model, we select two directions to attack: the gradient of the cross entropy loss for 
the original class, and the direction given by running the DDN attack. For each direction, we do a 
binary search on the norm to find the decision boundary. We take the best attack and refine it with a Boundary attack.

For more information on the DDN attack, refer to the paper_, and implementation_:

.. [1] Jérôme Rony, Luiz G. Hafemann, Luiz S. Oliveira, Ismail Ben Ayed, Robert Sabourin and  Eric Granger "Decoupling Direction and Norm for Efficient Gradient-Based L2 Adversarial Attacks and Defenses", arXiv:1811.09600


.. _NIPS 2018 Adversarial Vision Challenge: https://www.crowdai.org/challenges/nips-2018-adversarial-vision-challenge
.. _Jérôme Rony: http://github.com/jeromerony/
.. _paper: https://arxiv.org/abs/1811.09600
.. _implementation: https://github.com/jeromerony/fast_adversarial
.. _TinyImagenet: https://drive.google.com/open?id=1pvefHKnFqtyN8vfg5ScF6jwMCP8PNWSj
.. _resnet18_clean: https://storage.googleapis.com/luizgh-datasets/avc_models/resnet18_clean.pt
.. _resnext50_ddn: https://storage.googleapis.com/luizgh-datasets/avc_models/resnext50_32x4d_ddn.pt
.. _resnext50_32x4d: https://drive.google.com/open?id=1DAsFkKHFhMkAppp-L4nNN_QEcJkEwYq8

Installation
============

Clone this repository and install the dependencies by running ``pip install -r requirements.txt``

Download the TinyImagenet_ dataset and extract it:

.. code-block:: bash

    tar xvf tiny-imagenet-pytorch.tar.gz -C data

Optional: download trained models: resnext50_ddn_ (our robust model), resnet18_clean_ (not adversarially trained).

Training a model
================

Adversarially train a model (using the DDN attack) starting from an imagenet-pretrained resnext50_32x4d_ :

.. code-block:: bash

    python train_tiny_imagenet_ddn.py data --sf tiny_ddn --adv --max-norm 1 --arch resnext50_32x4d --pretrained


For monitoring training, you can start a visdom server, and then add the argument ``--visdom-port <port>`` to the
command above:

.. code-block:: bash

    python -m visdom.server -port <port>


Running the attack
==================

See "attack_example.py" for an example of the attack. If you downloaded the models from the Installation_ section,
you can run the following code:

.. code-block:: bash

    python attack_example.py --m resnet18_clean.pt --sm resnext50_32x4d_ddn.pt

This will create an attack against a resnet18 model, using an adversarially trained surrogate model.
