# Master thesis.

This repository is the main source of code related to Kristian Brudeli's master thesis work, written autumn 2022 at the [Department of Engineering Cybernetics](https://www.ntnu.edu/itk) at NTNU.

## What is it about?

The project is about using model-based Reinforcement Learning algorithms to solve a Path-Following and Collision-Avoidance task. 

The project builds upon the custom [`openai/gym`](https://github.com/openai/gym)-environment [`EivMeyer/gym-auv`](https://github.com/EivMeyer/gym-auv). The existing work shows promising and exciting results, but we do not know a-priori what the system plans to do. Even the existing systems themselves do not know.

To improve on this, the master thesis will use the papers [Dreamer](https://danijar.com/project/dreamer/) and [PlaNet](https://danijar.com/project/planet/) in order to do Model-Based Reinforcement learning. The method learns a latent-space model as described in the PlaNet-paper, which may be used for either MPC-style planning (PlaNet) or for training without access to the environment itself (Dreamer).

In both cases, the method learns a model that should be able to predict future observations

This master thesis work aims to:
- Adapt the `Dreamer`-algorithm to work with `gym-auv`
- Assess the performance of latent-space methods as Dreamer on the `gym-auv`-environment
- Develop visualizations illustrating the method's predictive power in this context
- Use a Convolutional Neural Network (CNN) instead of Meyer's manually engineered preprocessing step ("Feasibility pooling"). 

### Related prior publications:
- [Meyer and Heiberg et. al. COLREG-Compliant Collision Avoidance for Unmanned Surface Vehicle using Deep Reinforcement Learning](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2773871)
- [Meyer et. al. (2020): Taming an autonomous surface vehicle for path following and collision avoidance using deep reinforcement learning](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2723515) (paper)
- [E. Meyer: Master thesis](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2780874) (master thesis)


## Setup

Create a [Conda](https://docs.conda.io/en/latest/) environment:
```
conda env create --file environment.yml
```
Activate the environment and run the code.



