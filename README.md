# Convergence of Gradient Descent: Linear Softmax on Fashion-MNIST

A step-by-step mathematical study of multinomial logistic regression (one affine layer + softmax) on Fashion-MNIST, as a baseline for analysing GD and SGD.

## Contents

- `gd_analysis.ipynb`: **Part 1.** Model, cross-entropy loss, first- and second-order derivatives, and a proof that the objective is convex (but not strictly convex) in $(W, b)$, with numerical checks.
- `concepting/gd_sgd_analysis.ipynb`: my first draft, written from scratch. It still contains mistakes (for example, the `.mean(dim=0)` gradient bug discussed in Section 5 of the main notebook) and is kept to show how the work developed.

## In progress

- $L$-smoothness of the objective and the $\mathcal O(1/T)$ convergence rate of full-batch GD.
- Extension to SGD.

## Process note

The first draft was written independently. The cleaned-up notebook was edited with AI assistance (wording, structure, and checking the derivations). I re-derived every result in Sections 1–8 myself.
