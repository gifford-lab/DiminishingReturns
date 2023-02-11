This repository contains all the code needed to replicate the results from "Constrained Submodular Optimization for Vaccine Design". For more general details about out work, please see our paper at https://arxiv.org/abs/2206.08336 or visit our website at https://zheng-dai.github.io/DiminishingReturns/.

The contents of this repository consist of 4 notebooks:

-Optimize
-ComparisonWithBaseline
-EvaluationOfDiminishingReturns
-nTimesCoverage

Running Optimize will generate vaccine designs using diminishing returns and populate the outputs foler. The outputs folder should be prepopulated, so this is optional. The rest of the notebooks can be run in any order.

ComparisonWithBaseline will run greedy search over a set of random datasets and analyze the output. EvaluationOfDiminishingReturns evaluates and analyzes the output of Optimze using the diminishing returns framework. nTimesCoverage evaluates and analyzes the output of Optimze using n-times coverage.

data.pkl contains the credences necessary to run the optimization procedure and to evaluate the designs in our benchmarks.

benchmarks.pkl contains the benchmark sequences




