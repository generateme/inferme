# inferme

Yet another Bayesian inference library for Clojure

## What's inside

* Easy to use MCMC based framework
* Common PPL pattern: generative model -> inference...
* ...but pure CLJ, no continuations
* Explicit priors and log-likelihood in the model
* Easy to extend
* Available visualizations

Influenced by LaplacesDemonR and BayesianTools. Also Anglican and WebPPL.

## Model

To create a model you have to use `defmodel` or `make-model` macros. Here is the structure of the model:

```clojure
(defmodel NAME
  [PRIORS]
  (MODEL-BODY
    (MODEL-RESULT-FN LOG-LIKELIHOOD-SEQUENCE TRACE-VALUE-OR-MAP)
 ```
 
 For example:
 
 ```clojure
(defmodel observer-model
  [p (distr :uniform-real)]
  (let [coin-spinner (distr :binomial {:trials 20 :p p})
        bernoulli (distr :bernoulli {:p p})
        binomial (distr :binomial {:trials 10 :p p})]
    (model-result [(observe1 coin-spinner 15)]
                  {:next-outcome (r/sample bernoulli)
                   :next-ten-outcomes (r/sample binomial)})))
 ```
 
 
 Where:
 
* NAME is a name of your model, model is regular function
* PRIORS has form of `let` with symbols and distribution creators. Values of priors (created randomly or using MCMC steps) are passed to the model function as parameters. PRIORS can be empty.
* MODEL-BODY any Clojure code necessary to calculate model
* MODEL-RESULT-FN is one of `model-result` or `trace-result` macros. Macro creates internal structure used by inference algorithms.
    * `model-result` accepts two parameters: list of log likelihood values and parameters you want to trace
    * `trace-result` can be used when you want to do only trace parameters from the model
* LOG-LIKELIHOOD-SEQUENCE - list of log of likelihoods, log probabilities against your data, conditions etc. It should be just sequence of numbers where `##-Inf` means probability=0 and `0` means probability=1. Log of probabilities can be calculated with `condition`, `observe1` or `observe` helpers.
* TRACE-VALUE-OR-MAP is a any value you want to trace. It can be map or any value. One note: you don't have to trace priors, they will be added for you by inference algorithm.

## Inference

Currently 3 inference algorithms are implemented:

* `:forward-sampling` - every result with positive probability is traced, input parameters are randomly sampled from priors (if available). This method is good to generate samples from generative model.
* `:rejection-sampling` - every result with probability proportional to returned likelihood is traced, input parameters are randomly sampled from priors (if available)
* `:metropolis-hastings` - MCMC
* `:metropolis-within-gibbs` - MCMC, gibbs steps
* `:elliptical-slice-sampling` - "This algorithm is applicable only to models in which the prior mean of all parameters is zero.". See more [HERE](https://web.archive.org/web/20150619030502/http://www.bayesian-inference.com/mcmcess) and [HERE](https://arxiv.org/pdf/1001.0175.pdf)

To gather samples just call `(infer METHOD MODEL OPTIONAL-PARAMETERS)`. For example `(infer :metropolis-hastings observer-model)`.

Optional parameters is a map with following keys:

* `:samples` - maximum number of samples to gather
* `:max-iters` - maximum number of iterations
* `:max-time` - maximum time

Inference algorithm stops when on of the limits is passed.

Also for specific algorithm there are additional tuning parameters:

* `:log-bound` - log of bound value for rejection-sampling

For MCMC:

* `:burn` - number of dropped samples
* `:thin` - record every `thin` sample
* `:initial-point` - starting point for algorithm
* `:steps` - explicit MCMC step standard deviations
* `:step-scale` - scale for inferred steps
* `kernel` - jump kernel (default Gaussian)

## Returned values

Inference algorithm returns map where under the key `:accepted` you can find sequence of maps of returned values. To access specific trace you can use helper macro `(trace RESULT KEYWORD)` like `(trace observer-result :next-outcome)`

## Model result as distribution

[TODO]

## Distributions

Distributions are backed by `fastmath` library.
[TODO - list of parameters]

## Jump kernels

* regular, bactrian

[TODO]

## Visualizations

[TODO]

## Examples

Check notebooks folder for rework of ProMods book and some Agnlican examples
