(ns anglican.pencil-factory
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

(defmodel run-pencil-factory
  [p (:uniform-real)]
  (model-result [(observe1 (distr :bernoulli {:p p}) 0)]))

(def sampler (infer :metropolis-hastings run-pencil-factory {:samples 10000
                                                             :step-scale 1.8
                                                             :thin 5}))

(:acceptance-ratio sampler)

(stats/mean (trace sampler :p));; => 0.3294395150250123
(stats/variance (trace sampler :p));; => 0.054428165186102016

(plot/histogram (trace sampler :p))
(plot/lag (trace sampler :p))

;;

(defn exp-beta
  "expectation of beta distribution"
  ^double [^double a ^double b]
  (/ a (+ a b)))

(defn var-beta
  "variance of beta distribution"
  ^double [^double a ^double b]
  (/ (* a b) (* (m/pow (+ a b) 2.0) (+ a b 1.0))))

(defn exp-beta-pos
  "posterior expectation of beta distribution having observed K successes from N trials"
  [^double a ^double b ^double N ^double K]
  (exp-beta (+ a K) (- (+ b N) K)))

(defn var-beta-pos
  "posterior variance of beta distribution having observed K successes from N trials"
  [^double a ^double b ^double N ^double K]
  (var-beta (+ a K) (- (+ b N) K)))

(defn run-pencil-factory-fn1
  [a b n k]
  (make-model
   [p (:beta {:alpha a :beta b})]
   (model-result [(observe1 (distr :binomial {:trials n :p p}) k)])))

(def sampler (infer :metropolis-hastings (run-pencil-factory-fn1 10 3 7 0) {:samples 10000
                                                                            :step-scale 2
                                                                            :thin 2}))

(:acceptance-ratio sampler)

(stats/mean (trace sampler :p)) ;; => 0.5015896500032446
(stats/variance (trace sampler :p)) ;; => 0.011820122246159248

(exp-beta-pos 10 3 7 0) ;; => 0.5
(var-beta-pos 10 3 7 0);; => 0.011904761904761908

(plot/histogram (trace sampler :p))
(plot/lag (trace sampler :p))

;;

(defn run-pencil-factory-fn2
  [n k]
  (make-model
   [z (:exponential {:mean 2})]
   (let [p (min z 1.0)]
     (model-result [(observe1 (distr :binomial {:trials n :p p}) k)]))))

(def sampler (infer :metropolis-hastings (run-pencil-factory-fn2 10 3) {:samples 10000
                                                                        :step-scale 0.2
                                                                        :thin 5}))

(:acceptance-ratio sampler)

(stats/mean (trace sampler :z)) ;; => 0.3267021029649186
(stats/variance (trace sampler :z));; => 0.01680762227591559

(plot/histogram (trace sampler :z))
(plot/lag (trace sampler :z))
