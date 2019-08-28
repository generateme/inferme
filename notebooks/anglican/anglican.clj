(ns anglican.anglican
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;; from presentation

(defn generate-observations
  [p n]
  (repeatedly n #(flip p)))

(def observations-a (generate-observations 0.07 150))
(def observations-b (generate-observations 0.04 250))

(defmodel ab-test
  [p-a (:uniform-real)
   p-b (:uniform-real)]
  (model-result [(observe (distr :bernoulli {:p p-a}) observations-a)
                 (observe (distr :bernoulli {:p p-b}) observations-b)]
                {:p-delta (- p-a p-b)}))

(def posterior (time (infer :metropolis-hastings ab-test {:samples 20000 :burn 1000 :thin 10 :step-scale 0.08})))
(def posterior (time (infer :metropolis-within-gibbs ab-test {:samples 10000 :burn 1000 :thin 2 :step-scale 0.04})))

(:acceptance-ratio posterior)

(plot/histogram (trace posterior :p-a))
(plot/histogram (trace posterior :p-b))
(plot/histogram (trace posterior :p-delta))
(plot/lag (trace posterior :p-a))

