(ns probmods.08-process-models
  (:require [fastmath.core :as m]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Rational process models
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Prologue: The performance characteristics of different algorithms

;; The sampling hypothesis

(defmodel agent-belief
  [weight (:uniform-real)]
  (model-result [(observe1 (distr :binomial {:trials 5 :p weight}) 4)]
                (flipb weight)))

(defn agent-belief-distr
  []
  (-> (infer :rejection-sampling agent-belief)
      (as-categorical-distribution)))

(defn max-agent
  []
  (let [d (agent-belief-distr)]
    (> (score d true) (score d false))))

(plot/frequencies (repeatedly 100 max-agent))

(defn sample-agent
  []
  (sample (agent-belief-distr)))

(plot/frequencies (repeatedly 100 sample-agent))
