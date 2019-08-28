(ns simple
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [fastmath.vector :as v]
            [inferme.core :refer :all] 
            [inferme.plot :as plot]
            [inferme.jump :as jump]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;

;; Looking for mean (should be 3)

(defmodel normal
  [mu (:normal {:sd 1000})]
  (model-result [(observe (distr :normal {:mu mu}) (repeatedly 5 #(r/grand 3 1)))]))

(def res (infer :metropolis-hastings normal {:steps [0.4]
                                             ;; :kernel (jump/bactrian-kernel (distr :laplace) 0.5)
                                             ;; :kernel (jump/bactrian-kernel (distr :normal) 0.95)
                                             :samples 10000
                                             :max-iters 1e7
                                             :thin 20
                                             :burn 5000
                                             ;; :initial-point [0]
                                             }))


(:acceptance-ratio res)
(count (trace res :mu))
(:steps res)

(stats/mean (trace res :mu))
;; => 3.017771819797229

(plot/lag (trace res :mu))

(plot/histogram (trace res :mu))

;; Let's find optimal step size for given model

(defn find-step
  [step]
  (:acceptance-ratio (infer :metropolis-hastings normal {:steps [step]
                                                         ;; :kernel (jump/bactrian-kernel (distr :laplace) 0.9)
                                                         ;; :kernel (jump/bactrian-kernel (distr :normal) 0.5)
                                                         :samples 1000
                                                         :initial-point [0]
                                                         })))

(plot/scatter (pmap #(vector % (find-step %)) (range 0.001 2 0.005)))
