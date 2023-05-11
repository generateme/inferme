(ns simple
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all] 
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;

;; Looking for mean (should be 3)

(defmodel normal
  [mu (:normal {:sd 1000})]
  (model-result [(observe (distr :normal {:mu mu}) (repeatedly 3 #(r/grand 3 1)))]))

(def res (infer :metropolis-hastings normal {:steps [1.0]
                                             ;; :kernel (jump/bactrian-kernel (distr :laplace) 0.8)
                                             ;; :kernel (jump/bactrian-kernel (distr :normal) 0.9)
                                             :samples 10000
                                             :max-iters 1e7
                                             :thin 20
                                             :burn 5000
                                             ;; :initial-point [3]
                                             }))


(:acceptance-ratio res)
(count (trace res :mu))
(:steps res)

(stats/mean (trace res :mu))
;; => 3.017771819797229

(plot/lag (trace res :mu))

(plot/histogram (trace res :mu))

(plot/histogram (m/rank (trace res :mu)))

;; Let's find optimal step size for given model

(defn find-step
  [step]
  (:acceptance-ratio (infer :metropolis-hastings normal {:steps [step]
                                                         ;; :kernel (jump/bactrian-kernel (distr :laplace) 0.9)
                                                         ;; :kernel (jump/bactrian-kernel (distr :normal) 0.5)
                                                         :samples 1000
                                                         :initial-point [0]
                                                         })))

(plot/scatter (pmap #(vector % (find-step %)) (range 0.0001 2 0.001)))


;;

(defmodel daslu-example
  [mu (:normal)]
  (model-result [(observe1 (distr :normal {:mu mu}) 10.0)]))

(def res (infer :metropolis-hastings daslu-example {:steps [1.0]
                                                    :samples 10000
                                                    :max-iters 1e7
                                                    :thin 5
                                                    :burn 5000}))

(:acceptance-ratio res)
;; => 0.6087853430337914
(count (trace res :mu))
;; => 10000

(stats/mean (trace res :mu))
;; => 4.997783637857927

(stats/variance (trace res :mu))
;; => 0.49403256673031765

(plot/lag (trace res :mu))

(plot/histogram (trace res :mu))
(plot/histogram (m/rank (trace res :mu)))


;; bad samples, jump to big:

(def res (infer :metropolis-hastings daslu-example {:steps [12.5]
                                                  :samples 2000
                                                  :max-iters 1e7}))
(plot/lag (trace res :mu))
(plot/histogram (trace res :mu))
(plot/histogram (m/rank (trace res :mu)))
