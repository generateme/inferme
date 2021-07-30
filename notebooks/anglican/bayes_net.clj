(ns anglican.bayes-net
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.protocols :as prot]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(defn sprinkler-bayes-net-fn
  [sprinkler wet-grass]
  (make-model
   [is-cloudy (:bernoulli)]
   (let [is-cloudy-bool (== is-cloudy 1.0)
         is-raining (if is-cloudy-bool (flipb 0.8) (flipb 0.2))
         sprinkler-dist (distr :bernoulli {:p (if is-cloudy-bool 0.1 0.5)})
         wet-grass-dist (distr :bernoulli {:p (cond 
                                                (and sprinkler is-raining) 0.99
                                                (and (not sprinkler) (not is-raining)) 0.0
                                                (or sprinkler is-raining) 0.9)})]
     (model-result [(observe1 sprinkler-dist (if sprinkler 1.0 0.0))
                    (observe1 wet-grass-dist (if wet-grass 1.0 0.0))]
                   {:is-raining is-raining}))))

(def result (infer :metropolis-hastings (sprinkler-bayes-net-fn true true) {:samples 50000
                                                                            :step-scale 0.5}))

(:acceptance-ratio result)
(:out-of-prior result)

(plot/frequencies (trace result :is-raining))
(plot/frequencies (map int (trace result :is-cloudy)))
(plot/histogram (trace result :is-cloudy))

;;

(defmethod r/distribution :dirac
  [_ {:keys [x]
      :or {x 0.0}}]
  (reify prot/DistributionProto
    (lpdf [_ v] (if (= v x) 0.0 ##-Inf))
    (sample [_] x)
    (continuous? [_] true)))

(defmodel sprinkler-bayes-net
  [is-cloudy (:bernoulli)]
  (let [is-cloudy-bool (== is-cloudy 1.0)
        is-raining (if is-cloudy-bool (flipb 0.8) (flipb 0.2))
        sprinkler (if is-cloudy-bool (flipb 0.1) (flipb 0.5))
        wet-grass (flipb (cond 
                           (and sprinkler is-raining) 0.99
                           (and (not sprinkler) (not is-raining)) 0.0
                           (or sprinkler is-raining) 0.9))]
    (model-result [(observe1 (distr :dirac {:x sprinkler}) true)
                   (observe1 (distr :dirac {:x wet-grass}) true)]
                  {:is-raining is-raining})))


(def result (infer :metropolis-hastings sprinkler-bayes-net {:step-scale 0.5}))

(:acceptance-ratio result)
(:out-of-prior result)

(plot/frequencies (trace result :is-raining))
