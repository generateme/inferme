(ns probmods.10-lot-learning
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Learning with a language of thought
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; arithmetic expression failed

;; Example: Rational Rules

(defn make-obj [l] (zipmap [:trait1 :trait2 :trait3 :trait4 :fep] l))
(def feps (map make-obj [[0,0,0,1, 1], [0,1,0,1, 1], [0,1,0,0, 1], [0,0,1,0, 1], [1,0,0,0, 1]]))
(def non-feps (map make-obj [[0,0,1,1, 0], [1,0,0,1, 0], [1,1,1,0, 0], [1,1,1,1, 0]]))
(def others (map make-obj [[0,1,1,0], [0,1,1,1], [0,0,0,0], [1,1,0,1], [1,0,1,0], [1,1,0,0], [1,0,1,1]]))
(def data (concat feps non-feps))
(def all-objs (concat others feps non-feps))

(def human-feps [0.77, 0.78, 0.83, 0.64, 0.61])
(def human-non-feps [0.39, 0.41, 0.21, 0.15])
(def human-other [0.56, 0.41, 0.82, 0.40, 0.32, 0.53, 0.20])
(def human-data (concat human-other human-feps human-non-feps))

(def tau 0.3)
(def ^:const ^double noise-param (m/exp -1.5))

(defn sample-pred []
  (let [trait (rand-nth [:trait1 :trait2 :trait3 :trait4])
        value (flip)]
    (fn [x] (if (= (x trait) value) 1 0))))

(defn sample-conj []
  (if (flipb tau)
    (let [c (sample-conj)
          p (sample-pred)]
      (fn [x] (and+ ^long (c x) ^long (p x))))
    (sample-pred)))

(defn get-formula []
  (if (flipb tau)
    (let [c (sample-conj)
          p (get-formula)]
      (fn [x] (or+ ^long (c x) ^long (p x))))
    (sample-conj)))

(defmodel rule-model
  []
  (let [rule (get-formula)]
    (model-result (map (fn [datum]
                         (let [d (distr :bernoulli {:p (if (pos? ^long (rule datum))
                                                         (- 1.0 noise-param) noise-param)})]
                           (observe1 d (:fep datum)))) data)
                  (map rule all-objs))))

(def rule-posterior (infer :metropolis-hastings rule-model))
(def predictives (map stats/mean (apply map vector (trace rule-posterior :model-result))))

(plot/scatter (map vector predictives human-data))

