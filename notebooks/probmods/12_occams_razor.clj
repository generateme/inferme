(ns probmods.12-occams-razor
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;;;;;;;;;;;;;;;;;
;; Occam's Razor
;;;;;;;;;;;;;;;;;;

;; The Size Principle

(def small-distr (distr :categorical-distribution {:data [:a :b :c]}))
(def big-distr (distr :categorical-distribution {:data [:a :b :c :d :e :f]}))

(def data [:a])

(defmodel model
  [hypothesis (distr :bernoulli)]
  (model-result [(observe (if (== 1.0 hypothesis) big-distr small-distr) data)]))

(def post (infer :rejection-sampling model {:samples 1000}))

(plot/frequencies (trace post :hypothesis))

(def full-data [:a :b :a :b :b :a :b])
(def data-sizes [0 1 2 3 4 5 6 7])

(defn hypothesis-posterior
  [data]
  (let [model (make-model
               [hypothesis (distr :bernoulli)]
               (model-result [(observe (if (== 1.0 hypothesis) big-distr small-distr) data)]))]
    (stats/mean (trace (infer :rejection-sampling model) :hypothesis))))

(def prob-big (pmap #(hypothesis-posterior (take % full-data)) data-sizes))

(plot/line (map vector data-sizes prob-big))

;; Example: The Rectangle Game

;; https://nextjournal.com/generateme/the-rectangle-game---mcmc-example-in-clojure/

;; Generalizing the Size Principle: Bayes Occam’s Razor

(def A (distr :categorical-distribution {:data [:a :b :c :d] :probabilities [0.375, 0.375, 0.125, 0.125]}))
(def B (distr :categorical-distribution {:data [:a :b :c :d]}))

(def observed-data [:a :b :a :b :c :d :b :b])

(defmodel model
  [hypothesis (distr :bernoulli)]
  (model-result [(observe (if (== 1.0 hypothesis) A B) observed-data)]))

(def posterior (infer :rejection-sampling model))

(plot/frequencies (trace posterior :hypothesis))

;; Example: Fair or unfair coin?

(def observed-data [:h :h :t :h :t :h :h :h :t :h])
(def observed-data (repeat 10 :h))
(def observed-data (repeat 25 :h))
(def observed-data (repeatedly 54 #(if (flipb 0.85) :h :t)))
(def fair-prior 0.999)
(def pseudo-counts (distr :beta {:alpha 1 :beta 1}))

(let [observed-data-ints (mapv #(if (= :h %) 1 0) observed-data)]
  (defmodel model
    [fair (:bernoulli {:p fair-prior})]
    (let [coin-weight (if (pos? fair) 0.5 (r/sample pseudo-counts))]
      (model-result [(observe (distr :bernoulli {:p coin-weight}) observed-data-ints)]
                    {:weight coin-weight
                     :prior (if (flipb fair-prior) 0.5 (r/sample pseudo-counts))}))))

(def results (infer :metropolis-hastings model {:samples 100000}))

(plot/histogram (trace results :prior))
(plot/histogram (trace results :weight))
(plot/frequencies (trace results :fair))

;; The Effect of Unused Parameters

;;;;;;;;;;;;;;;;;;;;;;; Anglican
