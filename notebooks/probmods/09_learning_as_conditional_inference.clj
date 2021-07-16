(ns probmods.09-learning-as-conditional-inference
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]
            [clojure.string :as str]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Learning as conditional inference
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Example: Learning About Coins

;; head = 1
(def observed-data (repeat 5 1))
(def fair-prior 0.999)

(defmodel fairness-model
  [fair (:bernoulli {:p fair-prior})]
  (let [coin (distr :bernoulli {:p (if (pos? fair) 0.5 0.95)})]
    (model-result [(observe coin observed-data)])))

(def fairness-posterior (infer :metropolis-hastings fairness-model))

(:acceptance-ratio fairness-posterior)

(plot/frequencies (trace fairness-posterior :fair))

(defn fairness-posterior
  [observed-data]
  (infer :metropolis-hastings (make-model
                               []
                               (let [fair (flip 0.999)
                                     coin (distr :bernoulli {:p (if (pos? fair) 0.5 0.95)})]
                                 (model-result [(observe coin observed-data)]
                                               fair)))))

(def true-weight 0.9)
(def full-data-set (repeatedly 100 #(flip true-weight)))
(def observed-data-sizes [1 3 6 10 20 30 50 70 100])
(def estimates #(pmap (fn [N]
                        (stats/mean (trace (fairness-posterior (take N full-data-set))))) observed-data-sizes))

(plot/line (map vector observed-data-sizes (estimates)))

;; Independent and Exchangeable Sequences

(defmodel gen-sequence-model
  [x1 (:bernoulli)
   x2 (:bernoulli)])

(def sequence-dist (infer :forward-sampling gen-sequence-model {:samples 1e5}))

(plot/frequencies (trace sequence-dist :x1))
(plot/frequencies (trace sequence-dist :x2))

(defn sequence-cond-dist
  [first-val]
  (trace (infer :forward-sampling (make-model [x1 (:bernoulli)
                                               x2 (:bernoulli)]
                                              (model-result [(condition (= (pos? x1) first-val))]))
                {:samples 1e5})
         :x2))

(plot/frequencies (sequence-cond-dist true))
(plot/frequencies (sequence-cond-dist false))

(def words [:chef :omelet :soup :eat :work :bake :stop])
(def probs [0.0032, 0.4863, 0.0789, 0.0675, 0.1974, 0.1387, 0.0277])
(def categorical (distr :integer-discrete-distribution {:data (range (count words))
                                                        :probabilities probs}))
(def thunk (comp words #(r/sample categorical)))

(repeatedly 10 thunk)
;; => (:omelet :bake :work :work :work :stop :omelet :omelet :omelet :bake)

(let [probs (r/randval
             [0.0032, 0.4863, 0.0789, 0.0675, 0.1974, 0.1387, 0.0277]
             [0.3699, 0.1296, 0.0278, 0.4131, 0.0239, 0.0159, 0.0194])
      categorical (distr :integer-discrete-distribution {:data (range (count words))
                                                         :probabilities probs})]
  (def thunk (comp words #(r/sample categorical))))

(repeatedly 10 thunk)
;; => (:bake :work :work :omelet :soup :omelet :omelet :soup :work :work)

(defn sequence-cond-dist
  [first-val]
  (trace (infer :forward-sampling (make-model
                                   [] (let [prob (r/randval 0.2 0.7)
                                            x1 (flipb prob)
                                            x2 (flipb prob)]
                                        (model-result [(condition (= x1 first-val))]
                                                      {:x2 x2})))
                {:samples 1e5})
         :x2))

(plot/frequencies (sequence-cond-dist true))
(plot/frequencies (sequence-cond-dist false))

;; Example: Polya’s urn

(defn urn-seq
  [urn ^long num-samples]
  (loop [i (int 0)
         u urn
         s []]
    (if (== i num-samples)
      (str/join s)
      (let [ball (rand-nth u)]
        (recur (inc i) (conj u ball) (conj s ball))))))

(plot/frequencies (repeatedly 100000 #(urn-seq [\b \w] 3)))

(defn urn-de-finetti
  [urn num-samples]
  (let [num-white (count (filter #(= \w %) urn))
        num-black (- (count urn) num-white)
        latent-prior (distr :beta {:alpha num-white :beta num-black})
        latent (r/sample latent-prior)]
    (str/join (repeatedly num-samples (r/randval latent \b \w)))))

(plot/frequencies (repeatedly 100000 #(urn-seq [\b \w] 3)))

;; Example: Subjective Randomness

(defn is-fair-dist
  [s]
  (trace (infer :metropolis-hastings (make-model
                                      [is-fair (:bernoulli)]
                                      (let [real-weight (if (pos? is-fair) 0.5 0.2)
                                            coin (distr :bernoulli {:p real-weight})]
                                        (model-result [(observe coin s)])))) :is-fair))

(plot/frequencies (is-fair-dist [0 0 1 0 1]))
(plot/frequencies (is-fair-dist [0 0 0 0 0]))

;; Learning a Continuous Parameter

(def observed-data (repeat 5 1))

(defmodel weight-model
  [coin-weight (:uniform-real)]
  (let [coin (distr :bernoulli {:p coin-weight})]
    (model-result [(observe coin observed-data)])))

(plot/histogram (trace (infer :metropolis-hastings weight-model) :coin-weight))

(defn weight-posterior
  [observed-data]
  (trace (infer :metropolis-hastings (make-model
                                      [coin-weight (:uniform-real)]
                                      (let [coin (distr :bernoulli {:p coin-weight})]
                                        (model-result [(observe coin observed-data)]))) {:samples 1000}) :coin-weight))

(def full-dataset (repeat 100 1))
(def observed-data-sizes [0 1 2 4 8 16 25 30 50 70 100])
(def estimates (pmap (fn [^long N]
                       (stats/mean (weight-posterior (take (inc N) full-dataset)))) observed-data-sizes))

(plot/line (map vector observed-data-sizes estimates))

(defn weight-posterior
  [observed-data]
  (trace (infer :metropolis-hastings (make-model
                                      [coin-weight (:beta {:a 10 :b 10})]
                                      (let [coin (distr :bernoulli {:p coin-weight})]
                                        (model-result [(observe coin observed-data)]))) {:samples 1000
                                                                                         :burn 1000}) :coin-weight))

(def estimates (pmap (fn [^long N]
                       (stats/mean (weight-posterior (take (inc N) full-dataset)))) observed-data-sizes))

(plot/line (map vector observed-data-sizes estimates))

;; A More Structured Hypothesis Space

(defn weight-posterior
  [observed-data]
  (trace (infer :metropolis-hastings (make-model
                                      [is-fair (:bernoulli {:p 0.999})]
                                      (let [real-weight (if (pos? is-fair) 0.5 (r/drand))
                                            coin (distr :bernoulli {:p real-weight})]
                                        (model-result [(observe coin observed-data)]
                                                      {:real-weight real-weight}))) {:samples 10000
                                                                                     :burn 1000}) :real-weight))

(def full-dataset (repeat 50 1))
(def observed-data-sizes [0,1,2,4,6,8,10,12,15,20,25,30,40,50])
(def estimates (pmap (fn [^long N]
                       (stats/mean (weight-posterior (take (inc N) full-dataset)))) observed-data-sizes))

(plot/line (map vector observed-data-sizes estimates))

;; Example: Estimating Causal Power

(def observed-data [{:C true, :E true}, {:C true, :E true}, {:C false, :E false}, {:C true, :E true}])

(defmodel casual-power-model
  [cp (:uniform-real)
   b (:uniform-real)]
  (model-result (map (fn [{:keys [C E]}]
                       (let [e (or (and C (flipb cp)) (flipb b))]
                         (condition (= E e)))) observed-data)))

(def casual-power-post (infer :metropolis-hastings casual-power-model {:samples 10000}))

(plot/histogram (trace casual-power-post :cp))
