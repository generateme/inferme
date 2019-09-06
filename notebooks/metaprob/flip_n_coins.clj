(ns metaprob.flip-n-coins
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [inferme.core :refer :all]))

(defn flip-n-coins
  ([n] (flip-n-coins n []))
  ([n observations]
   (make-model
    [tricky (:bernoulli {:p 0.1})
     prior-p (:uniform-real)] ;; I have to introduce this prior to be able to set the value
    (let [p (if-not (zero? tricky) prior-p 0.5)]
      (model-result [(observe (distr :bernoulli {:p p}) observations)]
                    {:p p
                     :gen (repeatedly n #(randval p 1 0))})))))

(defn coin-flips-demo-n-flips
  [n]
  (let [generator (call (flip-n-coins n))]
    (call (flip-n-coins n (:gen generator)))))

(coin-flips-demo-n-flips 5)
;; => {:p 0.2403560213962307, :gen (0 0 0 1 0), :prior-p 0.2403560213962307, :tricky 1.0, :LL -3.675984278671968, :LP -5.978569371666014}
;; => {:p 0.5, :gen (1 0 1 0 0), :prior-p 0.8275730067219886, :tricky 0.0, :LL -3.4657359027997265, :LP -3.571096418457553}

(def ensure-tricky-and-biased {:tricky 1
                               :prior-p 0.99})

(defn coin-flips-demo-biased
  [n]
  (let [model (flip-n-coins n [0])]
    (call model ensure-tricky-and-biased)))

(coin-flips-demo-biased 5)
;; => {:p 0.99, :gen (1 1 1 1 1), :prior-p 0.99, :tricky 1.0, :LL -4.605170185988091, :LP -6.907755278982137}
;; => {:p 0.99, :gen (1 0 1 1 1), :prior-p 0.99, :tricky 1.0, :LL -4.605170185988091, :LP -6.907755278982137}
