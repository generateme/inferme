(ns laplaces-demon.1x
  (:require [fastmath.core :as m]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [fastmath.vector :as v]
            [fastmath.random :as r])
  (:import [org.apache.commons.math3.linear Array2DRowRealMatrix]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

(defn read-json [f] (with-open [reader (io/reader f)]
                      (doall (json/read-json reader true))))

(defn center-scale [xs] (v/mult (stats/standardize xs) 0.5))

;; 1 Adaptive Logistic Basis (ALB) Regression

(def demonsnacks (read-json "data/demonsnacks.json"))

(def N (count demonsnacks))
(def y (center-scale (v/log (map :Calories demonsnacks))))

(defn prepare-column [data column] (->> data (map column) v/log1p center-scale))
(def prepare-demonsnacks-column (partial prepare-column demonsnacks))

(def X (map prepare-demonsnacks-column [:Serving.Size :Saturated.Fat :Protein]))
(def J (count X))
(def K (inc (* 2 ^long J)))
(def K- (dec ^long K))

(def XA (.transpose (Array2DRowRealMatrix. (m/seq->double-double-array X))))

(defmodel alb-regression
  [zeta (:normal {:mu 0.0 :sd (m/sqrt 10.0)})
   tau (:half-cauchy {:scale 25})
   alpha (multi :normal K- {:mu 0.0 :sd (m/sqrt 10.0)})
   beta1 (multi :normal K- {:mu 0.0 :sd (m/sqrt 100.0)})
   beta2 (multi :normal K- {:mu 0.0 :sd (m/sqrt 100.0)})
   beta3 (multi :normal K- {:mu 0.0 :sd (m/sqrt 100.0)})
   delta (multi :normal K {:mu zeta :sd (max 1.0e-6 (double tau))})
   sigma (:half-cauchy {:scale 25})]
  (let [mplied (.add (.multiply ^Array2DRowRealMatrix XA
                                (Array2DRowRealMatrix. (m/seq->double-double-array [beta1 beta2 beta3])))
                     (Array2DRowRealMatrix. (m/seq->double-double-array (repeat N alpha))))
        exponent (Array2DRowRealMatrix. (m/seq->double-double-array (conj (vec (for [i (range K-)]
                                                                                 (v/exp (.getColumn mplied i)))) (double-array N 1.0))))
        deltaa (m/seq->double-array delta)
        mu (for [i (range N)
                 :let [r (.getColumn exponent i)]]
             (v/dot (v/div r (v/sum r)) deltaa))]
    (model-result (map #(observe1 (distr :normal {:mu %2 :sd sigma}) %1) y mu)
                  {:mu mu})))

(def res (infer :metropolis-within-gibbs alb-regression {:steps [1.0 1.0 0.3 1 1 1 0.1 0.5]
                                                         :initial-point [0 1 (repeat K- 0.0)
                                                                         (repeat K- 0.0)
                                                                         (repeat K- 0.0)
                                                                         (repeat K- 0.0)
                                                                         (repeat K 0.0) 1]
                                                         :samples 20000
                                                         :burn 50000
                                                         :max-time 180
                                                         :thin 25}))

(:acceptance-ratio res)
;; => 0.4737275

(plot/histogram (map last (trace res :alpha)))
(stats/mean (map first (trace res :alpha)))

(plot/histogram (trace res :sigma))
(stats/mean (trace res :sigma))
;; => 0.29807493045510935


(plot/lag (map first (trace res :beta1)))


;;;; ANCOVA

(def N 100)
(def J 5)
(def J- (dec ^long J))
(def K 3)
(def K- (dec ^long K))

(def X (let [js (range J)
             ks (range K)]
         [(repeatedly N #(rand-nth js))
          (repeatedly N #(rand-nth ks))
          (repeatedly N #(r/drand -2.0 2.0))]))

(def y (let [alpha (repeat N (r/drand -1.0 1.0))
             beta (vec (repeatedly J- #(r/drand -2.0 2.0)))
             beta (conj beta (- (v/sum beta)))
             gamma (vec (repeatedly K- #(r/drand -2.0 2.0)))
             gamma (conj gamma (- (v/sum gamma)))
             delta (r/drand -2.0 2.0)]
         (-> alpha
             (v/add (map beta (X 0)))
             (v/add (map gamma (X 1)))
             (v/add (v/mult (X 2) delta))
             (v/add (repeatedly N #(r/grand 0.0 0.1))))))

(defmodel ancova
  [alpha (:normal {:sd (m/sqrt 1000)})
   delta (:normal {:sd (m/sqrt 1000)})
   sigma (multi :half-cauchy 3 {:scale 25.0})
   beta (multi :normal J- {:sd (max 1.0e-100 (double (sigma 1)))})
   gamma (multi :normal K- {:sd (max 1.0e-100 (double (sigma 2)))})]
  (let [_println beta
        beta (conj beta (- (v/sum beta)))
        gamma (conj gamma (- (v/sum gamma)))
        mu (-> (repeat N alpha)
               (v/add (map beta (X 0)))
               (v/add (map gamma (X 1)))
               (v/add (v/mult (X 2) delta)))]
    (model-result (map #(observe1 (distr :normal {:mu %2 :sd (max 1.0e-100 (double (sigma 0)))}) %1) y mu)
                  {:mu mu
                   :s-beta (stats/stddev beta)
                   :s-gamma (stats/stddev gamma)
                   :s-epsilon (stats/stddev (v/sub y mu))})))


(def res (infer :metropolis-within-gibbs ancova {:steps [0.05 0.05 0.01 0.05 0.05]
                                                 :initial-point [0 0 [1 1 1]
                                                                 (vec (repeat J- 0.0))
                                                                 (vec (repeat K- 0.0))]
                                                 :samples 10000
                                                 :burn 1000
                                                 :max-time 180
                                                 :thin 5}))

(:acceptance-ratio res)

(let [t (trace res :s-epsilon)]
  (plot/histogram t)
  (plot/lag t)
  (stats/mean t))

;; 
