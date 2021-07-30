(ns laplaces-demon.2x
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]
            [cljplot.core :as pl]
            [cljplot.build :as b]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)


;; 20 Binomial Logit

(def exposed (repeat 10 100))
(def deaths (range 10 110 10))
(def dose (range 1 11))

(defmodel binomial-logit
  [beta (multi :normal 2 {:mu 0 :sd 1000})]
  (let [mu (map (fn [^double x]
                  (m/sigmoid (+ ^double (beta 0) (* ^double (beta 1) x)))) dose)
        binomials (map (fn [p n] (distr :binomial {:trials n :p p})) mu exposed)]
    (model-result (map (fn [b y] (observe1 b y)) binomials deaths)
                  (fn [] {:yhat (mapv (fn [b] (sample b)) binomials)}))))

(def res (infer :metropolis-hastings binomial-logit {:step-scale 0.04
                                                     :initial-point [[0.0 0.0]]
                                                     :samples 10000
                                                     :thin 100}))

(def res (infer :metropolis-within-gibbs binomial-logit {:step-scale 0.1
                                                         :initial-point [[0.0 0.0]]
                                                         :samples 10000
                                                         :thin 40}))


(:acceptance-ratio res)
(:steps res)

(stats/mean (map first (trace res :beta)))
(stats/mean (map second (trace res :beta)))

(plot/histogram (map first (trace res :beta)))
(plot/histogram (map second (trace res :beta)))

(stats/mean (map #(% 1) (trace res :yhat)))
(plot/frequencies (map #(% 1) (trace res :yhat)))

(plot/lag (map first (trace res :beta)))
(plot/lag (map second (trace res :beta)))

;; log posterior
(plot/histogram (map :LP (:accepted res)))

;; best
(best-result res)

;; 21 Binomial Probit

(defmodel binomial-probit
  [beta1 (:normal {:mu 0 :sd 1000})
   beta2 (:normal {:mu 0 :sd 1000})]
  (let [mu (map (fn [^double x]
                  (r/cdf r/default-normal (m/constrain (+ beta1 (* beta2 x)) -10.0 10.0))) dose)
        binomials (map (fn [p n] (distr :binomial {:trials n :p p})) mu exposed)]
    (model-result (map (fn [b y] (observe1 b y)) binomials deaths)
                  (fn [] {:yhat (mapv (fn [b] (sample b)) binomials)}))))

(def res (infer :metropolis-within-gibbs binomial-probit {:step-scale 0.03
                                                          :initial-point [0.0 0.0]
                                                          :samples 10000
                                                          :thin 50}))


(def res (infer :metropolis-hastings binomial-probit {:step-scale 0.03
                                                      :initial-point [0.0 0.0]
                                                      :samples 10000
                                                      :thin 50}))


(:acceptance-ratio res)
(:steps res)

(stats/mean (trace res :beta1))
(stats/mean (trace res :beta2))

(plot/histogram (trace res :beta1))
(plot/histogram (trace res :beta2))

(stats/mean (map #(% 1) (trace res :yhat)))
(plot/frequencies (map #(% 1) (trace res :yhat)))

(plot/lag (trace res :beta1))
(plot/lag (trace res :beta2))

;; log posterior
(plot/histogram (map :LP (:accepted res)))

;; best
(best-result res)

;; 22 Binomial Robit

(defmodel binomial-robit
  [beta1 (:normal {:mu 0 :sd 1000})
   beta2 (:normal {:mu 0 :sd 1000})
   nu (:uniform-real {:lower 5.0 :upper 10.0})]
  (let [t (distr :t {:degrees-of-freedom nu})
        mu (map (fn [^double x]
                  (r/cdf t (m/constrain (+ beta1 (* beta2 x)) -10.0 10.0))) dose)
        binomials (map (fn [p n] (distr :binomial {:trials n :p p})) mu exposed)]
    (model-result (map (fn [b y] (observe1 b y)) binomials deaths)
                  (fn [] {:yhat (mapv (fn [b] (sample b)) binomials)}))))

(def res (infer :metropolis-within-gibbs binomial-robit {:step-scale 0.2
                                                         :initial-point [0.0 0.0 5.0]
                                                         :samples 10000
                                                         :thin 50}))


(def res (infer :metropolis-hastings binomial-robit {:steps [0.03 0.03 0.5]
                                                     :initial-point [0.0 0.0 5.0]
                                                     :samples 10000
                                                     :thin 50}))


(:acceptance-ratio res)
(:steps res)

(stats/mean (trace res :beta1))
(stats/mean (trace res :beta2))
(stats/mean (trace res :nu))

(plot/histogram (trace res :beta1))
(plot/histogram (trace res :beta2))
(plot/histogram (trace res :nu))

(stats/mean (map #(% 6) (trace res :yhat)))
(plot/frequencies (map #(% 6) (trace res :yhat)))

(plot/lag (trace res :beta1))
(plot/lag (trace res :beta2))
(plot/lag (trace res :nu))

;; log posterior
(plot/histogram (map :LP (:accepted res)))

;; best
(best-result res)

;; 23

(def ys [1.12, 1.12, 0.99, 1.03, 0.92, 0.90, 0.81, 0.83, 0.65, 0.67, 0.60,
         0.59, 0.51, 0.44, 0.43, 0.43, 0.33, 0.30, 0.25, 0.24, 0.13, -0.01,
         -0.13, -0.14, -0.30, -0.33, -0.46, -0.43, -0.65])
(def xs [-1.39, -1.39, -1.08, -1.08, -0.94, -0.80, -0.63, -0.63, -0.25, -0.25,
         -0.12, -0.12, 0.01, 0.11, 0.11, 0.11, 0.25, 0.25, 0.34, 0.34, 0.44,
         0.59, 0.70, 0.70, 0.85, 0.85, 0.99, 0.99, 1.19])

(let [d (map vector xs ys)] 
  (-> (pl/xy-chart {:width 500 :height 500}
                   (b/series [:grid]
                             [:scatter d {:size 30}])
                   (b/add-axes :bottom)
                   (b/add-axes :left))
      (pl/show)))

(defmodel change-point-regression
  [alpha (:normal {:mu 0 :sd 1000}) 
   beta1 (:normal {:mu 0 :sd 1000})
   beta2 (:normal {:mu 0 :sd 1000})
   sigma (:half-cauchy {:scale 25}) 
   theta (:uniform-real {:lower -1.3 :upper 1.1})]
  (let [mu (map (fn [^double x]
                  (let [x-theta (- x theta)
                        a (if (pos? x-theta)
                            (* beta2 x-theta)
                            0.0)]
                    (+ alpha (* beta1 x) a))) xs)]
    (model-result (map (fn [m y]
                         (observe1 (distr :normal {:mu m :sigma sigma}) y)) mu ys)
                  (fn [] {:yhat (map #(r/grand %1 sigma) mu)}))))

(def res (infer :metropolis-within-gibbs change-point-regression {:step-scale 2.0
                                                                  :initial-point [0.2 -0.45 0.0 0.2 0.0]
                                                                  :samples 10000
                                                                  :burn 5000
                                                                  :thin 100}))


(def res (infer :metropolis-hastings change-point-regression {:steps [0.15 0.15 0.15 0.5 0.1]
                                                              :initial-point [0.2 -0.45 0.0 0.2 0.0]
                                                              :samples 10000
                                                              :burn 5000
                                                              :thin 100}))


(:acceptance-ratio res)
(:steps res)
(:out-of-prior res)


(stats/mean (trace res :alpha))
(stats/mean (trace res :beta1))
(stats/mean (trace res :beta2))
(stats/mean (trace res :sigma))
(stats/mean (trace res :theta))

(plot/histogram (trace res :alpha))
(plot/histogram (trace res :theta))
(plot/histogram (trace res :sigma))
(plot/histogram (trace res :beta1))
(plot/histogram (trace res :beta2))

(plot/lag (trace res :alpha))
(plot/lag (trace res :theta))
(plot/lag (trace res :sigma))
(plot/lag (trace res :beta1))
(plot/lag (trace res :beta2))

;; log posterior
(plot/histogram (map :LP (:accepted res)))

(best-result res)

(let [d (map vector xs ys)
      {:keys [^double alpha ^double beta1 ^double beta2 ^double theta]} (best-result res)] 
  (-> (pl/xy-chart {:width 500 :height 500}
                   (b/series [:grid]
                             [:abline [beta1 alpha nil theta] {:size 3 :color :black}]
                             [:abline [(+ beta1 beta2) (- alpha (* beta2 theta)) theta]
                              {:size 3 :color :black}]
                             [:vline theta {:color :red :size 3}]
                             [:scatter d {:size 10}])
                   (b/add-axes :bottom)
                   (b/add-axes :left))
      (pl/show)))
