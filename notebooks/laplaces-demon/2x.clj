(ns laplaces-demon.2x
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]
            [cljplot.core :as pl]
            [clojure2d.color :as c]
            [cljplot.build :as b]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;; 20 Binomial Logit

(def exposed (repeat 10 100))
(def deaths (range 10 110 10))
(def dose (range 1 11))

(defmodel binomial-logit
  [beta1 (:normal {:mu 0 :sd 1000})
   beta2 (:normal {:mu 0 :sd 1000})]
  (let [mu (map (fn [^double x]
                  (m/sigmoid (+ beta1 (* beta2 x)))) dose)]
    (model-result (map (fn [p y n]
                         (observe1 (distr :binomial {:trials n :p p}) y)) mu deaths exposed))))

(def res (infer :metropolis-hastings binomial-logit {:step-scale 0.03
                                                     :initial-point [0.0 0.0]
                                                     :samples 10000
                                                     :thin 100}))

(def res (infer :metropolis-within-gibbs binomial-logit {:step-scale 0.05
                                                         :initial-point [0.0 0.0]
                                                         :samples 10000
                                                         :thin 50}))


(:acceptance-ratio res)
(:steps res)

(stats/mean (trace res :beta1))
(stats/mean (trace res :beta2))

(plot/histogram (trace res :beta1))
(plot/histogram (trace res :beta2))

(plot/lag (trace res :beta1))
(plot/lag (trace res :beta2))

;; log posterior
(plot/histogram (map :LP (:accepted res)))

;; best
(best-result res)

;; 21 Binomial Probit

(defmodel binomial-probit
  [beta1 (:normal {:mu 0 :sd 1000})
   beta2 (:normal {:mu 0 :sd 1000})]
  (let [mu (map (fn [^double x]
                  (r/cdf r/default-normal (m/constrain (+ beta1 (* beta2 x)) -10.0 10.0))) dose)]
    (model-result (map (fn [p y n]
                         (observe1 (distr :binomial {:trials n :p p}) y)) mu deaths exposed))))

(def res (infer :metropolis-within-gibbs binomial-probit {:step-scale 0.03
                                                          :initial-point [0.0 0.0]
                                                          :samples 10000
                                                          :thin 50}))


(:acceptance-ratio res)
(:steps res)

(stats/mean (trace res :beta1))
(stats/mean (trace res :beta2))

(plot/histogram (trace res :beta1))
(plot/histogram (trace res :beta2))

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
                  (r/cdf t (m/constrain (+ beta1 (* beta2 x)) -10.0 10.0))) dose)]
    (model-result (map (fn [p y n]
                         (observe1 (distr :binomial {:trials n :p p}) y)) mu deaths exposed))))

(def res (infer :metropolis-within-gibbs binomial-robit {:step-scale 0.2
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
                         (observe1 (distr :normal {:mu m :sigma sigma}) y)) mu ys))))

(def res (infer :metropolis-within-gibbs change-point-regression {:step-scale 2.0
                                                                  :initial-point [0.2 -0.45 0.0 0.2 0.0]
                                                                  :samples 10000
                                                                  :burn 5000
                                                                  :thin 100}))


(:acceptance-ratio res)
(:steps res)

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

