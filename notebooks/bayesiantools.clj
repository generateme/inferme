;; https://cran.r-project.org/web/packages/BayesianTools/vignettes/BayesianTools.html
(ns bayesiantools
  (:require [inferme.core :as im]
            [inferme.plot :as plot]
            [fastmath.stats :as stats]
            [fastmath.core :as m]))

(def first-distr (im/distr :multi-normal {:means [0 0 0]}))

(im/defmodel first-model
  [m (im/multi :uniform-real 3 {:lower -10.0 :upper 10.0})]
  (im/model-result [(im/observe1 first-distr m)]))

(def res (im/infer :metropolis-hastings first-model {:step-scale 1.0
                                                     :thin 5}))

(:acceptance-ratio res)
(:out-of-prior res)

(first (:accepted res))

(plot/histogram (im/trace res :m 0))
(plot/histogram (im/trace res :m 1))
(plot/histogram (im/trace res :m 2))

(plot/lag (im/trace res :m 0))
(plot/lag (im/trace res :m 1))
(plot/lag (im/trace res :m 2))

(im/stats res :m 0)
;; => {:min -3.6105580733391083,
;;     :hdi-94% (-1.8996343984885335 1.8403603582276202),
;;     :mean -0.025526553402092002,
;;     :stddev 1.0040920773337505,
;;     :mode -0.3292465983982864,
;;     :size 10000,
;;     :median -0.039145436189661986,
;;     :max 3.6954980813482363,
;;     :percentiles
;;     {2.5 -1.963038131901796,
;;      97.5 1.9347321618898725,
;;      1 -2.315409939161508,
;;      95 1.6593596817486653,
;;      99 2.3388714974629914,
;;      :Q1 -0.7026926905027296,
;;      :Q3 0.6643359957425151,
;;      :median -0.039145436189661986,
;;      5 -1.6640965993135448},
;;     :lag nil}
