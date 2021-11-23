;; https://cran.r-project.org/web/packages/BayesianTools/vignettes/BayesianTools.html
(ns bayesiantools
  (:require [inferme.core :as im]
            [inferme.plot :as plot]))

(def first-distr (im/distr :multi-normal {:means [0 0 0]}))

(im/defmodel first-model
  [m (im/multi :uniform-real 3 {:lower -10.0 :upper 10.0})]
  (im/model-result [(im/observe1 first-distr m)]))

(def res (im/infer :metropolis-hastings first-model {:step-scale 1.0
                                                     :thin 5}))

(:acceptance-ratio res)
(:out-of-prior res)

(plot/histogram (im/trace res :a 0))
(plot/histogram (im/trace res :a 1))
(plot/histogram (im/trace res :a 2))

(plot/lag (im/trace res :a 0))
(plot/lag (im/trace res :a 1))
(plot/lag (im/trace res :a 2))

(im/stats res :a 0)
;; => {:min -3.5870747474350546,
;;     :mean -0.0033675373196662924,
;;     :stddev 0.9812328804140042,
;;     :mode 0.5292439577624617,
;;     :size 10000,
;;     :median -0.0036849060302567016,
;;     :max 3.6209208398761703,
;;     :percentiles
;;     {2.5 -1.9336907614319434,
;;      97.5 1.9300035117187384,
;;      1 -2.287981618624529,
;;      95 1.6264787850699494,
;;      99 2.309394292964303,
;;      :Q1 -0.6661517268978946,
;;      :Q3 0.6409158252972529,
;;      :median -0.0036849060302567016,
;;      5 -1.6137641340630937},
;;     :lag 5}

