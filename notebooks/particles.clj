;; https://github.com/nchopin/particles/blob/master/docs/source/notebooks/SMC_samplers_tutorial.ipynb

(ns particles
  (:require [inferme.core :as im]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.plot :as plot]))

(def T 1000)

(def my-data (-> (r/distribution :normal {:mu 3.14})
                 (r/->seq T)))

(defn toy-model
  [data]
  (im/make-model
   [mu (:normal {:sd 10.0})
    sigma (:gamma {:scale 1.0 :shape 1.0})]
   (im/model-result [(im/observe (im/distr :normal {:mu mu :sd sigma}) data)])))

(def thetas (repeatedly 5 (partial im/random-priors (toy-model my-data) true)))

thetas
;; => ({:mu 4.054065086000591, :sigma 0.22571464728078117}
;;     {:mu 1.2929671048412783, :sigma 1.2628673250997287}
;;     {:mu -3.166100638586244, :sigma 0.020847483157448405}
;;     {:mu 6.117304193607003, :sigma 2.4512726002325786}
;;     {:mu 0.368333623969723, :sigma 0.04464271965407036})

(map (comp :LP (partial im/call (toy-model (take 3 my-data)))) thetas)
;; => (-24.32084504686292
;;     -11.502669047839152
;;     -140943.38208640966
;;     -13.420018533365576
;;     -6164.348150041295)

(def res (im/infer :metropolis-within-gibbs (toy-model my-data)))

(:acceptance-ratio res)
(:out-of-prior res)

(plot/histogram (im/trace res :mu))
(plot/histogram (im/trace res :sigma))
(plot/histogram (im/trace res :LP))


(im/stats res :mu)
(im/stats res :sigma)

(def my-data [1.88253582, 3.52487957, 3.5693534])
(def theta [-11.56023514, 0.66673981])

(apply - ((juxt :LP :LL) (im/call (toy-model (take 3 my-data)) theta)))

(def mu (im/distr :normal {:sd 10.0}))
(def sigma (im/distr :gamma {:scale 1.0 :shape 1.0}))

(im/score mu (theta 0));; => -3.8897188086591727
(im/score sigma (theta 1)) ;; => -0.66673981
