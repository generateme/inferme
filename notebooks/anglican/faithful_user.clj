(ns anglican.faithful-user
  (:require [fastmath.core :as m]
            [inferme.core :refer :all]
            [inferme.plot :as plot]
            [fastmath.stats :as stats]
            [fastmath.random :as r]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;; https://probprog.github.io/anglican/examples/viewer/?worksheet=practical/faithful_user

;; 1. Genarative model

(plot/frequencies (sample 10000 (distr :poisson {:p 5})))

(defmodel user-behaviour-generative []
  (let [fake-rate (inc (double (sample (distr :poisson {:p 5}))))
        user-rate (inc (double (sample (distr :poisson {:p 30}))))]
    (loop [i 0
           faking? (flipb 0.05)
           events []]
      (if (< i 5)
        (recur (inc i) 
               (or faking? (flipb 0.05))
               (conj events (if-not faking?
                              (sample (distr :poisson {:p user-rate}))
                              (sample (distr :poisson {:p fake-rate})))))
        (trace-result {:events events
                       ;; we assume that there is a little observation noise
                       :fake? (flip (if faking? 0.99 0.01))})))))

(def user-data (-> (infer :forward-sampling user-behaviour-generative {:samples 100})
                   :accepted))

(take 10 user-data)
;; => ({:events [40 38 30 36 32], :fake? 0, :LL 0.0, :LP 0.0}
;;     {:events [23 25 28 22 30], :fake? 0, :LL 0.0, :LP 0.0}
;;     {:events [28 21 20 30 23], :fake? 0, :LL 0.0, :LP 0.0}
;;     {:events [29 29 36 37 36], :fake? 0, :LL 0.0, :LP 0.0}
;;     {:events [34 29 38 44 35], :fake? 0, :LL 0.0, :LP 0.0}
;;     {:events [39 25 2 6 5], :fake? 1, :LL 0.0, :LP 0.0}
;;     {:events [30 31 48 39 45], :fake? 0, :LL 0.0, :LP 0.0}
;;     {:events [24 13 19 18 17], :fake? 0, :LL 0.0, :LP 0.0}
;;     {:events [32 36 38 34 40], :fake? 0, :LL 0.0, :LP 0.0}
;;     {:events [13 23 23 18 21], :fake? 0, :LL 0.0, :LP 0.0})

(plot/frequencies (map #(if (m/one? (:fake? %)) :fake :honest) user-data))

(plot/histogram (mapcat :events user-data))

;; 2. Parameter inference

;; again deep hierarchy, bad performance...

(defn user-behaviour
  [data]
  (make-model
   [user-rates (:uniform-real {:lower 20.0 :upper 40.0})
    fake-rates (:uniform-real {:lower 1.0 :upper 10.0})
    faking-prob (:uniform-real {:lower 0.0 :upper 0.2})
    user-rate (multi :poisson (count data) {:p user-rates})
    fake-rate (multi :poisson (count data) {:p (max 1.0 (double fake-rates))})
    faking (:bernoulli {:p (max 0.0 (double faking-prob))})]
   (let [data-with-distrs (map (fn [e ^double u ^double f]
                                 (assoc e
                                        :user (distr :poisson {:p (inc u)})
                                        :fake (distr :poisson {:p (inc f)}))) data user-rate fake-rate)
         ;; faking (distr :bernoulli {:p faking-prob})
         res (map (fn [{:keys [fake? events user fake]}]
                    (let [
                          ;; sfaking (sample faking)
                          [^double buff ^long faking?] (reduce (fn [[^double buff ^long f] ^double v]
                                                                 (let [nf (or+ f (long (sample prior-faking)))]
                                                                   [(+ buff
                                                                       ;; 
                                                                       (if (m/zero? f)
                                                                         (+ (observe1 prior-faking nf)
                                                                            (observe1 user v))
                                                                         (observe1 fake v)))
                                                                    nf])) [0.0 (long faking)] events)]
                      [(+ buff (observe1 (distr :bernoulli {:p (if (m/one? faking?) 0.99 0.01)}) fake?))
                       faking?
                       (observe1 (distr :bernoulli {:p (if (m/one? faking?) 0.99 0.01)}) fake?)])) data-with-distrs)]
     (model-result (map first res)
                   (fn [] {:obs (map last res)
                          :inferred-faking (map second res)
                          :guessed-faking (map #(sample (distr :bernoulli {:p (if (m/one? (long (second %))) 0.99 0.01)})) res)})))))

(def user-behaviour-model (user-behaviour user-data))

(def inferred (infer :metropolis-hastings user-behaviour-model {:samples 10000
                                                                :thin 5
                                                                :initial-point-search-size 2000
                                                                ;; :initial-point
                                                                ;; (random-priors (user-behaviour user-data))
                                                                ;; [30.0 9.0 0.1 (repeatedly 100 #(r/irand 20 40))
                                                                ;; (repeatedly 100 #(r/irand 1 10)) 1.0]
                                                                :steps [0.7 0.7 0.05 0.4 0.4 0.6]}))

#_(observe1 (distr :bernoulli {:p (max 0.0 (double 0.02))}) 1)

(:acceptance-ratio inferred)
(count (distinct (map :faking (:accepted inferred))))
(:out-of-prior inferred)

(plot/histogram (trace inferred :user-rates))
(plot/histogram (trace inferred :fake-rates))
(plot/histogram (trace inferred :faking-prob))
(plot/frequencies (map long (trace inferred :faking)))

(def best (best-result inferred)) ;; => -3720.6762883495144

(call user-behaviour-model best)

(random-priors (user-behaviour user-data))

(plot/frequencies (map :fake? user-data))


(plot/frequencies (mapcat :obs (:accepted inferred)))

(defn predict-user-behaviour
  [inferred events cnt]
  (let [m (user-behaviour (map (fn [e] {:events e :fake? 0}) events))]
    (map :inferred-faking (repeatedly cnt #(call m (rand-nth (:accepted inferred)))))))

(frequencies (map first (predict-user-behaviour inferred [[26 31 6 5 5]] 10000)))

(frequencies (map first (predict-user-behaviour inferred [[27 23 22 30 24]] 10000)))

(frequencies (map first (predict-user-behaviour inferred [[1 1 1 1 1]] 10000)))
