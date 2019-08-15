(ns probmods.03-conditioning
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Conditioning
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Hypothetical Reasoning with `infer`

(defn model []
  (let [a (flip)
        b (flip)
        c (flip)
        d (+ a b c)]
    d))

(let [dist (repeatedly 10000 model)]
  (plot/frequencies dist))

(let [dist (distr :integer-discrete-distribution {:data (repeatedly 10000 model)})]
  (plot/pdf dist))

(defmodel model
  [a (:bernoulli)
   b (:bernoulli)
   c (:bernoulli)]
  (trace-result {:d (+ a b c)}))

(let [dist (trace (infer :rejection-sampling model) :d)]
  (plot/frequencies dist))

(let [dist (trace (infer :metropolis-hastings model) :d)]
  (plot/frequencies dist))

(let [dist (trace (infer :forward-sampling model) :d)]
  (plot/frequencies dist))

(defn model []
  (let [a (flip)
        b (flip)
        c (flip)
        d (+ a b c)]
    (when (= d 3) d)))

(let [dist (repeatedly 10000 model)]
  (plot/frequencies dist))

(defmodel model
  [a (:bernoulli)
   b (:bernoulli)
   c (:bernoulli)]
  (let [d (+ a b c)]
    (model-result [(condition (== d 3))] d)))

(let [dist (trace (infer :rejection-sampling model) :model-result)]
  (plot/frequencies dist))

(defmodel model
  [a (:bernoulli)
   b (:bernoulli)
   c (:bernoulli)]
  (let [d (+ a b c)]
    (model-result [(condition (>= d 2))] d)))

(let [dist (trace (infer :rejection-sampling model) :a)]
  (plot/frequencies dist))

(defmodel model []
  (let [a (flip)
        b (flip)
        c (flip)
        d (+ a b c)]
    (model-result [(condition (>= d 2))] a)))

(let [dist (trace (infer :forward-sampling model) :model-result)]
  (plot/frequencies dist))

;; Rejection Sampling

(defn take-sample []
  (let [a (flip)
        b (flip)
        c (flip)
        d (+ a b c)]
    (if (>= d 2) a (take-sample))))

(plot/frequencies (repeatedly 100 take-sample))

(let [dist (trace (infer :rejection-sampling model {:samples 100}) :model-result)]
  (plot/frequencies dist))

;; Conditional Distributions

(def observed-data 1)
(defn likelihood ^long [^long h] (if (== h 1) (flip 0.9) (flip 0.1)))

(def posterior (infer :rejection-sampling (make-model
                                           [hypothesis (:bernoulli)]
                                           (let [data (likelihood hypothesis)]
                                             (model-result [(condition (= data observed-data))])))))

(plot/frequencies (trace posterior :hypothesis))

;; Conditions and observations

(defmodel model
  [true-x (:normal)]
  (let [obsx (r/grand true-x 0.1)]
    (model-result [(condition (== obsx true-x))])))

(let [inferred (infer :rejection-sampling model {:max-time 2 :max-iters ##Inf})]
  {:stop-reason (:stop-reason inferred)
   :accepted-size (count (:accepted inferred))})
;; => {:stop-reason :max-time, :accepted-size 0}

(defmodel model
  [true-x (:normal)]
  (model-result [(observe1 (distr :normal {:mu true-x :sd 0.1}) 0.2)]))

(plot/histogram (trace (infer :rejection-sampling model {:log-bound 2}) :true-x))

;; Factors

(let [dist (infer :rejection-sampling (make-model
                                       [a (:bernoulli)]
                                       (model-result [(condition (== a 1))])))]
  (plot/frequencies (trace dist :a)))

;; wrong
(let [dist (infer :rejection-sampling (make-model
                                       [a (:bernoulli)]
                                       (model-result [(condition (== a 1) 1 0)])))]
  (plot/frequencies (trace dist :a)))

;; good 
(let [dist (infer :rejection-sampling (make-model
                                       [a (:bernoulli)]
                                       (model-result [(condition (== a 1) 1 0)])) {:log-bound 2})]
  (plot/frequencies (trace dist :a)))

(let [dist (infer :metropolis-hastings (make-model
                                        [a (:bernoulli)]
                                        (model-result [(condition (== a 1) 1 0)])))]
  (plot/frequencies (trace dist :a)))

;; Example: Reasoning about Tug of War

(let [strength (memoize (fn [person] (m/abs (r/grand 1 1))))
      lazy (fn [person] (flipb m/THIRD))
      pulling (fn [person] (if (lazy person)
                            (* 0.5 ^double (strength person))
                            (strength person)))
      total-pulling (fn [team] (stats/sum (map pulling team)))
      winner (fn [team1 team2]
               (if (> ^double (total-pulling team1) ^double (total-pulling team2))
                 team1 team2))]
  [(winner [:alice :bob] [:sue :tom])
   (winner [:alice :bob] [:sue :tom])
   (winner [:alice :sue] [:bob :tom])
   (winner [:alice :sue] [:bob :tom])
   (winner [:alice :tom] [:bob :sue])
   (winner [:alice :tom] [:bob :sue])])
;; => [[:sue :tom] [:sue :tom] [:alice :sue] [:alice :sue] [:bob :sue] [:bob :sue]]


(defmodel model
  []
  (let [strength (memoize (fn [person]
                            (m/abs (r/grand 1 1))))
        lazy (fn [person] (flipb m/THIRD))
        pulling (fn [person] (if (lazy person)
                               (* 0.5 ^double (strength person))
                               (strength person)))
        total-pulling (fn [team] (reduce clojure.core/+ (map pulling team)))
        winner (fn [team1 team2]
                 (if (> ^double (total-pulling team1) ^double (total-pulling team2))
                   team1 team2))
        beat (fn [team1 team2] (= (winner team1 team2) team1))]
    (model-result [(condition (beat [:bob :mary] [:tom :sue]))
                   (condition (beat [:bob :sue] [:tom :jim]))]
                  {:bob-strength (strength :bob)})))

(let [dist (infer :metropolis-hastings model {:samples 25000})]
  (plot/histogram (trace dist :bob-strength))
  (r/mean (as-real-discrete-distribution dist :bob-strength)))
;; => 1.8708055357028774

(defmodel model
  []
  (let [strength (memoize (fn [person]
                            (m/abs (r/grand 1 1))))
        lazy (fn [person] (flipb m/THIRD))
        pulling (fn [person] (if (lazy person)
                               (* 0.5 ^double (strength person))
                               (strength person)))
        total-pulling (fn [team] (reduce clojure.core/+ (map pulling team)))
        winner (fn [team1 team2]
                 (if (> ^double (total-pulling team1) ^double (total-pulling team2))
                   team1 team2))
        beat (fn [team1 team2] (= (winner team1 team2) team1))] 
    (model-result [(condition (>= ^double (strength :mary) ^double (strength :sue)))
                   (condition (beat [:bob] [:jim]))]
                  {:beat (beat [:bob :mary] [:jim :sue])})))

(let [dist (infer :metropolis-hastings model {:samples 25000})]
  (plot/frequencies (trace dist :beat)))

;; Example: Inverse intuitive physics

;; skipped

;; Example: Causal Inference in Medical Diagnosis

(let [cancer-dist (infer :rejection-sampling (make-model
                                              [breast-cancer (:bernoulli {:p 0.01})]
                                              (let [positive-mammogram (if (pos? breast-cancer)
                                                                         (flipb 0.8)
                                                                         (flipb 0.096))]
                                                (model-result [(condition positive-mammogram)]))))]
  (plot/frequencies (trace cancer-dist :breast-cancer)))

(let [cancer-dist (infer :rejection-sampling (make-model
                                              [breast-cancer (:bernoulli {:p 0.01})
                                               benign-cyst (:bernoulli {:p 0.2})]
                                              (let [positive-mammogram (or+ (and+ breast-cancer (flip 0.8))
                                                                            (and+ benign-cyst (flip 0.5)))]
                                                (model-result [(condition (pos? positive-mammogram))]))))]
  (plot/frequencies (trace cancer-dist :breast-cancer)))

(defmodel model 
  [lung-cancer (:bernoulli {:p 0.01})
   tb (:bernoulli {:p 0.005})]
  (let [cold (flip 0.2)
        stomach-flu (flip 0.1)
        other (flip 0.1)
        cough (or+ (and+ cold (flip 0.5))
                   (and+ lung-cancer (flip 0.3))
                   (and+ tb (flip 0.7))
                   (and+ other (flip 0.01)))
        fever (or+ (and+ cold (flip 0.3))
                   (and+ lung-cancer (flip 0.5))
                   (and+ tb (flip 0.2))
                   (and+ other (flip 0.01)))
        chest-pain (or+ (and+ lung-cancer (flip 0.4))
                        (and+ tb (flip 0.5))
                        (and+ other (flip 0.01)))
        shortness-of-breath (or+ (and+ lung-cancer (flip 0.4))
                                 (and+ tb (flip 0.5))
                                 (and+ other (flip 0.01)))]
    (model-result [(condition (pos? (and+ cough fever chest-pain shortness-of-breath)))])))

(let [inferred (infer :metropolis-hastings model {:samples 1e5 :max-iters 1e6})]
  (plot/frequencies (:accepted inferred) {:sort? false}))

(defmodel model 
  [works-in-hospital (:bernoulli {:p 0.01})
   smokes (:bernoulli {:p 0.2})]
  (let [lung-cancer (or+ (flip 0.01) (and+ smokes (flip 0.02)))
        tb (or+ (flip 0.005) (and+ works-in-hospital (flip 0.01)))
        cold (or+ (flip 0.2) (and+ works-in-hospital (flip 0.25)))
        stomach-flu (flip 0.1)
        other (flip 0.1)
        cough (or+ (and+ cold (flip 0.5))
                   (and+ lung-cancer (flip 0.3))
                   (and+ tb (flip 0.7))
                   (and+ other (flip 0.01)))
        fever (or+ (and+ cold (flip 0.3))
                   (and+ lung-cancer (flip 0.5))
                   (and+ tb (flip 0.2))
                   (and+ other (flip 0.01)))
        chest-pain (or+ (and+ lung-cancer (flip 0.4))
                        (and+ tb (flip 0.5))
                        (and+ other (flip 0.01)))
        shortness-of-breath (or+ (and+ lung-cancer (flip 0.4))
                                 (and+ tb (flip 0.5))
                                 (and+ other (flip 0.01)))]
    (model-result [(condition (pos? (and+ cough fever chest-pain shortness-of-breath)))]
                  {:lung-cancer lung-cancer :tb tb})))

(let [inferred (infer :metropolis-hastings model {:samples 1e5 :max-iters 1e6})]
  (plot/frequencies (map #(select-keys % [:lung-cancer :tb]) (:accepted inferred)) {:sort? false}))

