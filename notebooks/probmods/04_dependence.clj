(ns probmods.04-dependence
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Causal and statistical dependence
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Causal Dependence

(let [c (flipb)
      b (flipb)
      a (if b (flipb 0.1) (flipb 0.4))]
  (or a c))

(defmodel model []
  (let [smokes (flipb 0.2)
        lung-disease (or (and smokes (flipb 0.1))
                         (flipb 0.001))
        cold (flipb 0.02)
        cough (or (and cold (flipb 0.5))
                  (and lung-disease (flipb 0.5))
                  (flipb 0.001))
        fever (or (and cold (flipb 0.3))
                  (flipb 0.01))
        chest-pain (or (and lung-disease (flipb 0.2))
                       (flipb 0.01))
        shortness-of-breath (or (and lung-disease (flipb 0.2))
                                (flipb 0.01))]
    (model-result [(condition cough)]
                  {:cold cold :lung-disease lung-disease})))

(let [marg (infer :rejection-sampling model)]
  (plot/frequencies (trace marg :cold))
  (plot/frequencies (trace marg :lung-disease)))


(let [c (flipb)
      b (flipb)
      a (if c (if b (flipb 0.85) false) false)]
  a)

;; Detecting Dependence Through Intervention

(defn b-do-a
  [a-val]
  (trace (infer :forward-sampling (make-model
                                   [] (let [c (flipb)
                                            a a-val
                                            b (if a (flipb 0.1) (flipb 0.4))]
                                        (trace-result {:b b})))) :b))
(do
  (plot/frequencies (b-do-a true))
  (plot/frequencies (b-do-a false)))

(defmodel model []
  (let [smokes (flipb 0.2)
        lung-disease (or (and smokes (flipb 0.1))
                         (flipb 0.001))
        cold (flipb 0.02)
        cough (or (and cold (flipb 0.5))
                  (and lung-disease (flipb 0.5))
                  (flipb 0.001))
        fever (or (and cold (flipb 0.3))
                  (flipb 0.01))
        chest-pain (or (and lung-disease (flipb 0.2))
                       (flipb 0.01))
        shortness-of-breath (or (and lung-disease (flipb 0.2))
                                (flipb 0.01))]
    (trace-result {:cold cold :cough cough})))


(let [marg (infer :forward-sampling model)]
  (plot/frequencies (trace marg :cold))
  (plot/frequencies (trace marg :cough)))

(defmodel model []
  (let [smokes (flipb 0.2)
        lung-disease (or (and smokes (flipb 0.1))
                         (flipb 0.001))
        cold true
        cough (or (and cold (flipb 0.5))
                  (and lung-disease (flipb 0.5))
                  (flipb 0.001))
        fever (or (and cold (flipb 0.3))
                  (flipb 0.01))
        chest-pain (or (and lung-disease (flipb 0.2))
                       (flipb 0.01))
        shortness-of-breath (or (and lung-disease (flipb 0.2))
                                (flipb 0.01))]
    (trace-result {:cold cold :cough cough})))


(let [marg (infer :forward-sampling model)]
  (plot/frequencies (trace marg :cold))
  (plot/frequencies (trace marg :cough)))

;; Statistical Dependence

(defn b-cond-a
  [a-val]
  (trace (infer :forward-sampling (make-model
                                   [] (let [c (flipb)
                                            a (flipb)
                                            b (if a (flipb 0.1) (flipb 0.4))]
                                        (model-result [(condition (= a a-val))] 
                                                      {:b b})))) :b))
(do
  (plot/frequencies (b-cond-a true))
  (plot/frequencies (b-cond-a false)))


(defn b-cond-a
  [a-val]
  (trace (infer :forward-sampling (make-model
                                   [] (let [c (flipb)
                                            a (if c (flipb 0.5) (flipb 0.9))
                                            b (if c (flipb 0.1) (flipb 0.4))]
                                        (model-result [(condition (= a a-val))] 
                                                      {:b b})))) :b))
(do
  (plot/frequencies (b-cond-a true))
  (plot/frequencies (b-cond-a false)))

