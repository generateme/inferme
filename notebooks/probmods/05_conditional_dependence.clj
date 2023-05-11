(ns probmods.05-conditional-dependence
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Conditional dependence
;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Screening off

(defn b-cond-a
  [a-val]
  (-> (infer :rejection-sampling (make-model
                                  [] (let [c (flipb)
                                           b (if c (flipb 0.5) (flipb 0.9))
                                           a (if c (flipb 0.1) (flipb 0.4))]
                                       (model-result [(condition c)
                                                      (condition (= a a-val))] 
                                                     {:b b}))))
      (trace :b)))
(do
  (plot/frequencies (b-cond-a true) {:title "true"})
  (plot/frequencies (b-cond-a false) {:title "false"}))

;; Explaining away

(defn b-cond-a
  [a-val]
  (-> (infer :rejection-sampling (make-model
                                  [] (let [a (flipb)
                                           b (flipb)
                                           c (if (or a b) (flipb 0.9) (flipb 0.2))]
                                       (model-result [(condition c)
                                                      (condition (= a a-val))] 
                                                     {:b b}))))
      (trace :b)))
(do
  (plot/frequencies (b-cond-a true) {:title "true"})
  (plot/frequencies (b-cond-a false) {:title "false"}))

(defmodel sum-posterior-model
  []
  (let [a (r/irand 10)
        b (r/irand 10)]
    (model-result [(condition (== 9 (+ a b)))]
                  {:a a :b b})))

(plot/scatter (traces (infer :rejection-sampling sum-posterior-model) :a :b))

(defmodel sum-posterior-model
  []
  (let [a (r/irand 10)
        b (r/irand 10)]
    (model-result [(condition (== a b))]
                  {:a a :b b})))

(plot/scatter (traces (infer :rejection-sampling sum-posterior-model) :a :b))

;; Example: Medical Diagnosis

(defmodel model []
  (let [smokes (flipb 0.2)
        lung-disease (or (and smokes (flipb 0.1))
                         (flipb 0.001))
        cold (flipb 0.02)
        cough (or (and cold (flipb 0.5)) (and lung-disease (flipb 0.5)) (flipb 0.001))
        fever (or (and cold (flipb 0.3)) (flipb 0.01))
        chest-pain (or (and lung-disease (flipb 0.2)) (flipb 0.01))
        shortness-of-breath (or (and lung-disease (flipb 0.2)) (flipb 0.01))]
    (model-result [(condition (and cough
                                   chest-pain
                                   shortness-of-breath))]
                  {:smokes smokes})))

(let [marg (infer :rejection-sampling model)]
  (plot/frequencies (trace marg :smokes)))

(defmodel model []
  (let [smokes (flipb 0.2)
        lung-disease (or (and smokes (flipb 0.1))
                         (flipb 0.001))
        cold (flipb 0.02)
        cough (or (and cold (flipb 0.5)) (and lung-disease (flipb 0.5)) (flipb 0.001))
        fever (or (and cold (flipb 0.3)) (flipb 0.01))
        chest-pain (or (and lung-disease (flipb 0.2)) (flipb 0.01))
        shortness-of-breath (or (and lung-disease (flipb 0.2)) (flipb 0.01))]
    (model-result [(condition (and lung-disease
                                   (and cough
                                        chest-pain
                                        shortness-of-breath)))]
                  {:smokes smokes})))

(let [marg (infer :rejection-sampling model)]
  (plot/frequencies (trace marg :smokes)))

(defmodel model []
  (let [smokes (flipb 0.2)
        lung-disease (or (and smokes (flipb 0.1))
                         (flipb 0.001))
        cold (flipb 0.02)
        cough (or (and cold (flipb 0.5)) (and lung-disease (flipb 0.5)) (flipb 0.001))
        fever (or (and cold (flipb 0.3)) (flipb 0.01))
        chest-pain (or (and lung-disease (flipb 0.2)) (flipb 0.01))
        shortness-of-breath (or (and lung-disease (flipb 0.2)) (flipb 0.01))]
    (model-result [(condition cough)]
                  {:cold cold :lung-disease lung-disease})))

(let [marg (infer :rejection-sampling model)]
  (plot/frequencies (trace marg :cold) {:title "cold"})
  (plot/frequencies (trace marg :lung-disease) {:title "lung disease"}))


(defmodel model []
  (let [smokes (flipb 0.2)
        lung-disease (or (and smokes (flipb 0.1))
                         (flipb 0.001))
        cold (flipb 0.02)
        cough (or (and cold (flipb 0.5)) (and lung-disease (flipb 0.5)) (flipb 0.001))
        fever (or (and cold (flipb 0.3)) (flipb 0.01))
        chest-pain (or (and lung-disease (flipb 0.2)) (flipb 0.01))
        shortness-of-breath (or (and lung-disease (flipb 0.2)) (flipb 0.01))]
    (model-result [(condition (and cough (not cold)))]
                  {:cold cold :lung-disease lung-disease})))

(let [marg (infer :rejection-sampling model)]
  (plot/frequencies (trace marg :cold) {:title "cold"})
  (plot/frequencies (trace marg :lung-disease) {:title "lung disease"}))

(defmodel model []
  (let [smokes (flipb 0.2)
        lung-disease (or (and smokes (flipb 0.1))
                         (flipb 0.001))
        cold (flipb 0.02)
        cough (or (and cold (flipb 0.5)) (and lung-disease (flipb 0.5)) (flipb 0.001))
        fever (or (and cold (flipb 0.3)) (flipb 0.01))
        chest-pain (or (and lung-disease (flipb 0.2)) (flipb 0.01))
        shortness-of-breath (or (and lung-disease (flipb 0.2)) (flipb 0.01))]
    (model-result [(condition (and cough cold))]
                  {:cold cold :lung-disease lung-disease})))

(let [marg (infer :rejection-sampling model)]
  (plot/frequencies (trace marg :cold) {:title "cold"})
  (plot/frequencies (trace marg :lung-disease) {:title "lung disease"}))

;; Example: Trait Attribution

(defmodel exam-posterior-model
  []
  (let [exam-fair (flipb 0.8)
        does-homework (flipb 0.8)
        pass (flipb (if exam-fair
                      (if does-homework 0.9 0.4)
                      (if does-homework 0.6 0.2)))]
    (model-result [(condition (not pass))]
                  {:does-homework does-homework
                   :exam-fair exam-fair})))

(let [marg (infer :forward-sampling exam-posterior-model)]
  (plot/frequencies (trace marg :does-homework) {:title "does homework"})
  (plot/frequencies (trace marg :exam-fair) {:title "exam fair"}))

(defmodel exam-posterior-model
  []
  (let [exam-fair (memoize (fn [exam] (flipb 0.8)))
        does-homework (memoize (fn [student] (flipb 0.8)))
        pass (fn [student exam] (flipb (if (exam-fair exam)
                                        (if (does-homework student) 0.9 0.4)
                                        (if (does-homework student) 0.6 0.2))))]
    (model-result [(condition (not (pass :bill :exam1)))]
                  {:does-homework (does-homework :bill)
                   :exam-fair (exam-fair :exam1)})))

(let [marg (infer :forward-sampling exam-posterior-model)]
  (plot/frequencies (trace marg :does-homework) {:title "does homework"})
  (plot/frequencies (trace marg :exam-fair) {:title "exam fair"}))

;; Example: Of Blickets and Blocking

(defmodel blicket-posterior-model
  []
  (let [blicket (memoize (fn [block] (flipb 0.4)))
        power (fn [block] (if (blicket block) 0.9 0.05))
        machine (fn machine-fn
                  [blocks]
                  (if-not (seq blocks)
                    (flipb 0.05)
                    (or (flipb (power (first blocks)))
                        (machine-fn (rest blocks)))))]
    (model-result [(condition (machine [:a :b]))]
                  (blicket :a))))

(let [marg (infer :forward-sampling blicket-posterior-model)]
  (plot/frequencies (trace marg :model-result)))

;; A Case Study in Modularity: Visual Perception of Surface Color

(defmodel reflectance-model
  [reflectance (:normal {:mu 1 :sd 1})
   illumination (:normal {:mu 3 :sd 1})]
  (let [luminance (* reflectance illumination)]
    (model-result [(observe1 (distr :normal {:mu luminance :sd 1}) 3)])))

(let [marg (infer :metropolis-hastings reflectance-model)
      dist (as-continuous-distribution marg :reflectance)] 
  (plot/pdf dist)
  (r/mean dist))
;; => 1.1387891133759487

(defmodel reflectance-model
  [reflectance (:normal {:mu 1 :sd 1})
   illumination (:normal {:mu 3 :sd 1})]
  (let [luminance (* reflectance illumination)]
    (model-result [(observe1 (distr :normal {:mu luminance :sd 1}) 3)
                   (observe1 (distr :normal {:mu illumination :sd 0.1}) 0.5)])))

(let [marg (infer :metropolis-hastings reflectance-model)
      dist (as-continuous-distribution marg :reflectance)] 
  (plot/pdf dist)
  (r/mean dist))
;; => 2.0370220302179862
