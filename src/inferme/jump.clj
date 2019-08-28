(ns inferme.jump
  "Various jump kernels"
  (:require [fastmath.core :as m]
            [fastmath.random :as r]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

(defn- make-regular-sampler [d] (fn [] (r/sample d)))
(defn- make-bactrian-sampler
  [d ^double m]
  (let [s (- 1.0 (* m m))]
    (fn []
      (let [v (+ m (* s ^double (r/sample d)))]
        (r/randval v (- v))))))

(defn- find-independent-steps
  [steps ^double step-scale model]
  (let [cnt (int (reduce m/fast+ 0.0 (:parameter-count model)))]
    (vec (if (and steps (== (count steps) cnt))
           (map #(* step-scale ^double %) steps)
           (repeat cnt (* step-scale (/ 2.381204 (m/sqrt cnt))))))))

(defn- independent-kernel [d m]
  (let [sampler (if m
                  (make-bactrian-sampler d m)
                  (make-regular-sampler d))]
    (fn [steps step-scale model]
      (let [steps (find-independent-steps steps step-scale model)]
        [steps (fn [params]
                 (mapv (fn [^double p ^double s]
                         (+ p (* s ^double (sampler)))) params steps))]))))

(defn regular-kernel [d] (independent-kernel d nil))
(defn bactrian-kernel
  ([d m] (independent-kernel d m))
  ([d] (independent-kernel d 0.8)))
