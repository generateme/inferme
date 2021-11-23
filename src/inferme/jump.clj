(ns inferme.jump
  "Various jump kernels"
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.vector :as v]
            [clojure.string :as str]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

(defn- regular-sampler
  [d]
  (fn ^double [] (r/sample d)))

(defn- bactrian-sampler
  [d ^double m]
  (let [s (- 1.0 (* m m))]
    (fn ^double []
      (let [v (+ m (* s ^double (r/sample d)))]
        (r/randval v (- v))))))

(defn- random-vector-array
  [^long len sampler]
  (let [arr (double-array len)]
    (loop [idx (long 0)]
      (if (< idx len)
        (do (aset arr idx ^double (sampler))
            (recur (unchecked-inc idx)))
        arr))))

(defn- random-vector
  [^long len sampler]
  (vec (random-vector-array len sampler)))

(defn- zero-sum-random-vector
  [^long len sampler]
  (let [^doubles v (random-vector-array len sampler)]
    (vec (v/sub v (double-array len (/ (v/sum v) len))))))

(defn dirichlet-drift
  [^long len sampler ^double step]
  (fn [param]
    (v/add param (v/mult (zero-sum-random-vector len sampler) step))))

(defn multi-drift
  [^long len sampler ^double step]
  (fn [param]
    (v/add param (v/mult (random-vector len sampler) step))))

(defn- build-kernel
  [d ^long len sampler ^double step]
  (cond
    (= d :dirichlet) (dirichlet-drift len sampler step)
    (str/starts-with? (name d) "multi-") (multi-drift len sampler step)
    :else (fn [^double param]
            (+ param (* step ^double (sampler))))))

(defn- find-independent-steps
  [^long len steps step-scale]
  (when steps (assert (and (sequential? steps)
                           (= len (count steps))
                           (every? number? steps)) "Steps should be a sequence with length equal number of priors!"))
  (if steps
    (let [scale (double (or step-scale 1.0))]
      (map (fn [^double step] (* step scale)) steps))
    (repeat len (or step-scale 0.05))))

(defn- independent-kernel [distr m]
  (let [sampler (if m
                  (bactrian-sampler distr m)
                  (regular-sampler distr))]
    (fn [model steps step-scale]
      (let [steps (find-independent-steps (count (:parameter-names model)) steps step-scale)]
        [steps (map (fn [distr-name ^long len ^double step]
                      (build-kernel distr-name len sampler step)) (:distribution-names model) (:distribution-dims model) steps)]))))

(defn regular-kernel [distr] (independent-kernel distr nil))
(defn bactrian-kernel
  ([distr m] (independent-kernel distr m))
  ([distr] (independent-kernel distr 0.8)))

