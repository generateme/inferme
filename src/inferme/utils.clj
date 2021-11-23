(ns inferme.utils
  (:require [fastmath.core :as m]
            [fastmath.vector :as v]
            [clojure.string :as str])
  (:import [org.apache.commons.math3.linear Array2DRowRealMatrix CholeskyDecomposition]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

(defn cholesky
  [vss]
  (->> vss
       m/seq->double-double-array
       Array2DRowRealMatrix.
       CholeskyDecomposition.
       .getL
       .getData
       m/double-double-array->seq
       (map vec)))


(defn partial-derivative
  ([model params] (partial-derivative model params 1.0e-6))
  ([model ^clojure.lang.PersistentVector params ^double interval]
   (let [^double lp (:lp (model params))]
     (mapv (fn [^long id]
             (let [new-params (assoc params id (+ ^double (.nth params id) interval))]
               (/ (- ^double (:lp (model new-params)) lp) interval)))
           (range (count params))))))

(defn multi?
  [d]
  (or (str/starts-with? (name d) "multi")
      (= d :dirichlet)))

(comment
  (def covar [[0.018583700 -0.002682908]
              [-0.002682908  0.0006236064]])

  (def chol (cholesky covar))

  (def nn [-0.3890267 -0.5978324])

  (mapv #(v/dot % nn) chol)
  )
