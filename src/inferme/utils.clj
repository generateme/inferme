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

(defn multi?
  [d]
  (or (str/starts-with? (name d) "multi-")
      (= d :dirichlet)))

(comment
  (def covar [[0.018583700 -0.002682908]
              [-0.002682908  0.0006236064]])

  (def chol (cholesky covar))

  (def nn [-0.3890267 -0.5978324])

  (mapv #(v/dot % nn) chol)
  )
