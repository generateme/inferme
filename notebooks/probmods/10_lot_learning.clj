(ns probmods.10-lot-learning
  (:require [fastmath.core :as m]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]
            [fastmath.random :as r]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* true)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Learning with a language of thought
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; arithmetic expression failed

(defn random-constant [] (str (r/irand 10)))
(defn ** [a b] (long (m/pow a b)))

(defn random-combination
  [f g]
  (let [op (rand-nth ["+" "-" "*" "/" "**"])]
    (str "(" op " " f " " g ")")))

(defn random-arithmetic-expression
  []
  (randval (random-combination (random-arithmetic-expression) (random-arithmetic-expression))
           (random-constant)))

(random-arithmetic-expression)
;; => "(/ (- (* (** (+ (/ 6 2) 7) 3) 0) 7) (/ 1 (+ 9 (* 0 (* 8 6)))))"
;; => "7"
;; => "(+ 6 6)"
;; => "9"

;; Inferring an Arithmetic Function

(def binary-ops [["+" +] ["-" -] ["*" *] ["/" /] ["**" **]])
(def idnty ["x" identity])

(defn random-constant-fn
  []
  (let [c (r/irand 10)]
    [(str c) (constantly c)]))

(defn random-combination
  [[fs f] [gs g]]
  (let [[ops op] (rand-nth binary-ops)]
    [(str "(" ops " " fs " " gs ")") (fn [x] (op (f x) (g x)))]))

(defn random-arithmetic-expression
  [depth]
  (if (zero? depth)
    (randval idnty (random-constant-fn))
    (randval (random-combination (random-arithmetic-expression (dec depth)) (random-arithmetic-expression (dec depth)))
             (randval idnty (random-constant-fn)))))

(defn safe-call
  [f v]
  (try
    (f v)
    (catch Exception _ ##NaN)))

(defn random-arithmetic-expression-model
  [target x]
  (make-model
   []
   (let [[e f] (random-arithmetic-expression 3)]
     (model-result [(condition (== (safe-call f x) target))]
                   e))))

(-> (infer :forward-sampling (random-arithmetic-expression-model 3 1) {:samples 100})
    (trace)
    (distinct)
    (->> (sort-by count)))
;; => ("3"
;;     "(* x 3)"
;;     "(+ 2 x)"
;;     "(+ x 2)"
;;     "(- 4 x)"
;;     "(* 3 1)"
;;     "(/ 9 3)"
;;     "(/ 3 1)"
;;     "(+ (+ x x) x)"
;;     "(* 3 (* x x))"
;;     "(+ x (* x 2))"
;;     "(* x (** 3 x))"
;;     "(+ 2 (** x 6))"
;;     "(** 9 (/ x 2))"
;;     "(- (- (* x 5) x) x)"
;;     "(- (+ x (* 7 1)) 5)"
;;     "(* x (+ (+ x x) x))"
;;     "(/ (+ (+ x x) x) x)"
;;     "(/ (- (* x 7) x) 2)"
;;     "(+ 1 (+ x (* 1 x)))"
;;     "(/ (- 7 x) (+ x 1))"
;;     "(+ x (* (+ x x) x))"
;;     "(/ (* x 3) (/ 1 x))"
;;     "(+ (** x (* x x)) 2)"
;;     "(** 3 (+ 0 (/ x x)))"
;;     "(** (+ x (- 3 x)) x)"
;;     "(+ (- (** x x) 1) 3)"
;;     "(* 6 (/ (* 5 x) (+ x 9)))"
;;     "(+ (+ (+ 2 x) (- 7 8)) 1)"
;;     "(** (+ (+ x x) (/ x x)) x)"
;;     "(- (** (+ 5 0) (* x x)) 2)"
;;     "(* x (** (/ 7 4) (* 2 x)))"
;;     "(+ (/ 3 x) (* (** 1 5) 0))"
;;     "(** (+ (* 7 x) (- x 5)) x)"
;;     "(+ (- 3 x) (** x (- 4 7)))"
;;     "(* (** 3 1) (** 6 (* 2 0)))"
;;     "(+ (** (+ x x) (** 1 2)) x)"
;;     "(+ (* (* 2 0) (/ 2 x)) (- (/ 4 x) x))"
;;     "(- (+ x (+ x 6)) (+ (+ x x) (* x 3)))"
;;     "(* (+ (- 6 x) x) (- (+ x x) (/ 6 4)))"
;;     "(* (** (- 3 4) (- x x)) (** 3 (* 1 x)))"
;;     "(+ (** (+ 0 x) (+ 4 x)) (+ (** x x) x))"
;;     "(- (- (** 5 x) (+ 0 x)) (** x (** 4 x)))"
;;     "(+ (* (* x 2) (+ x x)) (- (/ 2 2) (/ 2 x)))")

;; Example: Rational Rules

(m/use-primitive-operators)

(defn make-obj [l] (zipmap [:trait1 :trait2 :trait3 :trait4 :fep] l))
(def feps (map make-obj [[0,0,0,1, 1], [0,1,0,1, 1], [0,1,0,0, 1], [0,0,1,0, 1], [1,0,0,0, 1]]))
(def non-feps (map make-obj [[0,0,1,1, 0], [1,0,0,1, 0], [1,1,1,0, 0], [1,1,1,1, 0]]))
(def others (map make-obj [[0,1,1,0], [0,1,1,1], [0,0,0,0], [1,1,0,1], [1,0,1,0], [1,1,0,0], [1,0,1,1]]))
(def data (concat feps non-feps))
(def all-objs (concat others feps non-feps))

(def human-feps [0.77, 0.78, 0.83, 0.64, 0.61])
(def human-non-feps [0.39, 0.41, 0.21, 0.15])
(def human-other [0.56, 0.41, 0.82, 0.40, 0.32, 0.53, 0.20])
(def human-data (concat human-other human-feps human-non-feps))

(def tau 0.3)
(def ^:const ^double noise-param (m/exp -1.5))

(defn sample-pred []
  (let [trait (rand-nth [:trait1 :trait2 :trait3 :trait4])
        value (flip)]
    (fn [x] (if (= (x trait) value) 1 0))))

(defn sample-conj []
  (if (flipb tau)
    (let [c (sample-conj)
          p (sample-pred)]
      (fn [x] (and+ ^long (c x) ^long (p x))))
    (sample-pred)))

(defn get-formula []
  (if (flipb tau)
    (let [c (sample-conj)
          p (get-formula)]
      (fn [x] (or+ ^long (c x) ^long (p x))))
    (sample-conj)))

(defmodel rule-model
  []
  (let [rule (get-formula)]
    (model-result (map (fn [datum]
                         (let [d (distr :bernoulli {:p (if (pos? ^long (rule datum))
                                                         (- 1.0 noise-param) noise-param)})]
                           (observe1 d (:fep datum)))) data)
                  (map rule all-objs))))

(def rule-posterior (infer :metropolis-hastings rule-model))
(def predictives (map stats/mean (apply map vector (trace rule-posterior :model-result))))

(plot/scatter (map vector predictives human-data))

