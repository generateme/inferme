(ns probmods.07-inference-algorithms
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [clojure2d.core :as c2d]
            [clojure2d.extra.utils :as utls]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Algorithms for inference
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Prologue: The performance characteristics of different algorithms

(defn inf-model
  [baserate]
  (make-model
   [] (let [a (flip baserate)
            b (flip baserate)
            c (flip baserate)]
        (model-result [(condition (>= (+ a b c) 2))]
                      {:a a}))))

(time (dotimes [i 10]
        (infer :rejection-sampling (inf-model 0.1) {:samples 100})))
(time (dotimes [i 10]
        (infer :rejection-sampling (inf-model 0.01) {:samples 100})))
(time (dotimes [i 10]
        (infer :forward-sampling (inf-model 0.1) {:samples 100})))
(time (dotimes [i 10]
        (infer :forward-sampling (inf-model 0.01) {:samples 100})))
(time (dotimes [i 10]
        (infer :metropolis-hastings (inf-model 0.1) {:samples 100})))
(time (dotimes [i 10]
        (infer :metropolis-hastings (inf-model 0.01) {:samples 100})))

(defn inf-model2
  [baserate numflips]
  (make-model
   [] (let [choices (repeatedly numflips #(r/flip baserate))]
        (model-result [(condition (>= (stats/sum choices) 2))]))))

(time (dotimes [i 10]
        (infer :rejection-sampling (inf-model2 0.1 3) {:samples 100})))
(time (dotimes [i 10]
        (infer :rejection-sampling (inf-model2 0.01 3) {:samples 100})))
(time (dotimes [i 10]
        (infer :forward-sampling (inf-model2 0.1 3) {:samples 100})))
(time (dotimes [i 10]
        (infer :forward-sampling (inf-model2 0.01 3) {:samples 100})))
(time (dotimes [i 10]
        (infer :metropolis-hastings (inf-model2 0.1 3) {:samples 100})))
(time (dotimes [i 10]
        (infer :metropolis-hastings (inf-model2 0.01 3) {:samples 100})))

(let [inferred (infer :rejection-sampling (inf-model 0.1) {:samples 100})]
  (plot/frequencies (trace inferred :a)))

(let [inferred (infer :metropolis-hastings (inf-model 0.1) {:thin 100})]
  (plot/frequencies (trace inferred :a)))

;; Markov chains as samplers

(def states [:a :b :c :d])
(def states-ids (range (count states)))
(def states-map (zipmap states (range (count states))))

(def transition-probs {:a [0.48 0.48 0.02 0.02]
                       :b [0.48 0.48 0.02 0.02]
                       :c [0.02 0.02 0.48 0.48]
                       :d [0.02 0.02 0.48 0.48]})

(def transition-distr
  (memoize (fn [state]
             (distr :integer-discrete-distribution
                    {:data states-ids :probabilities (transition-probs state)}))))

(defn transition [state]
  (states (r/sample (transition-distr state))))

(defn chain
  [state ^long n]
  (if (zero? n) state (recur (transition state) (dec n))))

(plot/frequencies (repeatedly 1000 #(chain :a 10)))
(plot/frequencies (repeatedly 1000 #(chain :c 10)))

(plot/frequencies (repeatedly 1000 #(chain :a 25)))
(plot/frequencies (repeatedly 1000 #(chain :c 25)))

(plot/frequencies (repeatedly 1000 #(chain :a 50)))
(plot/frequencies (repeatedly 1000 #(chain :c 50)))

(def transition-distr
  (memoize (fn [state]
             (distr :integer-discrete-distribution
                    {:data states-ids :probabilities [0.25 0.25 0.25 0.25]}))))

(plot/frequencies (repeatedly 1000 #(chain :a 10)))
(plot/frequencies (repeatedly 1000 #(chain :c 10)))

(plot/frequencies (repeatedly 1000 #(chain :a 25)))
(plot/frequencies (repeatedly 1000 #(chain :c 25)))

(plot/frequencies (repeatedly 1000 #(chain :a 50)))
(plot/frequencies (repeatedly 1000 #(chain :c 50)))

(def ^:const ^double p 0.7)

(defn transition
  [^long state]
  (if (== state 3)
    (r/sample (distr :integer-discrete-distribution
                     {:data [3 4] :probabilities [(- 1.0 (* 0.5 (- 1.0 p)))
                                                  (* 0.5 (- 1.0 p))]}))
    (r/sample (distr :integer-discrete-distribution
                     {:data [(dec state) state (inc state)]
                      :probabilities [0.5 (- 0.5 (* 0.5 (- 1.0 p))) (* 0.5 (- 1.0 p))]}))))

(defn chain
  [state ^long n]
  (if (zero? n) state (recur (transition state) (dec n))))

(plot/frequencies (repeatedly 5000 #(chain 3 250)))

(defn geometric
  ^long [p]
  (r/randval p 1 (inc (geometric p))))

(def post (infer :metropolis-hastings (make-model [] (let [mygeom (geometric p)]
                                                       (model-result [(condition (> mygeom 2))]
                                                                     mygeom))) {:samples 25000 :thin 10}))


(plot/frequencies (trace post :model-result))

;; Metropolis-Hastings

(def ^:const ^double p 0.7)

(defn target-dist
  ^double [^long x]
  (if (< x 3) 0.0 (* p (m/pow (- 1.0 p) (dec x)))))

(defn proposal-fn
  ^long [^long x]
  (r/randval (dec x) (inc x)))

(def proposal-dist (constantly 0.5))

(defn accept
  [^double x1 ^double x2]
  (flipb (min 1.0 (/ (* (target-dist x2) ^double (proposal-dist x2 x1))
                     (* (target-dist x1) ^double (proposal-dist x1 x2))))))

(defn transition
  [x]
  (let [proposed-x (proposal-fn x)]
    (if (accept x proposed-x) proposed-x x)))

(defn mcmc
  [state iterations]
  (take iterations (iterate transition state)))

(def chain (mcmc 3 10000))

(plot/frequencies chain)

;; Hamiltonian Monte Carlo

;; only metropolis hastings is available

(defn bin ^double [^double x] (/ (m/floor (* x 1000.0)) 1000.0))

(defmodel constrained-sum-model
  []
  (let [xs (repeatedly 10 r/drand)
        target-sum 5.0]
    (model-result [(observe1 (distr :normal {:mu target-sum :sd 0.005}) (stats/sum xs))]
                  (map bin xs))))

(def post (infer :metropolis-hastings constrained-sum-model {:samples 5000}))

(:acceptance-ratio post)
;; => 0.006363636363636364

(map #(hash-map :sum (stats/sum %) :nums %) (take 10 (shuffle (trace post :model-result))))
;; => ({:nums
;;      (0.784 0.968 0.541 0.042 0.122 0.996 0.829 0.158 0.442 0.116),
;;      :sum 4.998}
;;     {:nums (0.379 0.015 0.773 0.528 0.602 0.404 0.847 0.1 0.917 0.432),
;;      :sum 4.997000000000001}
;;     {:nums (0.415 0.198 0.711 0.417 0.835 0.146 0.803 0.043 0.969 0.45),
;;      :sum 4.987}
;;     {:nums (0.317 0.296 0.169 0.111 0.781 0.438 0.662 0.81 0.556 0.853),
;;      :sum 4.993}
;;     {:nums (0.457 0.01 0.908 0.636 0.642 0.545 0.39 0.354 0.739 0.31),
;;      :sum 4.991}
;;     {:nums
;;      (0.632 0.578 0.377 0.652 0.664 0.691 0.462 0.294 0.068 0.578),
;;      :sum 4.9959999999999996}
;;     {:nums (0.317 0.296 0.169 0.111 0.781 0.438 0.662 0.81 0.556 0.853),
;;      :sum 4.993}
;;     {:nums (0.211 0.903 0.596 0.44 0.733 0.426 0.545 0.514 0.07 0.559),
;;      :sum 4.997000000000001}
;;     {:nums
;;      (0.441 0.889 0.649 0.075 0.017 0.875 0.589 0.974 0.191 0.296),
;;      :sum 4.996}
;;     {:nums (0.434 0.588 0.086 0.767 0.56 0.357 0.248 0.513 0.882 0.556),
;;      :sum 4.991})

;; Particle Filters

;; no particle filters

(def number-of-observations 20)
(def true-loc [250 250])

(def observations
  [(repeatedly number-of-observations #(r/grand (first true-loc) 100))
   (repeatedly number-of-observations #(r/grand (second true-loc) 100))])

(defmodel radar-static-object
  [pos-x (:normal {:mu 200 :sd 100})
   pos-y (:normal {:mu 200 :sd 100})]
  (model-result [(observe (distr :normal {:mu pos-x :sd 5}) (first observations))
                 (observe (distr :normal {:mu pos-y :sd 5}) (second observations))]))

(def posterior (infer :metropolis-hastings radar-static-object {:step-scale 0.02 :burn 5000}))

(:acceptance-ratio posterior)

(let [ex (rand-nth (trace posterior :pos-x))
      ey (rand-nth (trace posterior :pos-y))]

  (utls/show-image (c2d/with-canvas [c (c2d/canvas 500 500)] 
                     (c2d/set-background c :white)
                     (c2d/set-color c :black)
                     (doseq [[x y] (apply map vector observations)]
                       (c2d/ellipse c x y 20 20 true))
                     (-> (c2d/set-color c :green)
                         (c2d/ellipse (first true-loc) (second true-loc) 20 20)
                         (c2d/set-color :blue)
                         (c2d/ellipse ex ey 20 20)))))

;; Variational Inference

;; using regular MH

(def true-mu 3.5)
(def true-sigma 0.8)
(def data (repeatedly 100 #(r/grand true-mu true-sigma)))

(defmodel gaussian-model
  [mu (:normal {:mu 0 :sd 20})
   sigma (:normal {:mu 0 :sd 5})]
  (model-result [(observe (distr :normal {:mu mu :sd (m/abs sigma)}) data)]))

(def post (infer :metropolis-hastings gaussian-model {:step-scale 0.01}))

(:acceptance-ratio post)

(plot/histogram (trace post :mu))
(plot/histogram (trace post :sigma))

;;

(def observed-luminance 3)

(defmodel model
  [reflectance (:normal {:mu 1 :sigma 1})
   illumination (:normal {:mu 3 :sigma 1})]
  (let [luminance (* reflectance illumination)]
    (model-result [(observe1 (distr :normal {:mu luminance :sd 1}) observed-luminance)])))

(def post (infer :metropolis-hastings model {:samples 15000 :thin 100 :max-iters 1e7}))

(:acceptance-ratio post)

(plot/density (traces post :reflectance :illumination))

