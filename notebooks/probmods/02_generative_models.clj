(ns probmods.02-generative-models
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Generative models
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Building Generative Models

;; flip returns 0/1
;; flipb returns true/false

;; flib/flipb are macros which just point to the functions
;; fastmath.random/flip and fastmath.random/flipb

(flip)
;; => 1

(flipb)
;; => false

(plot/frequencies (repeatedly 1000 r/flipb))

(+ (flip) (flip) (flip))
;; => 1

(defn sum-flips [] (+ (flip) (flip) (flip)))
(plot/frequencies (repeatedly 100 sum-flips))

(defn noisy-double [^double x ] (randval (+ x x) x))
(noisy-double 3)
;; => 6.0

;; Example: Flipping Coins

(defn fair-coin [] (randval 0.5 :h :t))
(plot/frequencies (repeatedly 20 fair-coin))

(defn trick-coin [] (randval 0.95 :h :t))
(plot/frequencies (repeatedly 20 trick-coin))

(defn make-coin [weight]
  (fn [] (randval weight :h :t)))

(let [fair-coin (make-coin 0.5)
      trick-coin (make-coin 0.95)
      bent-coin (make-coin 0.25)]

  (plot/frequencies (repeatedly 20 fair-coin) {:title "Fair coin"})
  (plot/frequencies (repeatedly 20 trick-coin) {:title "Trick coin"})
  (plot/frequencies (repeatedly 20 bent-coin) {:title "Bent coin"}))

(defn bend [coin]
  (let [c07 (make-coin 0.7)
        c01 (make-coin 0.1)]
    (fn [] (if (= :h (coin)) (c07) (c01)))))

(let [fair-coin (make-coin 0.5)
      bent-coin (bend fair-coin)]
  (plot/frequencies (repeatedly 100 bent-coin)))

(defn make-coin [weight] #(flip weight))

(let [coin (make-coin 0.8)
      data (repeatedly 1000 #(reduce m/fast+ (repeatedly 10 coin)))]
  (plot/frequencies data {:x-label "# heads"}))

;; Example: Causal Models in Medical Diagnosis

(let [lung-cancer (flipb 0.01)
      cold (flipb 0.2)
      cough (or cold lung-cancer)]
  cough)

(let [lung-cancer (flipb 0.01)
      tb (flipb 0.005)
      stomach-flu (flipb 0.1)
      cold (flipb 0.2)
      other (flipb 0.1)
      cough (or (and cold (flipb 0.5))
                (and lung-cancer (flipb 0.3))
                (and tb (flipb 0.7))
                (and other (flipb 0.01)))
      fever (or (and cold (flipb 0.3))
                (and stomach-flu (flipb 0.5))
                (and tb (flipb 0.2))
                (and other (flipb 0.01)))
      chest-pain (or (and lung-cancer (flipb 0.5))
                     (and tb (flipb 0.5))
                     (and other (flipb 0.01)))
      shortness-of-breath (or (and lung-cancer (flipb 0.5))
                              (and tb (flipb 0.2))
                              (and other (flipb 0.01)))
      symptoms {:cough cough
                :fever fever
                :chest-pain chest-pain
                :shortness-of-breath shortness-of-breath}]
  symptoms)
;; => {:cough false,
;;     :fever false,
;;     :chest-pain false,
;;     :shortness-of-breath false}

;; Prediction, Simulation, and Probabilities

[(flipb) (flipb)]
;; => [false true]

(let [random-pair (fn [] {:a (flipb) :b (flipb)})]
  (plot/frequencies (repeatedly 1000 random-pair) {:sort? false}))

;; Distributions in WebPPL

;; distr is an alias to fastmath.random/distribution

(def b (distr :bernoulli {:p 0.5}))

(sample b)
;; => 1

;; score in WebPPL
(score b 1)
;; => -0.6931471805599453
;; same as `observe1`
(observe1 b 1)
;; => -0.6931471805599453

(plot/pdf b)

(def g (distr :normal {:mu 0 :sd 1}))

(sample g)
;; => 1.2672791517091984

(observe1 g 0)
;; => -0.9189385332046727

;; to sample from gaussian distribution you can use simple function fastmath.random/grand
(r/grand 0 1)
;; => 0.156551948577399

(plot/pdf g)

(defn foo [] (* (r/grand) (r/grand)))

(foo)
;; => -1.6924571180361319

;; Constructing marginal distributions

(def d (distr :continuous-distribution {:data (repeatedly 1000 foo)}))

(sample d)
;; => -0.7654022408226443

(plot/pdf d)

;; Product Rule

(let [a (flipb)
      b (flipb)
      c [a b]]
  c)
;; => [true false]

(let [a (flipb)
      b (if a (flipb 0.3) (flipb 0.7))]
  [a b])
;; => [false true]

(let [random-pair #(let [a (flipb)
                         b (if a (flipb 0.3) (flipb 0.7))]
                     (str {:a a :b b}))]
  (plot/frequencies (repeatedly 1000 random-pair) {:sort? true}))

;; Sum Rule
(or (flipb) (flipb)) ;; booleans: true/false
;; => false
(or+ (flip) (flip)) ;; numbers: 0/1
;; => 1

;; Stochastic recursion

(defn geometric
  ^long [p]
  (randval p 0 (inc (geometric p))))

(let [g (repeatedly 1000 #(geometric 0.6))]
  (plot/frequencies g))

(defn eye-color
  [person]
  (rand-nth [:blue :green :brown]))

[(eye-color :bob)
 (eye-color :alice)
 (eye-color :bob)]
;; => [:brown :brown :blue]
;; => [:blue :green :brown]

(= (flip) (flip))
;; => false
;; => true

;; Persistent Randomness

(def mem-flip (memoize r/flip))

(= (mem-flip) (mem-flip))
;; => true

(def eye-color
  (memoize (fn [person]
             (rand-nth [:blue :green :brown]))))

[(eye-color :bob)
 (eye-color :alice)
 (eye-color :bob)]
;; => [:green :blue :green]
;; => [:green :blue :green]

(def flip-a-lot (memoize (fn [n] (flipb))))

[[(flip-a-lot 1) (flip-a-lot 12) (flip-a-lot 47) (flip-a-lot 1548)]
 [(flip-a-lot 1) (flip-a-lot 12) (flip-a-lot 47) (flip-a-lot 1548)]]
;; => [[false false false true] [false false false true]]


;; Example: Intuitive physics

;; skipped 

