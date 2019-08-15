(ns probmods.06-bayesian-data-analysis
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all] 
            [inferme.plot :as plot]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Bayesian data analysis
;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; People’s models of coins

(defmodel observer-model
  [p (:uniform-real)]
  (let [coin-spinner (distr :binomial {:trials 20 :p p})]
    (model-result [(observe1 coin-spinner 15)])))

(let [posterior-beliefs (infer :rejection-sampling observer-model)]
  (plot/histogram (trace posterior-beliefs :p))
  (r/mean (as-real-discrete-distribution posterior-beliefs :p)));; => 0.727514037337359

(defmodel observer-model
  [p (:uniform-real)]
  (let [coin-spinner (distr :binomial {:trials 20 :p p})
        bernoulli (distr :bernoulli {:p p})
        binomial (distr :binomial {:trials 10 :p p})]
    (model-result [(observe1 coin-spinner 15)]
                  {:next-outcome (r/sample bernoulli)
                   :next-ten-outcomes (r/sample binomial)})))

(let [posterior-beliefs (infer :metropolis-hastings observer-model)]
  (plot/frequencies (trace posterior-beliefs :next-outcome))
  (plot/frequencies (trace posterior-beliefs :next-ten-outcomes)))

(defmodel sceptical-model
  []
  (let [same-as-flipping (flipb 0.5)
        p (if same-as-flipping 0.5 (r/drand))
        coin-spinner (distr :binomial {:trials 20 :p p})
        bernoulli (distr :bernoulli {:p p})
        binomial (distr :binomial {:trials 10 :p p})]
    (model-result [(observe1 coin-spinner 15)]
                  {:same-as-flipping same-as-flipping
                   :p p
                   :next-outcome (r/sample bernoulli)
                   :next-ten-outcomes (r/sample binomial)})))

(let [posterior-beliefs (infer :rejection-sampling sceptical-model)]
  (plot/frequencies (trace posterior-beliefs :same-as-flipping))
  (plot/histogram (trace posterior-beliefs :p) {:bins 20})
  (plot/frequencies (trace posterior-beliefs :next-outcome))
  (plot/frequencies (trace posterior-beliefs :next-ten-outcomes)))

;; Scientists’ models of people

(defmodel observer-model
  [p (:uniform-real)]
  (let [coin-spinner (distr :binomial {:trials 20 :p p})
        binomial (distr :binomial {:trials 10 :p p})]
    (model-result [(observe1 coin-spinner 15)]
                  (r/sample binomial))))

(def observer-inference (infer :rejection-sampling observer-model {:samples 5000}))
(plot/frequencies (trace observer-inference :model-result))

(defmodel sceptical-model
  []
  (let [same-as-flipping (flipb 0.5)
        p (if same-as-flipping 0.5 (r/drand))
        coin-spinner (distr :binomial {:trials 20 :p p})
        binomial (distr :binomial {:trials 10 :p p})]
    (model-result [(observe1 coin-spinner 15)]
                  (r/sample binomial))))

(def sceptical-inference (infer :rejection-sampling sceptical-model {:samples 5000}))
(plot/frequencies (trace sceptical-inference :model-result))

(def experimental-data [9 8 7 5 4 5 6 7 9 4 8 7 8 3 9 6 5 7 8 5])
(def model-object {:observer-model (as-integer-discrete-distribution observer-inference :model-result)
                   :sceptical-model (as-integer-discrete-distribution sceptical-inference :model-result)})

(defmodel scientist-model
  [which (:bernoulli)]
  (let [the-better-model-name (if (zero? which) :observer-model :sceptical-model)
        the-better-model (model-object the-better-model-name)]
    (model-result [(observe the-better-model experimental-data)]
                  {:better-model the-better-model-name})))

(let [inferred-model (infer :metropolis-hastings scientist-model)]
  (plot/frequencies (trace inferred-model :better-model)))

;; A simple illustration

(def k 1)
(def n 20)

(defmodel model
  [p (:uniform-real)]
  (let [binomial (distr :binomial {:trials n :p p})
        posterior-predictive (r/sample binomial)
        prior-p (r/drand)
        prior-predictive (r/sample (distr :binomial {:trials n :p prior-p}))]
    (model-result [(observe1 binomial k)]
                  {:prior prior-p :prior-predictive prior-predictive
                   :posterior p :posterior-predictive posterior-predictive})))

(let [inferred-model (infer :rejection-sampling model {:samples 2000})]
  (plot/histogram (trace inferred-model :prior))
  (plot/frequencies (trace inferred-model :prior-predictive))
  (plot/histogram (trace inferred-model :posterior))
  (plot/frequencies (trace inferred-model :posterior-predictive)))

;; Posterior prediction and model checking

(def k1 0)
(def n1 10)
(def k2 10)
(def n2 10)

(defmodel model
  [p (:uniform-real)]
  (let [binomial1 (distr :binomial {:trials n1 :p p})
        binomial2 (distr :binomial {:trials n2 :p p})
        posterior-predictive1 (r/sample binomial1)
        posterior-predictive2 (r/sample binomial2)]
    (model-result [(observe1 binomial1 k1)
                   (observe1 binomial2 k2)]
                  {:posterior-predictive1 posterior-predictive1
                   :posterior-predictive2 posterior-predictive2})))

(def posterior (infer :metropolis-hastings model {:samples 20000 :burn 10000}))

(plot/histogram (trace posterior :p))

(let [tp (traces posterior :posterior-predictive1 :posterior-predictive2)
      cnt (double (count tp))
      f (frequencies (traces posterior :posterior-predictive1 :posterior-predictive2))
      to-viz (map (fn [[k ^long v]]
                    [k (/ v cnt)]) f)]
  (plot/heatmap to-viz))

(plot/frequencies (trace posterior :posterior-predictive1))

(let [k1 0
      k2 10
      ppmap1 (r/mean (as-integer-discrete-distribution posterior :posterior-predictive1))
      ppmap2 (r/mean (as-integer-discrete-distribution posterior :posterior-predictive2))]
  (plot/scatter [[ppmap1 k1] [ppmap2 k2]]))

;; Comparing hypotheses

(def k 7)
(def n 20)

(defmodel compare-models
  [prior (:bernoulli)]
  (let [x (if (pos? prior) :simple :complex)
        p (if (= x :simple) 0.5 (r/drand))
        binomial (distr :binomial {:trials n :p p})]
    (model-result [(observe1 binomial k)]
                  {:model x})))

(let [model-posterior (infer :rejection-sampling compare-models {:samples 2000})]
  (plot/frequencies (trace model-posterior :model)))

;; Bayes’ factor

(def simple-likelihood (m/exp (r/lpdf (distr :binomial {:trials n :p 0.5}) k)))
(def complex-model (distr :integer-discrete-distribution
                          {:data (repeatedly 10000 #(r/sample (distr :binomial {:trials n :p (r/drand)})))}))
(def complex-likelihood (m/exp (r/lpdf complex-model k)))
(def bayes-factor (/ ^double simple-likelihood ^double complex-likelihood))

bayes-factor
;; => 1.6212463378906243

;; Savage-Dickey method
;; not sure if it's the same

(defmodel complex-model-prior []
  (trace-result (r/drand)))

(def cmprior (infer :forward-sampling complex-model-prior {:samples 10000}))
(def cmprior-distr (as-continuous-distribution cmprior :model-result))

(defmodel complex-model-posterior
  [p (:uniform-real)]
  (let [binomial (distr :binomial {:trials n :p p})]
    (model-result [(observe1 binomial k)])))

(def cmposterior (infer :rejection-sampling complex-model-posterior {:samples 10000}))
(def cmposterior-distr (as-continuous-distribution cmposterior :p))

;; should be ok...
(def savage-dickey-denominator (r/pdf cmprior-distr 0.5))
(def savage-dickey-numerator (r/pdf cmposterior-distr 0.5))
(/ savage-dickey-numerator savage-dickey-denominator)
;; => 1.5887164059211862

;; Example: Linear regression and tug of war

(def tow-data (edn/read (java.io.PushbackReader. (io/reader "notebooks/probmods/towdata.edn"))))

(first tow-data)
;; => {:uniqueCondition "win_confounded evidence_single",
;;     :roundedRating 0.6,
;;     :outcome "win",
;;     :tournament "single",
;;     :id 1,
;;     :trial 1,
;;     :binaryResponse true,
;;     :nUniqueWins 1,
;;     :nWins 3,
;;     :ratingZ 0.6094,
;;     :rating 61,
;;     :pattern "confounded evidence"}

(defn levels [d a] (distinct (map a d)))

(levels tow-data :pattern)
;; => ("confounded evidence"
;;     "strong indirect evidence"
;;     "weak indirect evidence"
;;     "diverse evidence"
;;     "confounded with partner"
;;     "confounded with opponent"
;;     "round robin")

(levels tow-data :tournament)
;; => ("single" "double")

(levels tow-data :nWins)
;; => (3 -3 1 -1)

(levels tow-data :id)
;; => (1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

(plot/histogram (map :ratingZ tow-data) {:bins 40})

;; Single regression

;; there is no option to change the proposal kernel

(defmodel single-regression
  [b0 (:uniform-real {:lower -1.0 :upper 1.0})
   b1 (:uniform-real {:lower -1.0 :upper 1.0})
   sigma (:uniform-real {:lower 0.0 :upper 2.0})]
  (model-result (map (fn [d]
                       (let [predicted-y (+ b0 (* b1 ^double (:nWins d)))]
                         (observe1 (distr :normal {:mu predicted-y :sd sigma}) (:ratingZ d)))) tow-data)))


(def posterior (infer :metropolis-hastings single-regression {:samples 2500 :burn 1250 :step-scale 0.022}))

(:acceptance-ratio posterior)
;; => 0.43093333333333333

(plot/histogram (trace posterior :b0))
(plot/histogram (trace posterior :b1))
(plot/histogram (trace posterior :sigma))

;; Model criticism with posterior prediction

(def tow-data-grouped (->> tow-data
                           (group-by (juxt :pattern :tournament :outcome))
                           (map (fn [[k v]]
                                  [(str/join " - " k)
                                   (map #(select-keys % [:nWins :nUniqueWins :ratingZ :roundedRating]) v)]))
                           (into (sorted-map))))

(def tow-means (->> tow-data-grouped
                    (map (fn [[k v]]
                           [k (stats/mean (map :ratingZ v))]))
                    (into (sorted-map))))

(first tow-data-grouped)
;; => ["confounded evidence - single - loss"
;;     ({:nWins -3, :nUniqueWins -1, :ratingZ -2.1501, :roundedRating -2.2}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.3681, :roundedRating -0.4}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.5718, :roundedRating -0.6}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -1.0805, :roundedRating -1.1}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.6957, :roundedRating -0.7}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -1.2063, :roundedRating -1.2}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -1.9003, :roundedRating -1.9}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -1.192, :roundedRating -1.2}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.7595, :roundedRating -0.8}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.9378, :roundedRating -0.9}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -1.4846, :roundedRating -1.5}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -1.284, :roundedRating -1.3}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.1179, :roundedRating -0.1}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -1.5312, :roundedRating -1.5}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.9646, :roundedRating -1}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.8666, :roundedRating -0.9}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.3911, :roundedRating -0.4}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.2657, :roundedRating -0.3}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.9409, :roundedRating -0.9}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -1.5363, :roundedRating -1.5}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.4151, :roundedRating -0.4}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -1.4489, :roundedRating -1.4}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.8638, :roundedRating -0.9}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.6712, :roundedRating -0.7}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.9754, :roundedRating -1}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.9177, :roundedRating -0.9}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.1317, :roundedRating -0.1}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -0.5948, :roundedRating -0.6}
;;      {:nWins -3, :nUniqueWins -1, :ratingZ -1.0307, :roundedRating -1}
;;      {:nWins -3,
;;       :nUniqueWins -1,
;;       :ratingZ -0.6221,
;;       :roundedRating -0.6})]

(defmodel single-regression
  [b0 (:uniform-real {:lower -1.0 :upper 1.0})
   b1 (:uniform-real {:lower -1.0 :upper 1.0})
   sigma (:uniform-real {:lower 0.0 :upper 2.0})]
  (let [predicted-ys (into {} (map (fn [[k v]]
                                     [k (+ b0 (* b1 ^double (:nWins (first v))))]) tow-data-grouped))]
    (model-result (map (fn [[k y]]
                         (observe (distr :normal {:mu y :sd sigma})
                                  (map :ratingZ (tow-data-grouped k))))
                       predicted-ys)
                  predicted-ys)))

(def posterior (infer :metropolis-hastings single-regression {:samples 2500 :burn 1250 :step-scale 0.022}))

(:acceptance-ratio posterior)

(plot/histogram (trace posterior :b0))
(plot/histogram (trace posterior :b1))
(plot/histogram (trace posterior :sigma))

(def model-data-df (map (fn [[k d]]
                          (let [m (stats/mean (trace posterior k))]
                            [m d])) tow-means))

(plot/scatter model-data-df)

(def mean-squared-error (stats/mean (map (fn [[^double m ^double d]]
                                           (m/sq (- m d))) model-data-df)))

mean-squared-error
;; => 0.07023507698848784

(def variance-explained (m/sq (stats/correlation (map first model-data-df)
                                                 (map second model-data-df))))

variance-explained
;; => 0.911251741324505

;; Mutiple regression

(defmodel multiple-regression
  [b0 (:uniform-real {:lower -1.0 :upper 1.0})
   b1 (:uniform-real {:lower -1.0 :upper 1.0})
   b2 (:uniform-real {:lower -1.0 :upper 1.0})
   sigma (:uniform-real {:lower 0.0 :upper 2.0})]
  (let [predicted-ys (into {} (map (fn [[k v]]
                                     (let [fv (first v)]
                                       [k (+ b0
                                             (* b1 ^double (:nWins fv))
                                             (* b2 ^double (:nUniqueWins fv)))])) tow-data-grouped))]
    (model-result (map (fn [[k y]]
                         (observe (distr :normal {:mu y :sd sigma})
                                  (map :ratingZ (tow-data-grouped k)))) predicted-ys)
                  predicted-ys)))

(def posterior (infer :metropolis-hastings multiple-regression {:samples 2500 :burn 1250 :step-scale 0.018}))

(:acceptance-ratio posterior)

(plot/histogram (trace posterior :b0))
(plot/histogram (trace posterior :b1))
(plot/histogram (trace posterior :b2))
(plot/histogram (trace posterior :sigma))

(def model-data-df (map (fn [[k d]]
                          (let [m (stats/mean (trace posterior k))]
                            [m d])) tow-means))

(plot/scatter model-data-df)

(def mean-squared-error (stats/mean (map (fn [[^double m ^double d]]
                                           (m/sq (- m d))) model-data-df)))

mean-squared-error
;; => 0.06441573410243007

(def variance-explained (m/sq (stats/correlation (map first model-data-df)
                                                 (map second model-data-df))))

variance-explained
;; => 0.9186072006307192

;; BDA of Tug-of-war model

(def ^:const ^double laziness-prior 0.3)
(def ^:const ^double lazy-pulling 0.5)

(defmodel model
  []
  (let [strength (memoize (fn [person] (r/grand)))
        lazy (fn [person] (flipb laziness-prior))
        pulling (fn [person] (if (lazy person)
                               (* lazy-pulling ^double (strength person))
                               (strength person)))
        total-pulling (fn [team] (stats/sum (map pulling team)))
        winner (fn [team1 team2]
                 (if (> ^double (total-pulling team1) ^double (total-pulling team2))
                   team1 team2))
        beat (fn [team1 team2] (= (winner team1 team2) team1))] 
    (model-result [(condition (beat [:bob :mary] [:tom :sue]))]
                  (strength :bob))))

(def posterior (infer :rejection-sampling model {:samples 1000}))

(r/mean (as-real-discrete-distribution posterior :model-result))
;; => 0.33524566043031184

(plot/histogram (trace posterior :model-result))

;; Learning about the Tug-of-War model

(def match-configurations (edn/read (java.io.PushbackReader. (io/reader "notebooks/probmods/match.edn"))))

(def match-information (->> match-configurations
                            (group-by (juxt :pattern :tournament :outcome))
                            (map (fn [[k v]]
                                   [(str/join " - " k) (first v)]))
                            (into (sorted-map))))

(def bins (map #(m/approx % 1) (range -2.2 2.2 0.1)))

(def smooth-distr
  (memoize (fn [x]
             (let [smooth-probs (mapv #(let [d (distr :normal {:mu x :sd 0.05})]
                                         (+ 1.0e-6 ^double (r/pdf d %))) bins)]
               (distr :real-discrete-distribution {:data bins :probabilities smooth-probs})))))

(defn smooth-to-bins
  [dist]
  (distr :real-discrete-distribution
         {:data
          (repeatedly 500 #(r/sample (smooth-distr (r/sample dist))))}))

(defn pulling
  ^double [strength ^double laziness-prior person]
  (if (flipb laziness-prior)
    (* lazy-pulling ^double (strength person)) (strength person)))

(defn total-pulling [pulling team] (stats/sum (map pulling team)))
(defn winner [pulling team1 team2]
  (if (> ^double (total-pulling pulling team1) ^double (total-pulling pulling team2))
    team1 team2))
(defn beat [pulling team1 team2] (= (winner pulling team1 team2) team1))

(defn tug-of-war-model
  [^double lazy-pulling laziness-prior match-info] 
  (let [model (make-model
               [] (let [strength (memoize (fn [person] (r/grand))) 
                        local-pulling (partial pulling strength laziness-prior)]
                    (model-result [(condition (beat local-pulling (:winner1 match-info) (:loser1 match-info)))
                                   (condition (beat local-pulling (:winner2 match-info) (:loser2 match-info)))
                                   (condition (beat local-pulling (:winner3 match-info) (:loser3 match-info)))]
                                  (m/approx (strength "A") 1))))]
    (as-real-discrete-distribution (infer :rejection-sampling model {:samples 500}) :model-result)))

(plot/frequencies (r/->seq (tug-of-war-model 0.3 0.5 (rand-nth match-configurations)) 10000))

(let [d (tug-of-war-model 0.2 0.5 (rand-nth match-configurations))
      d2 (smooth-to-bins d)]
  (plot/histogram (r/->seq d 10000))
  (plot/histogram (r/->seq d2 10000)))

(def rounded-rating-data (into {} (map #(vector % (map :roundedRating (tow-data-grouped %)))
                                       (keys tow-data-grouped))))

(defmodel data-analysis-model
  [laziness-prior (:uniform-real {:lower 0.0 :upper 0.5})
   lazy-pulling (:uniform-real {:lower 0.0 :upper 1})]
  (let [data (pmap (fn [[k v]]
                     (let [mi (match-information k)
                           mp (tug-of-war-model lazy-pulling laziness-prior mi)
                           sp (smooth-to-bins mp)]
                       [k (r/mean mp) sp])) tow-data-grouped)]
    (model-result (mapv (fn [[k _ sp]]
                          (observe sp (rounded-rating-data k)))
                        data)
                  (into {} (map (comp vec butlast) data)))))

(def posterior (infer :metropolis-hastings data-analysis-model {:samples 50 :burn 0 :step-scale 0.05 :max-time 60
                                                                :initial [0.3 0.5]}))

(:acceptance-ratio posterior)
;; => 0.04

(count (:accepted posterior))
;; => 50

(plot/histogram (trace posterior :laziness-prior))
(plot/histogram (trace posterior :lazy-pulling))

(def model-data-df (map (fn [[k d]]
                          (let [m (stats/mean (trace posterior k))]
                            [m d])) tow-means))

(plot/scatter model-data-df)

(def mean-squared-error (stats/mean (map (fn [[^double m ^double d]]
                                           (m/sq (- m d))) model-data-df)))

mean-squared-error
;; => 0.03527753184444449

(def variance-explained (m/sq (stats/correlation (map first model-data-df)
                                                 (map second model-data-df))))

variance-explained
;; => 0.9580840169470687

