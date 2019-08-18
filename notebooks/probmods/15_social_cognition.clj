(ns probmods.15-social-cognition
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.vector :as v]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;; Prelude: Thinking About Assembly Lines

(def widget-machine (distr :real-discrete-distribution {:data [0.2 0.3 0.4 0.5 0.6 0.7 0.8]
                                                        :probabilities [0.05 0.1 0.2 0.3 0.2 0.1 0.05]}))

(defn make-widget-seq
  [^long num-widgets ^double threshold]
  (when-not (zero? num-widgets)
    (let [^double widget (sample widget-machine)] 
      (if (> widget threshold)
        (conj (make-widget-seq (dec num-widgets) threshold) widget)
        (make-widget-seq num-widgets threshold)))))

(defmodel widget-model
  [threshold (:real-discrete-distribution {:data [0.3 0.4 0.5 0.6 0.7]
                                           :probabilities [0.1 0.2 0.4 0.2 0.1]})]
  (let [good-widget-seq (make-widget-seq 3 threshold)]
    (model-result [(condition (= [0.6 0.7 0.8] good-widget-seq))])))

(def widget-result (infer :rejection-sampling widget-model))

(plot/frequencies (trace widget-result :threshold))

;;

(defn make-widget-seq
  [^long num-widgets ^double threshold]
  (as-categorical-distribution
   (infer :forward-sampling (make-model
                             []
                             (let [widgets (repeatedly num-widgets #(sample widget-machine))]
                               (model-result (map (fn [^double widget]
                                                    (condition (> widget threshold))) widgets)
                                             widgets))) {:samples 100})))


(defmodel widget-model
  [threshold (:real-discrete-distribution {:data [0.3 0.4 0.5 0.6 0.7]
                                           :probabilities [0.1 0.2 0.4 0.2 0.1]})]
  (let [good-widget-seq (make-widget-seq 3 threshold)]
    (model-result [(observe1 good-widget-seq [0.6 0.7 0.8])])))

(def widget-result (infer :rejection-sampling widget-model))

(plot/frequencies (trace widget-result :threshold))

;; Social Cognition

(def action-prior (distr :categorical-distribution {:data [:a :b]}))

(defn have-cookie [obj] (= obj :cookie))

(defn vending-machine
  [state action]
  (case action
    :a :bagel
    :b :cookie
    :nothing))

(defn choose-action
  [goal-satisfied transition state]
  (trace (infer :forward-sampling (make-model
                                   [] (let [action (sample action-prior)]
                                        (model-result
                                         [(condition (goal-satisfied (transition state action)))]
                                         action))))))

(plot/frequencies (choose-action have-cookie vending-machine :state))

(defn vending-machine
  [state action]
  (case action
    :a (randval 0.9 :bagel :cookie)
    :b (randval 0.1 :bagel :cookie)
    :nothing))

(plot/frequencies (choose-action have-cookie vending-machine :state))

;; Goal Inference

(defn choose-action
  [goal-satisfied transition state]
  (as-categorical-distribution (infer :forward-sampling (make-model
                                                         [] (let [action (sample action-prior)]
                                                              (model-result
                                                               [(condition (goal-satisfied (transition state action)))]
                                                               action))) {:samples 100})))

(defmodel goal-model
  []
  (let [goal (randval :bagel :cookie)
        goal-satisfied (fn [outcome] (= outcome goal))
        action-dist (choose-action goal-satisfied vending-machine :state)]
    (model-result [(observe1 action-dist :b)]
                  goal)))

(def goal-posterior (infer :rejection-sampling goal-model))

(plot/frequencies (trace goal-posterior))

(defn vending-machine
  [state action]
  (case action
    :a (randval 0.9 :bagel :cookie)
    :b (randval 0.5 :bagel :cookie)
    :nothing))

(def goal-posterior (infer :rejection-sampling goal-model))

(plot/frequencies (trace goal-posterior))

;; Preferences

(defn vending-machine
  [state action]
  (case action
    :a (randval 0.9 :bagel :cookie)
    :b (randval 0.1 :bagel :cookie)
    :nothing))

(defmodel goal-model
  [preference (:uniform-real)]
  (let [goal-prior #(randval preference :bagel :cookie)
        make-goal (fn [food] (fn [outcome] (= outcome food)))]
    (model-result [(condition (= :b (sample (choose-action (make-goal (goal-prior)) vending-machine :state))))
                   (condition (= :b (sample (choose-action (make-goal (goal-prior)) vending-machine :state))))
                   (condition (= :b (sample (choose-action (make-goal (goal-prior)) vending-machine :state))))]
                  (goal-prior))))

(def goal-posterior (infer :metropolis-hastings goal-model {:samples 20000}))

(:acceptance-ratio goal-posterior)

(plot/frequencies (trace goal-posterior))

(defn vending-machine
  [state action]
  (case action
    :a (randval 0.1 :bagel :cookie)
    :b (randval 0.1 :bagel :cookie)
    :nothing))

(def goal-posterior (infer :metropolis-hastings goal-model {:samples 5000
                                                            :step-scale 0.3}))

(:acceptance-ratio goal-posterior)

(plot/frequencies (trace goal-posterior))

;; Epistemic States

(defn make-vending-machine
  [a-effects b-effects]
  (let [a (first a-effects)
        b (first b-effects)]
    (fn [state action]
      (case action
        :a (randval a :bagel :cookie)
        :b (randval b :bagel :cookie)
        :nothing))))

(def dirichlet (distr :dirichlet {:alpha [1 1]}))

(defmodel goal-model
  [a-effects (:dirichlet {:alpha [1 1]})
   b-effects (:dirichlet {:alpha [1 1]})]
  (let [vending-machine (make-vending-machine a-effects b-effects)
        goal (randval :bagel :cookie)
        goal-satisfied (fn [outcome] (= outcome goal))]
    (model-result [(condition (and (= :cookie goal)
                                   (= :b (sample (choose-action goal-satisfied vending-machine :state)))))]
                  (second b-effects))))

(def goal-posterior (infer :metropolis-hastings goal-model {:samples 50000
                                                            :thin 15
                                                            :step-scale 0.2}))

(:acceptance-ratio goal-posterior)

(plot/lag (trace goal-posterior))

(plot/histogram (trace goal-posterior))

;;

(def action-prior
  (distr :categorical-distribution {:data [:a :aa :aaa]
                                    :probabilities [0.7 0.2 0.1]}))


(defn choose-action
  [goal-satisfied transition state]
  (first (trace (infer :rejection-sampling (make-model
                                            [] (let [action (sample action-prior)]
                                                 (model-result
                                                  [(condition (goal-satisfied (transition state action)))]
                                                  action))) {:samples 1}))))


(defmodel goal-model
  [a (:dirichlet {:alpha [1 1]})
   aa (:dirichlet {:alpha [1 1]})
   aaa (:dirichlet {:alpha [1 1]})]
  (let [vending-machine (fn [state action]
                          (randval (case action
                                     :a (first a)
                                     :aa (first aa)
                                     :aaa (first aaa)) :bagel :cookie))
        goal (randval :bagel :cookie)
        goal-satisfied (fn [outcome] (= outcome goal))
        chosen-action (choose-action goal-satisfied vending-machine :state)]
    (model-result [(condition (and (= goal :cookie)
                                   (= chosen-action :a)))]
                  {:once (second a)
                   :twice (second aa)})))

(def goal-posterior (infer :rejection-sampling goal-model {:samples 5000}))


#_(def goal-posterior (infer :metropolis-hastings goal-model {:samples 50000
                                                              :thin 5
                                                              :step-scale 0.2}))

(:acceptance-ratio goal-posterior)

(plot/histogram (trace goal-posterior :once))
(plot/histogram (trace goal-posterior :twice))

;; Joint inference about beliefs and desires

(defmodel goal-model
  [a (:dirichlet {:alpha [1 1]})
   aa (:dirichlet {:alpha [1 1]})
   aaa (:dirichlet {:alpha [1 1]})]
  (let [vending-machine (fn [state action]
                          (randval (case action
                                     :a (first a)
                                     :aa (first aa)
                                     :aaa (first aaa)) :bagel :cookie))
        goal (randval :bagel :cookie)
        goal-satisfied (fn [outcome] (= outcome goal))
        chosen-action (choose-action goal-satisfied vending-machine :state)]
    (model-result [(condition (= chosen-action :aa))
                   (condition (= :cookie (vending-machine :state :aa)))]
                  {:goal goal
                   :once (second a)
                   :twice (second aa)})))

(def goal-posterior (infer :rejection-sampling goal-model {:samples 5000}))

(plot/frequencies (trace goal-posterior :goal))
(plot/histogram (trace goal-posterior :once))
(plot/histogram (trace goal-posterior :twice))

;; No enumeration creates wrong results or errors in below example...

;; A Communication Game

(defn die-to-probs
  [die]
  (case die
    :A [0 0.2 0.8]
    :B [0.1 0.3 0.6]))

(def side-prior (distr :categorical-distribution {:data [:red :green :blue]}))
(def die-prior (distr :categorical-distribution {:data [:A :B]}))
(def die-distr (memoize (fn [die]
                          (distr :categorical-distribution {:data [:red :green :blue]
                                                            :probabilities (die-to-probs die)}))))
(defn roll
  [die]
  (sample (die-distr die)))

(declare learner)

(defn teacher
  [die depth]
  (as-categorical-distribution
   (infer :forward-sampling (make-model
                             []
                             (let [side (sample side-prior)] 
                               (model-result [(condition (= die (sample (learner side depth))))]
                                             side))) {:samples 100})))

(defn learner
  [side ^long depth]
  (as-categorical-distribution
   (infer :forward-sampling (make-model
                             []
                             (let [die (sample die-prior)]
                               (model-result [(condition (if (zero? depth)
                                                           (= side (roll die))
                                                           (= side (sample (teacher die (dec depth))))))]
                                             die))) {:samples 100})))

(def result (learner :green 1))

(plot/frequencies (r/->seq result 10000))

;; skipped last, due to problematic inference
