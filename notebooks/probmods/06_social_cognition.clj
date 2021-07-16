(ns probmods.06-social-cognition
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;; Prelude: Thinking About Assembly Lines

(def widget-machine (distr :real-discrete-distribution {:data [0.2 0.3 0.4 0.5 0.6 0.7 0.8]
                                                        :probabilities [0.05 0.1 0.2 0.3 0.2 0.1 0.05]}))

(defn get-good-widget
  ^double [^double tolerance]
  (let [^double widget (sample widget-machine)]
    (if (> widget tolerance)
      widget
      (recur tolerance))))

(def actual-widgets [0.6 0.7 0.8])

(defmodel widget-model
  [tolerance (:uniform-real {:lower 0.3 :upper 0.7})]
  (model-result [(condition (== (get-good-widget tolerance) ^double (actual-widgets 0)))
                 (condition (== (get-good-widget tolerance) ^double (actual-widgets 1)))
                 (condition (== (get-good-widget tolerance) ^double (actual-widgets 2)))]))

(def widget-result (infer :rejection-sampling widget-model))

(plot/histogram (trace widget-result :tolerance))

;;

(defn get-good-widget
  ^double [^double tolerance]
  (-> (infer :rejection-sampling (make-model
                                  []
                                  (let [^double widget (sample widget-machine)]
                                    (model-result [(condition (> widget tolerance))]
                                                  widget)))
             {:samples 15})
      (trace)
      (rand-nth)))

(defmodel widget-model
  [tolerance (:uniform-real {:lower 0.3 :upper 0.7})]
  (model-result [(condition (== (get-good-widget tolerance) ^double (actual-widgets 0)))
                 (condition (== (get-good-widget tolerance) ^double (actual-widgets 1)))
                 (condition (== (get-good-widget tolerance) ^double (actual-widgets 2)))]))

(def widget-result (infer :metropolis-hastings widget-model {:samples 100000
                                                             :thin 3
                                                             :step-scale 0.02
                                                             :initial-point [0.5]}))

(:acceptance-ratio widget-result)
(:out-of-prior widget-result)

(plot/histogram (trace widget-result :tolerance))

;;

(defn get-good-widget
  [^double tolerance]
  (-> (infer :rejection-sampling (make-model
                                  []
                                  (let [^double widget (sample widget-machine)]
                                    (model-result [(condition (> widget tolerance))]
                                                  widget)))
             {:samples 100})
      (as-real-discrete-distribution)))


(defmodel widget-model
  [tolerance (:uniform-real {:lower 0.3 :upper 0.7})]
  (let [good-widget-dist (get-good-widget tolerance)]
    (model-result [(observe good-widget-dist actual-widgets)])))

(def widget-result (infer :metropolis-hastings widget-model {:step-scale 0.02
                                                             :initial-point [0.5]}))

(:acceptance-ratio widget-result)
(:out-of-prior widget-result)

(plot/histogram (trace widget-result :tolerance))

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

(def goal-posterior (infer :metropolis-hastings goal-model {:samples 10000
                                                            :step-scale 0.07}))

(:acceptance-ratio goal-posterior)

(plot/frequencies (trace goal-posterior))

(defn vending-machine
  [state action]
  (case action
    :a (randval 0.1 :bagel :cookie)
    :b (randval 0.1 :bagel :cookie)
    :nothing))

(def goal-posterior (infer :metropolis-hastings goal-model {:samples 5000
                                                            :step-scale 0.07}))

(:acceptance-ratio goal-posterior)

(plot/frequencies (trace goal-posterior))

;; Inferring what they know

(def action-prior (distr :categorical-distribution {:data [:a :b]}))

(defn choose-action
  ([goal-state transition] (choose-action goal-state transition :start))
  ([goal-state transition state]
   (-> (infer :forward-sampling (make-model
                                 [] (let [action (sample action-prior)]
                                      (model-result
                                       [(condition (= goal-state (transition state action)))]
                                       action))) {:samples 100})
       (as-categorical-distribution ))))

(defmodel goal-model
  [a (:uniform-real)
   b (:uniform-real)]
  (let [vending-machine (fn [state action]
                          (randval (parameters-map action) :bagel :cookie))
        goal (randval :bagel :cookie)]
    (model-result [(condition (= :cookie goal))
                   (observe1 (choose-action goal vending-machine) :b)]
                  (fn [] {:button-a (vending-machine :state :a)
                         :button-b (vending-machine :state :b)}))))

(def goal-posterior (infer :metropolis-hastings goal-model {:step-scale 0.07
                                                            :initial-point [0.5 0.5]}))

(:acceptance-ratio goal-posterior)

(do
  (plot/frequencies (trace goal-posterior :button-a) {:title "button A"})
  (plot/frequencies (trace goal-posterior :button-b) {:title "button B"}))

;;

(def action-prior
  (distr :categorical-distribution {:data [:a :aa :aaa]
                                    :probabilities [0.7 0.2 0.1]}))


(defmodel goal-model
  [a (:uniform-real)
   aa (:uniform-real)
   aaa (:uniform-real)]
  (let [vending-machine (fn [state action]
                          (randval (parameters-map action) :bagel :cookie))
        goal (randval :bagel :cookie)]
    (model-result [(condition (= :cookie goal))
                   (observe1 (choose-action goal vending-machine) :a)]
                  (fn [] {:once (vending-machine :state :a)
                         :twice (vending-machine :state :aa)}))))

(def goal-posterior (infer :metropolis-hastings goal-model {:step-scale 0.07
                                                            :initial-point [0.5 0.5 0.5]}))

(:acceptance-ratio goal-posterior)

(do
  (plot/frequencies (trace goal-posterior :once) {:title "once"})
  (plot/frequencies (trace goal-posterior :twice) {:title "twice"}))


;; Joint inference about knowledge and goals

(defmodel goal-model
  [a (:uniform-real)
   aa (:uniform-real)
   aaa (:uniform-real)]
  (let [vending-machine (fn [state action]
                          (randval (parameters-map action) :bagel :cookie))
        goal (randval :bagel :cookie)]
    (model-result [(observe1 (choose-action goal vending-machine) :aa)
                   (condition (= (vending-machine :state :aa) :cookie))]
                  (fn [] {:goal goal
                         :one-press-result (vending-machine :state :a)
                         :two-press-result (vending-machine :state :aa)
                         :one-press-cookie-prob (- 1.0 a)}))))

(def goal-posterior (infer :metropolis-hastings goal-model {:samples 5000
                                                            :step-scale 0.07
                                                            :initial-point [0.5 0.5 0.5]}))

(do
  (plot/frequencies (trace goal-posterior :goal) {:title "goal"})
  (plot/frequencies (trace goal-posterior :one-press-result) {:title "one press result"})
  (plot/frequencies (trace goal-posterior :two-press-result) {:title "two press result"})
  (plot/histogram (trace goal-posterior :one-press-cookie-prob)))

;; Inferring whether they know

(def action-prior (distr :categorical-distribution {:data [:a :b]}))
(def buttons-to-bagel-probs {:a 0.9 :b 0.1})

(defn true-vending-machine
  [state action]
  (randval (buttons-to-bagel-probs action) :bagel :cookie))

(defn random-machine
  [state action]
  (randval :bagel :cookie))

(defmodel goal-model
  [knows (:bernoulli)]
  (model-result [(observe1 (choose-action :cookie (if (pos? knows) true-vending-machine random-machine)) :a)
                 (condition (= :bagel (true-vending-machine :start :a)))]))

(def goal-posterior (infer :rejection-sampling goal-model))

(plot/frequencies (trace goal-posterior :knows))

;; Inferring what they believe

(defmodel goal-model
  [a (:uniform-real)
   b (:uniform-real)]
  (let [sally-machine (fn [state action]
                        (randval (parameters-map action) :bagel :cookie))]
    (model-result [(observe1 (choose-action :cookie sally-machine) :a)
                   (condition (= :bagel (true-vending-machine :start :a)))]
                  (fn [] {:sally-belief-button-a (sally-machine :start :a)}))))

(def goal-posterior (infer :metropolis-hastings goal-model {:step-scale 0.07
                                                            :initial-point [0.5 0.5]}))

(plot/frequencies (trace goal-posterior :sally-belief-button-a))

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

(def teacher
  (memoize (fn [die depth]
             (infer :forward-sampling (make-model
                                       []
                                       (let [side (sample side-prior)] 
                                         (model-result [(condition (= die (rand-nth (trace (learner side depth)))))]
                                                       side)))))))

(def learner
  (memoize (fn [side ^long depth]
             (infer :forward-sampling (make-model
                                       []
                                       (let [die (sample die-prior)]
                                         (model-result [(condition (if (zero? depth)
                                                                     (= side (roll die))
                                                                     (= side (rand-nth (trace (teacher die (dec depth)))))))]
                                                       die)))))))

(def result (learner :green 3))

(plot/frequencies (trace result))

;; Communicating with Words

(defn all-sprouted [^long state] (== state 3))
(defn some-sprouted [^long state] (pos? state))
(defn none-sprouted [^long state] (zero? state))
(defn meaning
  [words]
  (case words
    :all all-sprouted
    :some some-sprouted
    :none none-sprouted
    (throw (Exception. "Unknown words"))))

(def state-prior (distr :integer-discrete-distribution {:data [0 1 2 3]}))
(def sentence-prior (distr :categorical-distribution {:data [:all :some :none]}))

(declare listener)

(def speaker
  (memoize (fn [state depth]
             (infer :forward-sampling (make-model
                                       []
                                       (let [words (sample sentence-prior)]
                                         (model-result [(condition (= state (rand-nth (trace (listener words depth)))))]
                                                       words)))))))

(def listener
  (memoize (fn [words ^long depth]
             (infer :forward-sampling (make-model
                                       []
                                       (let [state (sample state-prior)
                                             words-meaning (meaning words)]
                                         (model-result [(condition (if (zero? depth)
                                                                     (words-meaning state)
                                                                     (= words (rand-nth (trace (speaker state (dec depth)))))))]
                                                       state)))))))

(def result (listener :some 1))

(plot/frequencies (trace result))
