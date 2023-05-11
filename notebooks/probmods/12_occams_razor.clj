(ns probmods.12-occams-razor
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;;;;;;;;;;;;;;;;;
;; Occam's Razor
;;;;;;;;;;;;;;;;;;

;; The Size Principle

(def small-distr (distr :categorical-distribution {:data [:a :b :c]}))
(def big-distr (distr :categorical-distribution {:data [:a :b :c :d :e :f]}))

(def data [:a])

(defmodel model
  [hypothesis (distr :bernoulli)]
  (model-result [(observe (if (== 1.0 hypothesis) big-distr small-distr) data)]))

(def post (infer :rejection-sampling model {:samples 1000}))

;; 1.0 - big
;; 0.0 - small
(plot/frequencies (trace post :hypothesis))

(def full-data [:a :b :a :b :b :a :b])
(def data-sizes [0 1 2 3 4 5 6 7])

(defn hypothesis-posterior
  [data]
  (let [model (make-model
               [hypothesis (distr :bernoulli)]
               (model-result [(observe (if (== 1.0 hypothesis) big-distr small-distr) data)]))]
    (stats/mean (trace (infer :rejection-sampling model) :hypothesis))))

(def prob-big (map #(hypothesis-posterior (take % full-data)) data-sizes))

(plot/line (map vector data-sizes prob-big))

;; Example: The Rectangle Game

;; https://nextjournal.com/generateme/the-rectangle-game---mcmc-example-in-clojure/

;; Generalizing the Size Principle: Bayes Occam’s Razor

(def A (distr :categorical-distribution {:data [:a :b :c :d] :probabilities [0.375, 0.375, 0.125, 0.125]}))
(def B (distr :categorical-distribution {:data [:a :b :c :d]}))

(def observed-data [:a :b :a :b :c :d :b :b])

(defmodel model
  [hypothesis (:bernoulli)]
  (model-result [(observe (if (== 1.0 hypothesis) A B) observed-data)]
                (if (zero? hypothesis) :B :A)))

(def posterior (infer :metropolis-hastings model {:step-scale 1}))

(:acceptance-ratio posterior)

;; 1.0 - A
;; 0.0 - B
(plot/frequencies (trace posterior))

;; Example: Fair or unfair coin?

(def observed-data [:h :h :t :h :t :h :h :h :t :h])
(def observed-data (repeat 10 :h))
(def observed-data (repeat 25 :h))
(def observed-data (repeatedly 54 #(if (flipb 0.85) :h :t)))
(def fair-prior 0.999)
(def pseudo-counts (distr :beta {:alpha 1 :beta 1}))

(let [observed-data-ints (mapv #(if (= :h %) 1 0) observed-data)]
  (defmodel model
    [fair (:bernoulli {:p fair-prior})]
    (let [coin-weight (if (pos? fair) 0.5 (r/sample pseudo-counts))]
      (model-result [(observe (distr :bernoulli {:p coin-weight}) observed-data-ints)]
                    {:weight coin-weight
                     :prior (if (flipb fair-prior) 0.5 (r/sample pseudo-counts))}))))

(def results (infer :metropolis-hastings model {:samples 100000}))

(plot/histogram (trace results :prior))
(plot/frequencies (trace results :weight))
(plot/frequencies (trace results :fair))

(defn post-model
  [data]
  (let [observed-data-ints (mapv #(if (= :h %) 1 0) data)]
    (-> (infer :metropolis-hastings
               (make-model
                [fair (:bernoulli {:p fair-prior})]
                (let [coin-weight (if (pos? fair) 0.5 (r/sample pseudo-counts))]
                  (model-result [(observe (distr :bernoulli {:p coin-weight}) observed-data-ints)]
                                {:weight coin-weight
                                 :prior (if (flipb fair-prior) 0.5 (r/sample pseudo-counts))})))
               {:samples 10000})
        (trace :weight)
        (stats/mean))))

(def full-data-set (repeatedly #(randval 0.9 :h :t)))
(def data-sizes [0 1 3 6 10 20 30 40 50 60 70 100])
(def predictions (mapv (fn [s] (post-model (take s full-data-set))) data-sizes))

(plot/line (map vector data-sizes predictions))

;; The Effect of Unused Parameters

(def fair-prior 0.999)
(def observed-data [:h])

(let [observed-data-ints (mapv #(if (= :h %) 1 0) observed-data)]
  (defmodel model
    [fair (:bernoulli {:p fair-prior})
     unfair-weight (:beta {:alpha 1 :beta 1})]
    (let [coin-weight (if (pos? fair) 0.5 unfair-weight)]
      (model-result [(observe (distr :bernoulli {:p coin-weight}) observed-data-ints)]
                    {:weight coin-weight
                     :prior (if (flipb fair-prior) 0.5 (r/sample pseudo-counts))}))))

(def results (infer :metropolis-hastings model {:samples 1000000 :step-scale 1}))

(:acceptance-ratio results)

(plot/histogram (trace results :prior))
(plot/frequencies (trace results :weight))
(plot/histogram (trace results :unfair-weight))
(plot/frequencies (trace results :fair))

;; Example: Curve Fitting

(defn make-poly
  [as]
  (fn [^double x]
    (reduce m/fast+ (map-indexed (fn [^long i ^double a]
                                   (* a (m/pow x i))) as))))

(def observed-data [{:x -4,:y 69.76636938284166}
                    {:x -3,:y 36.63586217969598}
                    {:x -2,:y 19.95244368751754}
                    {:x -1,:y 4.819485497724985}
                    {:x 0,:y 4.027631414787425}
                    {:x 1,:y 3.755022418210824}
                    {:x 2,:y 6.557548104903805}
                    {:x 3,:y 23.922485493795072}
                    {:x 4,:y 50.69924692420815}])

(defmodel model
  [coeffs (:multi-normal {:means [0 0 0 0]
                          :covariances [[2 0 0 0]
                                        [0 2 0 0]
                                        [0 0 2 0]
                                        [0 0 0 2]]})
   order (:uniform-int {:upper 3})]
  (let [f (make-poly (take (inc order) coeffs))]
    (model-result (map #(observe1 (distr :normal {:mu (f (:x %)) :sd 2.0}) (:y %)) observed-data))))

(def post (infer :metropolis-hastings model {:step-scale 0.15
                                             :burn 2000
                                             :lag 5
                                             :samples 10000}))

(:acceptance-ratio post)

(plot/scatter (map (juxt :x :y) observed-data))
(plot/frequencies (trace post :order))

(def coeffs (trace post :coeffs))
(plot/histogram (map first coeffs))
(plot/histogram (map second coeffs))
(plot/histogram (map #(nth % 2) coeffs))
(plot/histogram (map #(nth % 3) coeffs))

(defn gen-data-set
  []
  (let [coeffs (repeatedly 3 #(r/grand 0 2))
        f (make-poly coeffs)]
    (println coeffs)
    (map #(hash-map :x % :y (f %)) (range -4 5 1))))

(def observed-data (gen-data-set))

;; Example: Scene Inference

(def x-size 4)
(def y-size 2)

(def background (for [x (range x-size)
                      y (range y-size)]
                  {:x x :y y :color 0}))

(defn layer
  [{:keys [^int x-loc ^int y-loc ^int h-size ^int v-size]
    :as object} image]
  (map (fn [{:keys [^int x ^int y]
            :as pixel}]
         (let [hits-object (and (>= x x-loc)
                                (< x (+ x-loc h-size))
                                (>= y y-loc)
                                (< y (+ y-loc v-size)))
               ncol (if hits-object (:color object) (:color pixel))]
           (assoc pixel :color ncol))) image))

(defn sample-properties
  []
  {:x-loc (r/irand x-size)
   :y-loc (r/irand y-size)
   :h-size 1
   :v-size (randval 1 2)
   :color 1})

(def observed-image (layer {:x-loc 1 :y-loc 0 :h-size 1 :v-size 2 :color 1} background))

(defmodel model
  [num-objects (:uniform-int {:lower 1 :upper 2})]
  (let [obj1 (sample-properties)
        obj2 (sample-properties)
        image1 (layer obj1 background)
        image (if (> num-objects 1) (layer obj2 image1) image1)]
    (model-result [(condition (= observed-image image))])))

(def results (infer :forward-sampling model))

(plot/frequencies (trace results :num-objects))

(def categorical (distr :integer-discrete-distribution {:data [-1 0 1]
                                                        :probabilities [0.3 0.4 0.3]}))

(defn move [object]
  (update object :x-loc m/fast+ (r/sample categorical)))

(def observed-image1 (layer {:x-loc 1 :y-loc 0 :h-size 1 :v-size 2 :color 1} background))
(def observed-image2 (layer {:x-loc 2 :y-loc 0 :h-size 1 :v-size 2 :color 1} background))

(defmodel model
  [num-objects (:uniform-int {:lower 1 :upper 2})]
  (let [obj1t1 (sample-properties)
        obj2t1 (sample-properties)
        obj1t2 (move obj1t1)
        obj2t2 (move obj2t1)
        image1 (layer obj1t1 background)
        image1 (if (> num-objects 1) (layer obj2t1 image1) image1)
        image2 (layer obj1t2 background)
        image2 (if (> num-objects 1) (layer obj2t2 image2) image2)]
    (model-result [(condition (= observed-image1 image1))
                   (condition (= observed-image2 image2))])))

(def results (infer :forward-sampling model))

(plot/frequencies (trace results :num-objects))
