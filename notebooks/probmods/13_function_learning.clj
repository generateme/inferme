(ns probmods.13-function-learning
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.vector :as v]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;; Fitting curves with neural nets

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
    (model-result (map #(observe1 (distr :normal {:mu (f (:x %)) :sd 2.0}) (:y %)) observed-data)
                  {:result (conj coeffs order)})))

(def post (infer :metropolis-hastings model {:step-scale 0.15
                                             :burn 2000
                                             :lag 5
                                             :samples 100}))

(:acceptance-ratio post)

(plot/scatter (map (juxt :x :y) observed-data))
(plot/frequencies (trace post :order))

(def post-distribution (as-categorical-distribution post :result))

(defn post-fn-sample
  []
  (let [p (sample post-distribution)]
    (make-poly (take (int (inc ^double (last p))) p))))

(plot/line (let [f (post-fn-sample)]
             (map #(vector %1 (f %1)) (range -5 5 0.1))))

;;

(def dm 10)

(defn make-fn
  [M1 M2 B1]
  (fn [^double x]
    (-> (v/mult M1 x)
        (v/add B1)
        (v/sigmoid)
        (v/dot M2))))

#_(let [f (make-fn (v/generate-vec4 r/grand) (v/generate-vec4 r/grand) (v/generate-vec4 r/grand))]
    (plot/line (map #(vector %1 (f %1)) (range -5 5 0.1))))

(defmodel model
  [M1 (:multi-normal {:means (repeat dm 1.0)})
   M2 (:multi-normal {:means (repeat dm 1.0)})
   B1 (:multi-normal {:means (repeat dm 1.0)})]
  (let [f (make-fn M1 M2 B1)]
    (model-result (map #(observe1 (distr :normal {:mu (f (:x %)) :sd 0.05}) (:y %)) observed-data)
                  {:result parameters-map})))

(def post (infer :metropolis-hastings model {:step-scale 0.01
                                             :burn 10000
                                             :thin 3
                                             :samples 1000}))

(:acceptance-ratio post)

(def post-distribution (as-categorical-distribution post :result))

(defn post-fn-sample
  []
  (let [p (sample post-distribution)]
    (make-fn (:M1 p) (:M2 p) (:B1 p))))

(plot/line (let [f (post-fn-sample)]
             (map #(vector %1 (f %1)) (range -5 5 0.1))))

;; version with Delta guide skipped

;; Deep generative models

;; skipped no option to generate random matrices

