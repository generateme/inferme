(ns inferme.plot
  (:refer-clojure :exclude [frequencies])
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [cljplot.core :as plot]
            [cljplot.build :as b]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

(defn- maybe-add-title
  [plot title]
  (if-not title
    plot
    (b/add-label plot :top title {:font-size 16 :font-style :bold})))

(defn frequencies
  ([data] (frequencies data {}))
  ([data {:keys [x-label y-label sort? title]
          :or {y-label "probability"
               x-label "data"
               sort? true}}]
   (-> (plot/xy-chart {:width 500 :height 500}
                      (b/series [:grid]
                                [:frequencies data
                                 {:range? true :sort? sort? :pmf? true}])
                      (b/update-scale :x :fmt (fn [v] (if (nil? v) "/nil/" v)))
                      (b/add-axes :bottom)
                      (b/add-axes :left)
                      (b/add-label :bottom x-label)
                      (b/add-label :left y-label)
                      (maybe-add-title title))
       (plot/show))))

(defn histogram
  ([data] (histogram data {}))
  ([data {:keys [x-label y-label bins title]
          :or {y-label "probability"
               x-label "data"}}]
   (-> (plot/xy-chart {:width 500 :height 500}
                      (b/series [:grid]
                                [:histogram data
                                 {:bins bins :density? true :padding-out -0.1 :stroke? false}]
                                [:density data {:color [10 10 10 80] :area? true}])
                      (b/update-scale :x :fmt (fn [v] (if (nil? v) "/nil/" v)))
                      (b/add-axes :bottom)
                      (b/add-axes :left)
                      (b/add-label :bottom x-label)
                      (b/add-label :left y-label)
                      (maybe-add-title title))
       (plot/show))))


(defn pdf
  [d]
  (let [l (r/lower-bound d)
        u (r/upper-bound d)
        l (m/dec (double (if (m/invalid-double? l) -10 l)))
        u (+ 2.0 (double (if (m/invalid-double? u) 10 u)))] 
    (-> (plot/xy-chart {:width 500 :height 500}
                       (b/series [:grid]
                                 [:function (partial r/pdf d) {:domain [l u]
                                                               :samples 400}])
                       (b/add-axes :bottom)
                       (b/add-axes :left)
                       (b/add-label :bottom "Distribution")
                       (b/add-label :left "PDF"))
        (plot/show))))

(defn scatter
  [d]
  (-> (plot/xy-chart {:width 500 :height 500}
                     (b/series [:grid]
                               [:scatter d {:size 10 :jitter 0.1}])
                     (b/add-axes :bottom)
                     (b/add-axes :left))
      (plot/show)))

(defn heatmap
  [d]
  (-> (plot/xy-chart {:width 500 :height 500}
                     (b/series [:grid]
                               [:heatmap d {:annotate? true
                                            :annotate-fmt "%.3f"}])
                     (b/add-axes :bottom)
                     (b/add-axes :left))
      (plot/show)))

(defn density
  [d]
  (-> (plot/xy-chart {:width 500 :height 500}
                     (b/series [:grid]
                               [:density-2d d])
                     (b/add-axes :bottom)
                     (b/add-axes :left))
      (plot/show)))

(defn line
  [d]
  (let [[_ ^double max-y] (stats/extent (map second d))]
    (-> (plot/xy-chart {:width 500 :height 500}
                       (b/series [:grid]
                                 [:line d])
                       (b/update-scale :y :domain [0 (* 1.1 max-y)])
                       (b/add-axes :bottom)
                       (b/add-axes :left))
        (plot/show))))

(defn lag
  [d]
  (-> (plot/xy-chart {:width 500 :height 500}
                     (b/series [:grid nil {:position [0 1]}] [:acf d {:lags 50 :position [0 1] :label "ACF"}]
                               [:grid nil {:position [0 0]}] [:pacf d {:lags 50 :position [0 0] :label "PACF"}])
                     (b/update-scales :x :fmt int)
                     (b/add-axes :bottom)
                     (b/add-axes :left)
                     (b/add-label :bottom "lag")
                     (b/add-label :left "autocorrelation"))
      (plot/show)))

(defn bar
  [d]
  (-> (plot/xy-chart {:width 500 :height 500}
          (b/series [:grid]
                    [:stack-vertical [:bar d]])
        (b/add-axes :bottom)
        (b/add-axes :left))
      (plot/show)))

(m/unuse-primitive-operators)
