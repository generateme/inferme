(ns probmods.11-hierarchical-models
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [fastmath.vector :as v]
            [inferme.core :refer :all]
            [inferme.plot :as plot]
            [inferme.jump :as jump]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;;;;;;;;;;;;;;;;;;;;;;;
;; Hierarchical models
;;;;;;;;;;;;;;;;;;;;;;;;

;; Learning a Shared Prototype: Abstraction at the Basic Level

(def colors [:black :blue :green :orange :red])

(def dirichlet (distr :dirichlet {:alpha (repeat (count colors) 1)}))

(def make-bag (memoize (fn [bag-name]
                         (distr :integer-discrete-distribution
                                {:data (range (count colors))
                                 :probabilities (r/sample dirichlet)}))))

(defn draw-marbles
  [bag-name num-draws]
  (map colors (r/->seq (make-bag bag-name) num-draws)))

(plot/frequencies (draw-marbles :bag-a 100))
(plot/frequencies (draw-marbles :bag-a 100))
(plot/frequencies (draw-marbles :bag-a 100))
(plot/frequencies (draw-marbles :bag-b 100))

(def colors-map (zipmap colors (range (count colors))))

(def observed-data
  [{:bag :bag1 :draw :blue}
   {:bag :bag1 :draw :blue}
   {:bag :bag1 :draw :black}
   {:bag :bag1 :draw :blue}
   {:bag :bag1 :draw :blue}
   {:bag :bag1 :draw :blue}
   {:bag :bag2 :draw :blue}
   {:bag :bag2 :draw :green}
   {:bag :bag2 :draw :blue}
   {:bag :bag2 :draw :blue}
   {:bag :bag2 :draw :blue}
   {:bag :bag2 :draw :red}
   {:bag :bag3 :draw :blue}
   {:bag :bag3 :draw :orange}])

(defmodel predictives-model
  []
  (let [make-bag (memoize (fn [bag-name]
                            (distr :integer-discrete-distribution
                                   {:data (range (count colors))
                                    :probabilities (r/sample dirichlet)})))] 
    (model-result (map (fn [datum]
                         (observe1 (make-bag (:bag datum)) (colors-map (:draw datum)))) observed-data)
                  (fn [] {:bag1 (colors (r/sample (make-bag :bag1)))
                         :bag2 (colors (r/sample (make-bag :bag2)))
                         :bag3 (colors (r/sample (make-bag :bag3)))
                         :bagN (colors (r/sample (make-bag :bagN)))}))))

(def predictives (infer :metropolis-hastings predictives-model))

(:acceptance-ratio predictives)

(plot/frequencies (trace predictives :bag1))
(plot/frequencies (trace predictives :bag2))
(plot/frequencies (trace predictives :bag3))
(plot/frequencies (trace predictives :bagN))

(defmodel predictives-model
  [prototype (:dirichlet {:alpha (repeat 5 1)})]
  (let [dirichlet (distr :dirichlet {:alpha (v/mult prototype 5)})
        make-bag (memoize (fn [bag-name]
                            (distr :integer-discrete-distribution
                                   {:data (range (count colors))
                                    :probabilities (r/sample dirichlet)})))] 
    (model-result (map (fn [datum]
                         (observe1 (make-bag (:bag datum)) (colors-map (:draw datum)))) observed-data)
                  (fn [] {:bag1 (colors (r/sample (make-bag :bag1)))
                         :bag2 (colors (r/sample (make-bag :bag2)))
                         :bag3 (colors (r/sample (make-bag :bag3)))
                         :bagN (colors (r/sample (make-bag :bagN)))}))))

(def predictives (infer :metropolis-hastings predictives-model {:samples 20000
                                                                :thin 3
                                                                :step-scale 0.02}))


(:acceptance-ratio predictives)

(plot/frequencies (trace predictives :bag1))
(plot/frequencies (trace predictives :bag2))
(plot/frequencies (trace predictives :bag3))
(plot/frequencies (trace predictives :bagN))

;; The Blessing of Abstraction

(def colors [:red :blue])
(def colors-length (count colors))
(def colors-map (zipmap colors (range colors-length)))

(defn posterior
  [observed-data]
  (infer :metropolis-hastings (make-model
                               [phi (:dirichlet {:alpha (repeat colors-length 1)})]
                               (let [prototype (v/mult phi colors-length)
                                     dirichlet (distr :dirichlet {:alpha prototype})
                                     bag-probs (memoize (fn [_] (r/sample dirichlet)))
                                     make-bag (memoize (fn [bag]
                                                         (distr :integer-discrete-distribution
                                                                {:data (range colors-length)
                                                                 :probabilities (bag-probs bag)})))]
                                 (model-result (map (fn [datum]
                                                      (observe1 (make-bag (:bag datum)) (colors-map (:draw datum)))) observed-data)
                                               {:bag1 (first (bag-probs :bag1))
                                                :global (first phi)}))) {:samples 50000 :max-iters 1e6
                                                                         :thin 2
                                                                         :initial-point [[0.5 0.5]]
                                                                         :step-scale 0.01}))

(def data [{:bag :bag1 :draw :red} {:bag :bag2 :draw :red} {:bag :bag3 :draw :blue}
           {:bag :bag4 :draw :red} {:bag :bag5 :draw :red} {:bag :bag6 :draw :blue}
           {:bag :bag7 :draw :red} {:bag :bag8 :draw :red} {:bag :bag9 :draw :blue}
           {:bag :bag10 :draw :red} {:bag :bag11 :draw :red} {:bag :bag12 :draw :blue}])

(def data [{:bag :bag1 :draw :red} {:bag :bag1 :draw :red} {:bag :bag1 :draw :blue}
           {:bag :bag1 :draw :red} {:bag :bag1 :draw :red} {:bag :bag1 :draw :blue}
           {:bag :bag1 :draw :red} {:bag :bag1 :draw :red} {:bag :bag1 :draw :blue}
           {:bag :bag1 :draw :red} {:bag :bag1 :draw :red} {:bag :bag1 :draw :blue}])


(def num-obs [1 3 6 9 12])
(def posteriors (doall (pmap #(posterior (take % data)) num-obs)))

(defn mean-dev
  ^double [dist param ^double truth]
  (stats/mean (map (fn [^double v]
                     (m/sq (- v truth))) (trace dist param))))

(def initial-spec (mean-dev (first posteriors) :bag1 0.66))
(def spec-errors (map #(/ (mean-dev % :bag1 0.66) ^double initial-spec) posteriors))

(def initial-glob (mean-dev (first posteriors) :global 0.66))
(def glob-errors (map #(/ (mean-dev % :global 0.66) ^double initial-glob) posteriors))

(plot/line (map vector num-obs spec-errors))
(plot/line (map vector num-obs glob-errors))

;; Learning Overhypotheses: Abstraction at the Superordinate Level

(def colors [:black :blue :green :orange :red])
(def colors-length (count colors))
(def colors-range (range colors-length))
(def colors-map (zipmap colors colors-range))

(def observed-data
  [{:bag  :bag1 :draw  :blue} {:bag  :bag1 :draw  :blue} {:bag  :bag1 :draw  :blue}
   {:bag  :bag1 :draw  :blue} {:bag  :bag1 :draw  :blue} {:bag  :bag1 :draw  :blue}
   {:bag  :bag2 :draw  :green} {:bag  :bag2 :draw  :green} {:bag  :bag2 :draw  :green}
   {:bag  :bag2 :draw  :green} {:bag  :bag2 :draw  :green} {:bag  :bag2 :draw  :green}
   {:bag  :bag3 :draw  :red} {:bag  :bag3 :draw  :red} {:bag  :bag3 :draw  :red}
   {:bag  :bag3 :draw  :red} {:bag  :bag3 :draw  :red} {:bag  :bag3 :draw  :red}
   {:bag  :bag4 :draw  :orange}])

(def observed-data2
  [{:bag  :bag1 :draw :blue} {:bag  :bag1 :draw  :red} {:bag  :bag1 :draw  :green}
   {:bag  :bag1 :draw :black} {:bag  :bag1 :draw  :red} {:bag  :bag1 :draw  :blue}
   {:bag  :bag2 :draw :green} {:bag  :bag2 :draw  :red} {:bag  :bag2 :draw  :black}
   {:bag  :bag2 :draw :black} {:bag  :bag2 :draw  :blue} {:bag  :bag2 :draw  :green}
   {:bag  :bag3 :draw :red} {:bag  :bag3 :draw  :green} {:bag  :bag3 :draw  :blue}
   {:bag  :bag3 :draw :blue} {:bag  :bag3 :draw  :black} {:bag  :bag3 :draw  :green}
   {:bag  :bag4 :draw :orange}])

(defn predictives-model
  [observed-data]
  (make-model
   [phi (:dirichlet {:alpha (repeat colors-length 1)})
    alpha (:gamma)]
   (let [prototype (v/mult phi alpha)
         dirichlet (distr :dirichlet {:alpha prototype})
         make-bag (memoize (fn [bag]
                             (distr :integer-discrete-distribution
                                    {:data colors-range
                                     :probabilities (r/sample dirichlet)})))] 
     (model-result (map (fn [datum]
                          (observe1 (make-bag (:bag datum)) (colors-map (:draw datum)))) observed-data)
                   (fn [] {:bag1 (colors (r/sample (make-bag :bag1)))
                          :bag2 (colors (r/sample (make-bag :bag2)))
                          :bag3 (colors (r/sample (make-bag :bag3)))
                          :bag4 (colors (r/sample (make-bag :bag4)))
                          :bagN (colors (r/sample (make-bag :bagN)))})))))

(def predictives (infer :metropolis-hastings (predictives-model observed-data2) {:samples 30000
                                                                                 :thin 2
                                                                                 :initial-point-search-size 10000
                                                                                 ;; :initial-point [[0.2 0.2 0.2 0.2 0.2] 1.0]
                                                                                 :steps [0.025 0.1]}))


(:acceptance-ratio predictives)

(plot/frequencies (trace predictives :bag1))
(plot/frequencies (trace predictives :bag2))
(plot/frequencies (trace predictives :bag3))
(plot/frequencies (trace predictives :bag4))
(plot/frequencies (trace predictives :bagN))
(plot/histogram (trace predictives :alpha))

;; Example: The Shape Bias

(def attributes [:shape :color :texture :size])
(def values (zipmap attributes (repeat (range 11))))
(def observed-data [{:cat :cat1 :shape 1, :color 1, :texture 1, :size 1},
                    {:cat :cat1 :shape 1, :color 2, :texture 2, :size 2},
                    {:cat :cat2 :shape 2, :color 3, :texture 3, :size 1},
                    {:cat :cat2 :shape 2, :color 4, :texture 4, :size 2},
                    {:cat :cat3 :shape 3, :color 5, :texture 5, :size 1},
                    {:cat :cat3 :shape 3, :color 6, :texture 6, :size 2},
                    {:cat :cat4 :shape 4, :color 7, :texture 7, :size 1},
                    {:cat :cat4 :shape 4, :color 8, :texture 8, :size 2},
                    {:cat :cat5 :shape 5, :color 9, :texture 9, :size 1}])

;; Naive approach which doesn't work this case. We do not catch internal prior and this way we don't produce penalty for wrong sampling
#_(defmodel category-model
    [shape (:dirichlet {:alpha (repeat (count (:shape values)) 1.0)})
     color (:dirichlet {:alpha (repeat (count (:color values)) 1.0)})
     texture (:dirichlet {:alpha (repeat (count (:texture values)) 1.0)})
     size (:dirichlet {:alpha (repeat (count (:size values)) 1.0)})
     shape-e (:exponential)
     color-e (:exponential)
     texture-e (:exponential)
     size-e (:exponential)]
    (let [prototype (memoize (fn [attr]
                               (case attr
                                 :shape (v/mult shape shape-e)
                                 :color (v/mult color color-e)
                                 :texture (v/mult texture texture-e)
                                 :size (v/mult size size-e))))
          make-attr-dist (memoize (fn [_ attr]
                                    (let [d (distr :dirichlet {:alpha (prototype attr)})
                                          probs (sample d)]
                                      (distr :integer-discrete-distribution
                                             {:data (values attr)
                                              :probabilities probs}))))]
      (model-result (mapcat (fn [{:keys [cat shape color texture size]}]
                              [(observe1 (make-attr-dist cat :shape) shape)
                               (observe1 (make-attr-dist cat :color) color)
                               (observe1 (make-attr-dist cat :texture) texture)
                               (observe1 (make-attr-dist cat :size) size)])
                            observed-data)
                    (fn [] {:cat5shape (sample (make-attr-dist :cat5 :shape))
                           :cat5color (sample (make-attr-dist :cat5 :color))
                           :catNshape (sample (make-attr-dist :catN :shape))
                         :catNcolor (sample (make-attr-dist :catN :color))}))))

(map :color (filter #(= :cat1 (:cat %)) observed-data))

(defmodel category-model
  [shape (:dirichlet {:alpha (repeat (count (:shape values)) 1.0)})
   color (:dirichlet {:alpha (repeat (count (:color values)) 1.0)})
   texture (:dirichlet {:alpha (repeat (count (:texture values)) 1.0)})
   size (:dirichlet {:alpha (repeat (count (:size values)) 1.0)})
   shape-e (:exponential)
   color-e (:exponential)
   texture-e (:exponential)
   size-e (:exponential)]
  (let [prototype (memoize (fn [attr]
                             (case attr
                               :shape (v/mult shape shape-e)
                               :color (v/mult color color-e)
                               :texture (v/mult texture texture-e)
                               :size (v/mult size size-e))))
        ;; internal model is necessary to properly train hierarchical model in this case
        ;; higher level model produces prior for internal model(s) 
        make-attr-dist (memoize (fn [c attr]
                                  (let [data (map attr (filter #(= c (:cat %)) observed-data))
                                        internal-model (make-model
                                                        [d (:dirichlet {:alpha (prototype attr)})]
                                                        (let [r (distr :integer-discrete-distribution
                                                                       {:data (values attr)
                                                                        :probabilities d})]
                                                          (model-result [(observe r data)]
                                                                        r)))
                                        inference-result (infer :metropolis-hastings internal-model {:samples 500
                                                                                                     :step-scale 0.0025})]
                                    (best-result inference-result :model-result))))]
    (model-result (->> observed-data
                       (pmap (fn [{:keys [cat shape color texture size]}]
                               [(observe1 (make-attr-dist cat :shape) shape)
                                (observe1 (make-attr-dist cat :color) color)
                                (observe1 (make-attr-dist cat :texture) texture)
                                (observe1 (make-attr-dist cat :size) size)]))
                       (mapcat identity))
                  (fn [] {:cat5shape (sample (make-attr-dist :cat5 :shape))
                         :cat5color (sample (make-attr-dist :cat5 :color))
                         :catNshape (sample (make-attr-dist :catN :shape))
                         :catNcolor (sample (make-attr-dist :catN :color))}))))

(defn gen-steps[d-val e-val]
  (concat (repeat 4 d-val) (repeat 4 e-val)))

(def category-posterior (infer :metropolis-hastings category-model {:samples 5000
                                                                    :burn 50
                                                                    :max-time 180
                                                                    :steps (gen-steps 0.0025 0.15)}))

(:acceptance-ratio category-posterior)

(count (:accepted category-posterior))
(count (distinct (:accepted category-posterior)))
(:steps category-posterior)
(:out-of-prior category-posterior)

(plot/frequencies (trace category-posterior :cat5shape))
(plot/frequencies (trace category-posterior :cat5color))
(plot/frequencies (trace category-posterior :catNshape))
(plot/frequencies (trace category-posterior :catNcolor))
(plot/histogram (map first (trace category-posterior :size)))
(plot/histogram (trace category-posterior :shape-e))
(plot/histogram (trace category-posterior :color-e))

;; Example: X-Bar Theory

(def categories [:D :N :T :V :A :Adv])
(def categories-map (zipmap categories (range (count categories))))

(defn head-to-comp
  [head]
  (case head
    :D :N
    :T :V
    :N :A
    :V :Adv
    :A :none
    :Adv :none
    :error))

(defn make-phrase-dist
  [head-to-phrase-dirs]
  (let [model (make-model
               [] (let [head (rand-nth categories)]
                    (trace-result (if (= :none (head-to-comp head))
                                    [head]
                                    (if (flipb (head-to-phrase-dirs head))
                                      [(head-to-comp head) head]
                                      [head (head-to-comp head)])))))]
    (as-categorical-distribution (infer :forward-sampling model {:samples 500}))))

(def data [[:D :N]])

(defmodel model
  [language-dir (:beta {:alpha 1.0 :beta 1.0})]
  (let [dirichlet (distr :dirichlet {:alpha [language-dir (- 1.0 language-dir)]})
        head-to-phrase-dirs (zipmap categories (repeatedly #(last (r/sample dirichlet))))
        phrase-dist (make-phrase-dist head-to-phrase-dirs)]
    (model-result [(observe phrase-dist data)]
                  (if (flipb (head-to-phrase-dirs :N))
                    "N second"
                    "N first"))))

(def results (infer :metropolis-hastings model {:samples 20000 :step-scale 0.5}))

(:acceptance-ratio results)
(:out-of-prior results)
(:steps results)

(plot/frequencies (trace results :model-result))


