(ns probmods.14-mixture-models
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.vector :as v]
            [fastmath.stats :as stats]
            [inferme.core :refer :all]
            [inferme.plot :as plot]
            [clojure.string :as str]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

(def colors [:blue :green :red])

(def observed-data [{:name :obs1 :draw :red}
                    {:name :obs2 :draw :red}
                    {:name :obs3 :draw :blue}
                    {:name :obs4 :draw :blue}
                    {:name :obs5 :draw :red}
                    {:name :obs6 :draw :blue}])

(defmodel predictives-model
  [phi (:dirichlet {:alpha [1 1 1]})]
  (let [prototype (v/mult phi 0.1)
        make-bag (memoize (fn [bag]
                            (let [dirichlet (distr :dirichlet {:alpha prototype})
                                  color-probs (sample dirichlet)]
                              (distr :categorical-distribution {:data colors :probabilities color-probs}))))
        obs-to-bag (memoize (fn [obs-name]
                              (rand-nth [:bag1 :bag2 :bag3])))]
    (model-result (map #(observe1 (make-bag (obs-to-bag (:name %))) (:draw %)) observed-data)
                  (fn [] {:same-bag-1-and-2 (= (obs-to-bag :obs1) (obs-to-bag :obs2))
                         :same-bag-1-and-3 (= (obs-to-bag :obs1) (obs-to-bag :obs3))}))))

(def post (infer :metropolis-hastings predictives-model {:step-scale 0.01
                                                         :burn 10000
                                                         :thin 5
                                                         :samples 30000}))


(:acceptance-ratio post)

(plot/frequencies (trace post :same-bag-1-and-2))
(plot/frequencies (trace post :same-bag-1-and-3))

;;

(def dirichlet-one (distr :dirichlet {:alpha [1 1 1]}))

(defmodel predictives-model
  [phi (:dirichlet {:alpha [1 1 1]})]
  (let [prototype (v/mult phi 0.1)
        make-bag (memoize (fn [bag]
                            (let [dirichlet (distr :dirichlet {:alpha prototype})
                                  color-probs (sample dirichlet)]
                              (distr :categorical-distribution {:data colors :probabilities color-probs}))))
        obs-to-bag (memoize (fn [obs-name]
                              (let [bag-mixture (sample dirichlet-one)]
                                (sample (distr :categorical-distribution
                                               {:data [:bag1 :bag2 :bag3]
                                                :probabilities bag-mixture})))))]
    (model-result (map #(observe1 (make-bag (obs-to-bag (:name %))) (:draw %)) observed-data)
                  (fn [] {:same-bag-1-and-2 (= (obs-to-bag :obs1) (obs-to-bag :obs2))
                         :same-bag-1-and-3 (= (obs-to-bag :obs1) (obs-to-bag :obs3))}))))

(def post (infer :metropolis-hastings predictives-model {:step-scale 0.01
                                                         :burn 10000
                                                         :thin 5
                                                         :samples 30000}))


(:acceptance-ratio post)

(plot/frequencies (trace post :same-bag-1-and-2))
(plot/frequencies (trace post :same-bag-1-and-3))

;;

;; works bad

(def observed-data  [{:name "a0" :x 1.5343898902525506 :y 2.3460878867298494} {:name "a1" :x 1.1810142951204246 :y 1.4471493362364427} {:name "a2" :x 1.3359476185854833 :y 0.5979097803077312} {:name "a3" :x 1.7461500236610696 :y 0.07441351219375836} {:name "a4" :x 1.1644280209698559 :y 0.5504283671279169} {:name "a5" :x 0.5383179421667954 :y 0.36076578484371535} {:name "a6" :x 1.5884794217838352 :y 1.2379018386693668} {:name "a7" :x 0.633910148716343 :y 1.21804947961078} {:name "a8" :x 1.3591395983859944 :y 1.2056207607743645} {:name "a9" :x 1.5497995798191613 :y 1.555239222467223} 
                     {:name "b0" :x -1.7103539324754713 :y -1.178368516925668} {:name "b1" :x -0.49690324128135566 :y -1.4482931166889297} {:name "b2" :x -1.0191455290951414 :y -0.4103273022785636} {:name "b3" :x -1.6127046244033036 :y -1.198330563419632} {:name "b4" :x -0.8146486481025548 :y -0.33650743701348906} {:name "b5" :x -1.2570582864922166 :y -0.7744102418371701} {:name "b6" :x -1.2635542813354101 :y -0.9202555846522052} {:name "b7" :x -1.3169953429184593 :y -0.40784942495184096} {:name "b8" :x -0.7409787028330914 :y -0.6105091049436135} {:name "b9" :x -0.7683709878962971 :y -1.0457286452094976}])

(defmodel predictives-model
  []
  (let [cat-mixture (r/drand)
        obs-to-cat (memoize (fn [obs-name] (randval cat-mixture :cat1 :cat2)))
        cat-to-mean (memoize (fn [cat] {:xmean (r/grand) :ymean (r/grand)}))]
    (model-result (mapcat (fn [{:keys [x y name]}]
                            (let [mus (cat-to-mean (obs-to-cat name))]
                              [(observe1 (distr :normal {:mu (:xmean mus) :sd 0.01}) x)
                               (observe1 (distr :normal {:mu (:ymean mus) :sd 0.01}) y)])) observed-data)
                  (fn [] (assoc (cat-to-mean (obs-to-cat :new-obs)) :cat-mixture cat-mixture)))))

(def post (infer :metropolis-hastings predictives-model {:max-iters 1e7
                                                         :burn 0
                                                         :thin 100
                                                         :samples 1000}))

(count (:accepted post))
(:acceptance-ratio post)
(count (distinct (trace post :ymean)))
(:out-of-prior post)
(:steps post)

(plot/scatter (map #(v/add % [(r/grand 0.03) (r/grand 0.03)]) ;; jitter a little bit
                   (map vector (trace post :xmean) (trace post :ymean))))

(plot/histogram (trace post :cat-mixture))

;;

(def vocabulary ["DNA" "evolution" "parsing" "phonology"])
(def eta (repeat (count vocabulary) 1.0))
(def dirichlet-alpha (distr :dirichlet {:alpha [1 1]}))

(def corpus
  (concat (repeat 3 (str/split "DNA evolution DNA evolution DNA evolution DNA evolution DNA evolution" #" "))
          (repeat 3 (str/split "parsing phonology parsing phonology parsing phonology parsing phonology parsing phonology" #" "))))

(defmodel predictives-model
  [topic1 (:dirichlet {:alpha eta})
   topic2 (:dirichlet {:alpha eta})]
  (model-result (flatten (map (fn [doc]
                                (let [topic-dist (sample dirichlet-alpha)]
                                  (map (fn [word]
                                         (let [topic (randval (first topic-dist) topic1 topic2)]
                                           (observe1 (distr :categorical-distribution {:data vocabulary
                                                                                       :probabilities topic}) word))) doc))) corpus))))

(def post (infer :metropolis-hastings predictives-model {:step-scale 0.002
                                                         :thin 2
                                                         :samples 20000}))

(:acceptance-ratio post)

(plot/bar (map vector vocabulary (map stats/mean (apply map vector (trace post :topic1)))))
(plot/bar (map vector vocabulary (map stats/mean (apply map vector (trace post :topic2)))))

;; Example: Categorical Perception of Speech Sounds

(def prototype1 0.0)
(def prototype2 5.0)
(def stimuli (range prototype1 prototype2 0.2))

(defn perceived-value
  [stim]
  (-> (infer :metropolis-hastings (make-model
                                   [value1 (:normal {:mu prototype1 :sd 1})
                                    value2 (:normal {:mu prototype2 :sd 1})
                                    category (:bernoulli)]
                                   (let [value (if (pos? category) value1 value2)]
                                     (model-result [(observe1 (distr :normal {:mu value :sd 1}) stim)]
                                                   value)))
             {:samples 10000})
      (trace)
      (stats/mean)))

(plot/scatter (map #(vector %1 (perceived-value %1)) stimuli))

;; Unknown Numbers of Categories

(def observed-data [1 1 1 1 0 0 0 0])
(def observed-data [1 1 1 1 1 1 1 1])

(defmodel model
  []
  (let [coins (randval [:c1] [:c1 :c2])
        coin-to-weight (memoize (fn [c] (r/drand)))]
    (model-result (map #(observe1 (distr :bernoulli {:p (coin-to-weight (rand-nth coins))}) %) observed-data)
                  (count coins))))

(def results (infer :rejection-sampling model {:samples 100}))

(plot/frequencies (trace results))

;;

(def colors [:blue :green :red])
(def observed-marbles [:red :red :blue :blue :red :blue])

(defmodel model
  [phi (:dirichlet {:alpha [1 1 1]})
   poisson (:poisson {:p 1})]
  (let [prototype (v/mult phi 0.1)
        make-bag (memoize (fn [bag]
                            (let [dirichlet (distr :dirichlet {:alpha prototype})]
                              (distr :categorical-distribution {:data colors
                                                                :probabilities (sample dirichlet)}))))
        num-bags (inc poisson)
        bags (mapv #(keyword (str "bag" %)) (range num-bags))]
    (model-result (map (fn [d]
                         (observe1 (make-bag (rand-nth bags)) d)) observed-marbles)
                  num-bags)))

(def results (infer :rejection-sampling model {:samples 100}))

(plot/frequencies (trace results))

;; Infinite mixtures

(defn residuals
  [[^double f & r]]
  (if (seq r) 
    (conj (residuals r)
          (/ f (+ f (stats/sum r))))
    (list 1.0)))

(residuals [0.2, 0.3, 0.1, 0.4])
;; => (0.2 0.37499999999999994 0.2 1.0)

(defn my-sample-discrete
  [resid ^long i]
  (randval (nth resid i 0) i (my-sample-discrete resid (inc i))))

(plot/frequencies (repeatedly 5000 #(my-sample-discrete (residuals [0.2 0.3 0.1 0.4]) 0)))

(def probs (sample (distr :dirichlet {:alpha [1 1 1 1]})))

(plot/frequencies (repeatedly 5000 #(my-sample-discrete (residuals probs) 0)))

;;
(def beta11 (distr :beta {:alpha 1.0 :beta 1.0}))

(def residuals (conj (vec (repeatedly 3 #(sample beta11))) 1.0))

(plot/frequencies (repeatedly 5000 #(my-sample-discrete residuals 0)))

;;

(defn my-sample-discrete
  [resid ^long i]
  (randval (resid i) i (my-sample-discrete resid (inc i))))

(def residuals (memoize (fn [i] (sample beta11))))

(plot/frequencies (repeatedly 5000 #(my-sample-discrete residuals 0)))

;;

(def colors [:blue :green :red])

(def observed-data [{:name :obs1 :draw :red}
                    {:name :obs2 :draw :blue}
                    {:name :obs3 :draw :red}
                    {:name :obs4 :draw :blue}
                    {:name :obs5 :draw :red}
                    {:name :obs6 :draw :blue}])

(defmodel predictives-model
  [phi (:dirichlet {:alpha [1 1 1]})]
  (let [prototype (v/mult phi 0.1)
        make-bag (memoize (fn [bag]
                            (let [dirichlet (distr :dirichlet {:alpha prototype})
                                  color-probs (sample dirichlet)]
                              (distr :categorical-distribution {:data colors :probabilities color-probs}))))
        residuals (memoize (fn [i] (sample beta11)))
        get-bag (memoize (fn [obs-name] (my-sample-discrete residuals 0)))]
    (model-result (map #(observe1 (make-bag (get-bag (:name %))) (:draw %)) observed-data)
                  (fn [] {:same-bag-1-and-2 (= (get-bag :obs1) (get-bag :obs2))
                         :same-bag-1-and-3 (= (get-bag :obs1) (get-bag :obs3))}))))

(def post (infer :metropolis-hastings predictives-model {:step-scale 0.025
                                                         :burn 10000
                                                         :thin 5
                                                         :samples 30000}))


(:acceptance-ratio post)

(plot/frequencies (trace post :same-bag-1-and-2))
(plot/frequencies (trace post :same-bag-1-and-3))
