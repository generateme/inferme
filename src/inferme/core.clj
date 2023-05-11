(ns inferme.core
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.vector :as v]
            [inferme.jump :as jump]
            [inferme.multi]
            [inferme.utils :as utils]
            [clojure.walk :as walk]
            [fastmath.stats :as stats]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

(defrecord ModelResult [ll result])
(defrecord ModelResultFinal [^double ll ^double lp ^clojure.lang.PersistentVector params parameters-map result])

(defmacro model-result
  "Result creator returned by model consisting list of log likelihoods and optional model value."
  ([ll result]
   `(ModelResult. ~ll ~result))
  ([ll]
   `(ModelResult. ~ll nil))
  ([]
   `(ModelResult. nil nil)))

(defmacro trace-result
  "Result creator returned by model consisting only model value. Log likelihood is set to `nil` (everything is probable)"
  ([result]
   `(ModelResult. nil ~result))
  ([]
   `(ModelResult. nil nil)))

(defn safe-sum
  [^double state ^double ll]
  (if (m/invalid-double? ll)
    (reduced ##-Inf)
    (+ state ll)))

;;

#_(defn- ^:private multidimensional? [d] (satisfies? prot/MultivariateDistributionProto d))

#_(defn- emit-let 
    [ppriors model-input]
    (first
     (reduce (fn [[buff ^int id] {:keys [multidimensional? continuous? prior-param-symbol ^int param-cnt]}]
               (let [symbols
                     (cond
                       multidimensional? [prior-param-symbol `(subvec ~model-input ~id ~(+ id param-cnt))]
                       (not continuous?) [prior-param-symbol `(m/floor (.nth ~model-input ~id))]
                       :else [prior-param-symbol `(double (.nth ~model-input ~id))])]
                 [(concat buff symbols) (+ id param-cnt)]))
             [[] (int 0)] ppriors)))

#_(defn- emit-sampling-priors
    [ppriors]
    (let [lst (map (fn [{:keys [prior-distr-symbol]}]
                     `(r/sample ~prior-distr-symbol)) ppriors)]
      (if (some :multidimensional? ppriors)
        `(vec (flatten [~@lst]))
        `[~@lst])))

#_(defn- preprocess-prior
    [[p d]]
    (let [[fixed safe-eval?] (if (and (list? d)
                                      (keyword? (first d)))
                               [(conj d 'r/distribution) false]
                               [d true]) 
          dstr (eval (if (or safe-eval?
                             (not (map? (last fixed)))) fixed (butlast fixed)))
          multi? (multidimensional? dstr)
          ^int dims (if (and multi? (not safe-eval?) (map? (last fixed)))
                      (if-let [ds (:dimensions (last fixed))]
                        ds
                        (r/dimensions (eval fixed)))
                      (r/dimensions dstr))
          distr-id (r/distribution-id dstr)]
      {:distr-id distr-id
       :continuous? (r/continuous? dstr)
       :multidimensional? multi?
       :dimensions dims
       :param-cnt dims
       :prior-param-symbol p
       :prior-distr-symbol (symbol (str "prior-" p))
       :distribution fixed}))

#_(defn- preprocess-priors
    [priors params-sym]
    (let [ppriors (map preprocess-prior (partition 2 priors))]
      {:let (emit-let ppriors params-sym)
       :prior-calc `(+ 0.0 ~@(map (fn [{:keys [prior-distr-symbol prior-param-symbol]}]
                                    `(r/observe1 ~prior-distr-symbol ~prior-param-symbol)) ppriors))
       :parameters-map `(hash-map ~@(mapcat (fn [{:keys [prior-param-symbol]}]
                                              [(keyword prior-param-symbol)
                                               prior-param-symbol]) ppriors))
       :prior-distributions `(vector ~@(map :prior-distr-symbol ppriors))
       :prior-distributions-let (mapcat (juxt :prior-distr-symbol :distribution) ppriors)
       :prior-names `(vector ~@(map (comp keyword :prior-param-symbol) ppriors))
       :sample-priors (emit-sampling-priors ppriors)
       :param-cnt `[~@(map :param-cnt ppriors)]}))

(defn- add-tag
  [s d]
  (if (utils/multi? d)
    (with-meta s {:tag 'clojure.lang.PersistentVector})
    (with-meta s {:tag 'double})))

(defn- make-prior-symbol
  [s]
  (symbol (str "prior-" s)))

(defn- analyze-deps
  "Find if prior do have back reference to any existing prior"
  [params symbols-set]
  (if (seqable? params)
    (mapcat (fn [v]
              (cond
                (seqable? v) (analyze-deps v symbols-set)
                (symbol? v) [(symbols-set v)]
                :else [false])) params)
    [(symbol? params)]))

(defn- maybe-fix-deps
  [d symbols-set deps?]
  (if-not deps?
    d
    (let [[params & r] (reverse d)]
      (->> (walk/postwalk (fn [v]
                            (if (symbols-set v)
                              `(r/sample ~(make-prior-symbol v))
                              v)) params)
           (conj r)
           (reverse)))))

(defn- preprocess-prior
  [s d symbols-set]
  (let [deps? (some identity (analyze-deps (last d) symbols-set))
        [s' d'] (if (list? d)
                  (let [[a b] d]
                    (cond
                      (keyword? a) [(add-tag s a) (conj d `r/distribution)] ;; (:normal ...)
                      (and (= (name a) "distribution")
                           (keyword? b)) [(add-tag s b) d] ;; (r/distribution :normal ...)
                      :else [s d]))
                  [s d])]
    {:symbol s'
     :prior-symbol (make-prior-symbol s')
     :distribution-sampled-deps (maybe-fix-deps d' symbols-set deps?)
     :distribution d'
     :deps? deps?}))

(defn- preprocess-priors
  [priors params-sym]
  (let [ppriors (first (reduce (fn [[curr symbols] [s d]]
                                 [(conj curr (preprocess-prior s d symbols))
                                  (conj symbols s)]) [[] #{}] (partition 2 priors)))
        distrs (map :distribution-sampled-deps ppriors)
        param-names (map :symbol ppriors)
        param-names-k (map keyword param-names)
        distr-names (map :prior-symbol ppriors)]
    {:parameters-let (mapcat (fn [idx p] [p `(.nth ~params-sym ~idx)]) (range) param-names)
     :parameter-names param-names
     :parameter-names-k param-names-k
     :parameters-map (zipmap param-names-k param-names)
     :sample-priors-let `[~@(interleave param-names (map (fn [{:keys [prior-symbol distribution deps?]}]
                                                           (if-not deps?
                                                             `(r/sample ~prior-symbol)
                                                             `(r/sample ~distribution))) ppriors))]
     :distributions-let `[~@(interleave distr-names distrs)]
     :distributions-refresh-let `[~@(mapcat (juxt :prior-symbol :distribution) (filter :deps? ppriors))]
     :distribution-names distr-names
     :calc-priors `(+ 0.0 ~@(map (fn [d p]
                                   `(r/lpdf ~d ~p)) distr-names param-names))}))

#_(:sample-priors-let (preprocess-priors '[a (:dirichlet)
                                           b (:exponential)
                                           c (:normal {:mu b})] 'params))






(defmacro make-model
  "Create model."
  [priors & r]
  (assert (vector? priors) "Priors should be a vector!")
  (assert (even? (count priors)) "Odd number of elements in priors.")
  (let [params (with-meta (symbol "params") {:tag 'clojure.lang.PersistentVector})
        preprocessed-priors (preprocess-priors priors params)]
    `(let [~@(:distributions-let preprocessed-priors)
           ~'distributions [~@(:distribution-names preprocessed-priors)]]
       {:distributions ~'distributions
        :distribution-dims (map r/dimensions ~'distributions)
        :distribution-names (map r/distribution-id ~'distributions)
        :parameter-names [~@(:parameter-names-k preprocessed-priors)]
        :model (fn local-model#
                 ([] (let [~@(:sample-priors-let preprocessed-priors)]
                       [~@(:parameter-names preprocessed-priors)]))
                 ([~params] (local-model# ~params true))
                 ([~params with-prior?#]
                  (let [~@(:parameters-let preprocessed-priors)
                        ~@(:distributions-refresh-let preprocessed-priors)
                        ~'prior (if with-prior?# ~(:calc-priors preprocessed-priors) 0.0)]
                    (when-not (and with-prior?# (m/invalid-double? ~'prior)) ;; if priors are invalid, return nil and avoid further computation
                      (let [~'parameters-map ~(:parameters-map preprocessed-priors)
                            ^ModelResult mr# (or (do ~@r) (model-result))
                            llsum# (double (reduce safe-sum 0.0 (.ll mr#)))
                            lpsum# (+ llsum# ~'prior)]
                        (ModelResultFinal. llsum#
                                           lpsum#
                                           ~params
                                           ~'parameters-map
                                           (.result mr#)))))))})))


#_(defmacro make-model
    "Create model."
    [priors & r]
    (assert (vector? priors) "Priors should be a vector!")
    (assert (even? (count priors)) "Odd number of elements in priors.")
    (let [params (with-meta (symbol "params") {:tag 'clojure.lang.PersistentVector})
          preprocessed-priors (preprocess-priors priors params)]
      `(let [~@(:prior-distributions-let preprocessed-priors)
             ~'prior-distributions ~(:prior-distributions preprocessed-priors)
             ~'prior-names ~(:prior-names preprocessed-priors)]
         {:parameter-names ~'prior-names
          :prior-distributions ~'prior-distributions
          :parameter-count ~(:param-cnt preprocessed-priors)
          :model (fn local-model#
                   ([] ~(:sample-priors preprocessed-priors))
                   ([~params] (local-model# ~params true))
                   ([~params with-prior?#]
                    (let [~@(:let preprocessed-priors)
                          ~'prior (double (if with-prior?# ~(:prior-calc preprocessed-priors) 0.0))]
                      (when-not (and with-prior?# (m/invalid-double? ~'prior)) ;; if priors are invalid, return nil and avoid further computation
                        (let [~'parameters-map ~(:parameters-map preprocessed-priors)
                              ^ModelResult mr# (or (do ~@r) (model-result))
                              llsum# (double (reduce safe-sum 0.0 (.ll mr#)))
                              lpsum# (+ llsum# ~'prior)]
                          (ModelResultFinal. llsum#
                                             lpsum#
                                             ~params
                                             ~'parameters-map
                                             (.result mr#)))))))})))


(defmacro defmodel
  "Create and define model."
  [nm priors & r]
  `(def ~nm (make-model ~priors ~@r)))

;;;;;;;;;;;

(defmulti infer (fn [k & _] k))

(defn- recursive-res
  [res]
  (cond
    (fn? res) (recur (res))
    (map? res) res
    :else {:model-result res}))

(defn- process-result
  [^ModelResultFinal res]
  (assoc
   (merge (.parameters-map res)
          (recursive-res (.result res)))
   :LL (.ll res) :LP (.lp res)))

(defn- finalizer
  [^long max-iters ^long samples ^double max-time]
  (let [start-time (System/currentTimeMillis)]
    (fn [a ^long iter]
      (let [reason (cond
                     (>= iter max-iters) :max-iters
                     (>= (count a) samples) :samples
                     (> (/ (- (System/currentTimeMillis) start-time) 1000.0) max-time) :max-time
                     :else false)]
        (when reason
          (assoc {:accepted (persistent! a)} :stop-reason reason))))))

(defmethod infer :rejection-sampling
  ([_ model] (infer :rejection-sampling model {}))
  ([_ {:keys [model]} {:keys [^int max-iters ^int samples ^double max-time ^double log-bound]
                       :or {max-iters 100000 samples 1000 max-time 30.0 log-bound 0.0}}]
   (let [finalize? (finalizer max-iters samples max-time)] 
     (loop [iter (int 0)
            accepted  (transient [])
            accepted-cnt (int 0)]
       (if-let [finalized (finalize? accepted iter)]
         (assoc finalized
                :acceptance-ratio (/ accepted-cnt (double iter)))
         (let [params (model)
               ^ModelResultFinal model-res (model params false)
               result (process-result model-res)
               diff (- (.ll model-res) log-bound)]
           (if (and (m/valid-double? diff)
                    (or (>= diff 0.0) (< (m/log (r/drand)) diff)))
             (recur (inc iter) (conj! accepted result) (inc accepted-cnt))
             (recur (inc iter) accepted accepted-cnt))))))))

(defmethod infer :forward-sampling
  ([_ model] (infer :forward-sampling model {}))
  ([_ {:keys [model]} {:keys [^int max-iters ^int samples ^double max-time]
                       :or {max-iters 100000 samples 1000 max-time 30.0}}]
   (let [finalize? (finalizer max-iters samples max-time)] 
     (loop [iter (int 0)
            accepted (transient [])
            accepted-cnt (int 0)]
       (if-let [finalized (finalize? accepted iter)]
         (assoc finalized
                :acceptance-ratio (/ accepted-cnt (double iter)))
         (let [params (model)
               ^ModelResultFinal model-res (model params false)
               result (process-result model-res)]
           (if-not (m/invalid-double? (.ll model-res))
             (recur (inc iter) (conj! accepted result) (inc accepted-cnt))
             (recur (inc iter) accepted accepted-cnt))))))))

;;
(declare best-initial-point)

(defn initial-point-calc
  ([model initial-point] (initial-point-calc model initial-point 50))
  ([model initial-point search-size]
   (let [m (:model model)]
     (if-let [res (and initial-point (m initial-point))]
       res
       (let [inferred (infer :forward-sampling model {:samples search-size})]
         (if (pos? (count (:accepted inferred)))
           (m (best-initial-point model inferred))
           (throw (Exception. "Can't find valid initial point. Check priors, model conditions or provide own initial point."))))))))

(defmethod infer :metropolis-hastings
  ([_ model] (infer :metropolis-hastings model {}))
  ([_ model {:keys [^int max-iters ^int samples ^double max-time ^int burn
                    initial-point ^long initial-point-search-size steps step-scale ^int thin kernel]
             :or {max-iters 100000 samples 10000 max-time 30.0 burn 500
                  initial-point-search-size 50
                  thin 1 kernel (jump/regular-kernel (r/distribution :normal))}}]
   (let [finalize? (finalizer max-iters samples max-time)
         [step-vals step-fns] (kernel model steps step-scale)
         m (:model model)]
     (loop [iter (int 0)
            accepted (transient [])
            accepted-cnt (int 0)
            out-of-prior (int 0)
            ^ModelResultFinal mr (initial-point-calc model initial-point initial-point-search-size)]
       (if-let [finalized (finalize? accepted iter)]
         (assoc finalized
                :acceptance-ratio (/ accepted-cnt (double iter))
                :out-of-prior out-of-prior
                :steps step-vals)
         
         (if-let [^ModelResultFinal new-mr (m (mapv (fn [k v] (k v)) step-fns (.params mr)))] ;; make step and call model
           (let [reject? (or (m/invalid-double? (.lp new-mr)) ;; reject condition
                             (let [diff (- (.lp new-mr) (.lp mr))]
                               (and (neg? diff) (> (m/log (r/drand)) diff))))
                 new-accepted (if (and (>= iter burn) (zero? (mod iter thin))) ;; burn-in/thinning
                                (conj! accepted (if reject? (process-result mr) (process-result new-mr)))
                                accepted)]
             
             (if reject? ;; interate for next step
               (recur (inc iter) new-accepted accepted-cnt out-of-prior mr)
               (recur (inc iter) new-accepted (inc accepted-cnt) out-of-prior new-mr)))
           
           (let [new-accepted (if (and (>= iter burn) (zero? (mod iter thin))) ;; if prior fail, store current point and interate
                                (conj! accepted (process-result mr))
                                accepted)]
             (recur (inc iter) new-accepted accepted-cnt (inc out-of-prior) mr))))))))

(defmethod infer :metropolis-within-gibbs
  ([_ model] (infer :metropolis-within-gibbs model {}))
  ([_ model {:keys [^int max-iters ^int samples ^double max-time ^int burn
                    initial-point ^long initial-point-search-size steps step-scale ^int thin kernel]
             :or {max-iters 100000 samples 10000 max-time 30.0 burn 500
                  initial-point-search-size 50
                  thin 1 kernel (jump/regular-kernel (r/distribution :normal))}}]
   (let [finalize? (finalizer max-iters samples max-time)
         [step-vals step-fns] (kernel model steps step-scale)
         ^ModelResultFinal init-mr (initial-point-calc model initial-point initial-point-search-size)
         m (:model model)
         param-cnt (count (.params init-mr))]
     (loop [iter (int 0)
            accepted (transient [])
            accepted-cnt (int 0)
            ^ModelResultFinal mr init-mr]
       (if-let [finalized (finalize? accepted iter)] 
         (assoc finalized
                :acceptance-ratio (/ accepted-cnt (* param-cnt (double iter)))
                :steps step-vals)
         (let [^clojure.lang.PersistentVector new-params (mapv (fn [k v] (k v)) step-fns (.params mr))
               [^int inner-accepted
                ^ModelResultFinal curr-mr] (loop [inner-accepted (int 0)
                                                  positions (shuffle (range param-cnt)) ;; iterate through parameters and update them one by one
                                                  ^ModelResultFinal curr-mr mr]
                                             (if-not (seq positions)
                                               [inner-accepted curr-mr]
                                               (let [pos (first positions)
                                                     ^ModelResultFinal new-mr (m (assoc (.params curr-mr) pos
                                                                                        (.nth new-params pos)))]
                                                 (if-not new-mr
                                                   (recur inner-accepted (rest positions) curr-mr)
                                                   (if (and (m/valid-double? (.lp new-mr))
                                                            (let [diff (- (.lp new-mr) (.lp curr-mr))]
                                                              (or (>= diff 0.0)
                                                                  (< (m/log (r/drand)) diff))))
                                                     (recur (inc inner-accepted) (rest positions) new-mr)
                                                     (recur inner-accepted (rest positions) curr-mr))))))
               new-accepted (if (and (>= iter burn) (zero? (mod iter thin)))
                              (conj! accepted (process-result curr-mr))
                              accepted)]

           (recur (inc iter) new-accepted (+ accepted-cnt inner-accepted) curr-mr)))))))

#_(def data (repeatedly 1000 #(r/grand 2 0.2)))

#_(defmodel tmodel
    [mu (:normal {:sd 10})
     sd (:uniform-real {:lower 0.0001})]
    (model-result [(observe (distr :normal {:mu mu :sd sd}) data)]))

#_(defn partial-derivative
    ([model params] (partial-derivative model params 1.0e-6))
    ([model ^clojure.lang.PersistentVector params ^double interval]
     (let [^double lp (:lp (model params))]
       #_     (println {:params params :lp lp})
       (mapv (fn [^long id]
               (let [new-params (assoc params id (+ ^double (.nth params id) interval))]
                 #_      (println {:new-params new-params :lp (:lp (model new-params))
                                   :diff (- ^double (:lp (model new-params)) lp)})
                 (/ (- ^double (:lp (model new-params)) lp) interval)))
             (range (count params))))))

#_(let [model tmodel
        m (:model model)
        steps nil
        step-scale nil
        kernel (jump/regular-kernel (r/distribution :normal))
        [step-vals step-fns] (kernel model steps step-scale)
        ^ModelResultFinal init-mr (initial-point-calc model [1.0 0.2]  10)
        Mo0 init-mr
        parm (.params init-mr)
        epsilon 1.0e-3
        gr0 (partial-derivative (:model tmodel) parm)
        mr init-mr
        prop parm
        momentum0 (mapv (fn [k v] (k v)) step-fns [0.0 0.0])
        kinetic0 (* 0.5 (v/dot momentum0 momentum0))
        momentum1 (v/add momentum0 (v/mult gr0 (* 0.5 epsilon)))
        Mo01 Mo0
        prop (v/add prop (v/mult momentum1 epsilon))
        ^ModelResultFinal Mo1 (m prop)
        gr1 (partial-derivative (:model tmodel) (.params Mo1))
        momentum1 (v/sub (v/add momentum1 (v/mult gr1 (* 0.5 epsilon))))
        kinetic1 (* 0.5 (v/dot momentum1 momentum1))
        H0 (+ kinetic0 (- (.lp Mo0)))
        H1 (+ kinetic1 (- (.lp Mo1)))
        delta (- H1 H0)
        alpha (min 1.0 (m/exp (- delta)))]
    [momentum0 momentum1 gr0 gr1 (.params init-mr) (.params Mo1) kinetic0 kinetic1 alpha])


#_(call tmodel [0.0 0.01])

;; (def res (infer :metropolis-hastings tmodel {:steps [0.01 0.01]}))

;; (:acceptance-ratio res)
;; (:steps res)

;; (plot/histogram (trace res :sd))
;; (plot/histogram (trace res :mu))


;; (call tmodel)

;;

#_(defmethod infer :hamiltonian-monte-carlo
    ([_ model] (infer :hamiltonian-monte-carlo model {}))
    ([_ model {:keys [^int max-iters ^int samples ^double max-time ^int burn
                      initial-point ^long initial-point-search-size steps step-scale ^int thin kernel]
               :or {max-iters 100000 samples 10000 max-time 30.0 burn 500
                    initial-point-search-size 50
                    thin 1 kernel (jump/regular-kernel (r/distribution :normal))}}]
     (let [finalize? (finalizer max-iters samples max-time)
           [step-vals step-fns] (kernel model steps step-scale)
           ^ModelResultFinal init-mr (initial-point-calc model initial-point initial-point-search-size)
           m (:model model)
           param-cnt (count (.params init-mr))]
       (loop [iter (int 0)
              accepted (transient [])
              accepted-cnt (int 0)
              ^ModelResultFinal mr init-mr]
         (if-let [finalized (finalize? accepted iter)] 
           (assoc finalized
                  :acceptance-ratio (/ accepted-cnt (* param-cnt (double iter)))
                  :steps step-vals)
           (let [^clojure.lang.PersistentVector new-params (mapv (fn [k v] (k v)) step-fns (.params mr))
                 [^int inner-accepted
                  ^ModelResultFinal curr-mr] (loop [inner-accepted (int 0)
                                                    positions (shuffle (range param-cnt)) ;; iterate through parameters and update them one by one
                                                    ^ModelResultFinal curr-mr mr]
                                               (if-not (seq positions)
                                                 [inner-accepted curr-mr]
                                                 (let [pos (first positions)
                                                       ^ModelResultFinal new-mr (m (assoc (.params curr-mr) pos
                                                                                          (.nth new-params pos)))]
                                                   (if-not new-mr
                                                     (recur inner-accepted (rest positions) curr-mr)
                                                     (if (and (m/valid-double? (.lp new-mr))
                                                              (let [diff (- (.lp new-mr) (.lp curr-mr))]
                                                                (or (>= diff 0.0)
                                                                    (< (m/log (r/drand)) diff))))
                                                       (recur (inc inner-accepted) (rest positions) new-mr)
                                                       (recur inner-accepted (rest positions) curr-mr))))))
                 new-accepted (if (and (>= iter burn) (zero? (mod iter thin)))
                                (conj! accepted (process-result curr-mr))
                                accepted)]

             (recur (inc iter) new-accepted (+ accepted-cnt inner-accepted) curr-mr)))))))

;;

(defmethod infer :elliptical-slice-sampling
  ([_ model] (infer :elliptical-slice-sampling model {}))
  ([_ model {:keys [^int samples ^double max-time ^int burn
                    initial-point-search-size
                    initial-point covariances ^int thin]
             :or {samples 10000 max-time 30.0 burn 500 thin 1 initial-point-search-size 50}}]
   (let [finalize? (finalizer (inc (+ (* thin samples) burn)) samples max-time)
         ^ModelResultFinal init-mr (initial-point-calc model initial-point initial-point-search-size)
         param-cnt (count (.params init-mr))
         m (:model model)
         proposal (r/distribution :multi-normal {:means (repeat param-cnt 0.0)
                                                 :covariances covariances})]
     (loop [iter (int 0)
            accepted (transient [])
            ^ModelResultFinal mr init-mr]
       (if-let [finalized (finalize? accepted iter)] 
         (assoc finalized :acceptance-ratio 1.0)
         (let [nu (r/sample proposal)
               log-u (m/log (r/drand))
               init-theta (r/drand m/TWO_PI)
               ^ModelResultFinal curr-mr (loop [theta init-theta
                                                theta-min (- init-theta m/TWO_PI)
                                                theta-max init-theta]
                                           (let [prop (v/add (v/mult (.params mr) (m/cos theta))
                                                             (v/mult nu (m/sin theta)))
                                                 ^ModelResultFinal new-mr (m prop)]
                                             (if (and new-mr (m/valid-double? (.lp new-mr)))
                                               (if (or (< (m/abs theta) 1.0e-6)
                                                       (< log-u (- (.lp new-mr) (.lp mr))))
                                                 new-mr
                                                 (let [theta-min (if (neg? theta) theta theta-min)
                                                       theta-max (if (pos? theta) theta theta-max)]
                                                   (recur (r/drand theta-min theta-max) theta-min theta-max)))
                                               mr)))]
           
           (recur (inc iter)
                  (if (and (>= iter burn) (zero? (mod iter thin)))
                    (conj! accepted (process-result curr-mr))
                    accepted)
                  curr-mr)))))))

;;

(defmacro multi
  ([distribution dimensions]
   `(multi ~distribution ~dimensions nil))
  ([distribution dimensions params]
   `(r/distribution :multi {:dims ~dimensions
                            :distribution ~distribution
                            :parameters ~params})))

(defmacro multi*
  ([distribution dimensions]
   `(multi ~distribution ~dimensions nil))
  ([distribution dimensions params]
   `(r/distribution :multi {:dims ~dimensions
                            :distribution ~distribution
                            :parameters ~params
                            :multiple-parameters? true})))


(defmacro and+
  ([] `1)
  ([a] `~a)
  ([a & r] `(if (zero? (* ~a ~@r)) 0 1)))

(defmacro or+
  ([] `0)
  ([a] `~a)
  ([a & r] `(if (zero? (+ ~a ~@r)) 0 1)))

(defmacro distr [& r] `(r/distribution ~@r))
(defmacro flip [& r] `(r/flip ~@r))
(defmacro flipb [& r] `(r/flipb ~@r))
(defmacro sample
  ([distribution] `(r/sample ~distribution))
  ([n distribution] `(r/->seq ~distribution ~n)))
(defmacro randval [& r] `(r/randval ~@r))

(defmacro condition
  ([predicate]
   `(if ~predicate 0.0 ##-Inf))
  ([predicate true-value false-value]
   `(if ~predicate ~true-value ~false-value)))

(defmacro observe1
  [distr v]
  `(r/lpdf ~distr ~v))

(defmacro score
  [distr v]
  `(r/lpdf ~distr ~v))

(defmacro observe
  [distr data]
  `(r/log-likelihood ~distr ~data))

(defn random-priors
  ([model] (random-priors model false))
  ([model as-map?]
   (let [ps ((:model model))]
     (if as-map?
       (zipmap (:parameter-names model) ps)
       ps))))

(defn- extract-parameters
  [model parameters]
  ((apply juxt (:parameter-names model)) (merge (random-priors model true) parameters)))

(defn call
  ([model] (call model ((:model model))))
  ([model parameters] (call model parameters true))
  ([model parameters priors?]
   (let [^ModelResultFinal mr ((:model model) (if (map? parameters)
                                                (extract-parameters model parameters)
                                                parameters) priors?)]
     (process-result mr))))

(defn best-result
  ([inference-result]
   (first (sort-by :LP clojure.core/> (:accepted inference-result))))
  ([inference-result selector]
   ((best-result inference-result) selector)))

(defn best-initial-point
  ([model inference-result]
   (when-let [pnames (seq (:parameter-names model))] 
     (vec ((apply juxt pnames) (best-result inference-result)))))
  ([model]
   (best-initial-point model (infer :forward-sampling model))))

(defmacro ^:private as-distribution
  [distribution]
  (let [s (symbol (str "as-" (name distribution)))
        i (symbol "inferred")
        f (symbol "field")]
    `(defn ~s
       ([~i ~f]
        (r/distribution ~distribution {:data (map ~f (:accepted ~i))}))
       ([~i]
        (~s ~i :model-result)))))

(as-distribution :integer-discrete-distribution)
(as-distribution :real-discrete-distribution)
(as-distribution :continuous-distribution)
(as-distribution :categorical-distribution)

(defn trace
  ([inferred]
   (trace inferred :model-result))
  ([inferred selector]
   (map #(get % selector) (:accepted inferred)))
  ([inferred selector pos]
   (map #(nth (get % selector) pos) (:accepted inferred))))

(defn traces
  [inferred & r]
  (map (apply juxt (map #(fn [v]
                           (get v %)) r)) (:accepted inferred)))

;; stats

(defn stats
  [inferred & trace-params]
  (let [t (-> (apply trace inferred trace-params)
              (m/seq->double-array))
        p (stats/percentiles t [1.0 2.5 5.0
                                25.0 50.0 75.0
                                95.0 97.5 99.0])]
    {:hdi-94% (butlast (stats/hpdi-extent t 0.94))
     :size (alength t)
     :mean (stats/mean t)
     :stddev (stats/stddev t)
     :mode (stats/mode t)
     :median (p 4)
     :percentiles (zipmap [1 2.5 5 :Q1 :median :Q3 95 97.5 99] p)
     :min (smile.math.MathEx/min t)
     :max (smile.math.MathEx/max t)
     :lag (let [acf (stats/acf-ci t 50)]
            (->> (map (fn [id ^double a ^double c]
                        [id (<= a c)]) (range) (:acf acf) (:cis acf))
                 (filter second)
                 (ffirst)))}))
