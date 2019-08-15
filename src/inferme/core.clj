(ns inferme.core
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

(defrecord ModelResult [ll result ^double LL])

(defmacro model-result
  ([ll result]
   `(->ModelResult ~ll ~result 0.0))
  ([ll]
   `(->ModelResult ~ll nil 0.0))
  ([]
   `(->ModelResult nil nil 0.0)))

(defmacro trace-result
  ([result]
   `(->ModelResult nil ~result 0.0))
  ([]
   `(->ModelResult nil nil 0.0)))

(defn- safe-sum
  [^double state ^double ll]
  (if (m/invalid-double? ll)
    (reduced ##-Inf)
    (+ state ll)))

(defn log-likelihood-sum
  ^double [^ModelResult result]
  (reduce safe-sum 0.0 (.ll result)))

(defn prior-sum
  ^double [params priors]
  (reduce safe-sum 0.0 (mapv #(r/lpdf %1 %2) priors params)))

(defn unnormalized-posterior-sum
  ^double [result params priors]
  (+ (log-likelihood-sum result)
     (prior-sum params priors)))

(def ^:private is-discrete? (memoize (fn [nm] (not (r/continuous? (r/distribution nm))))))
(def ^:private is-multidimensional? (memoize (fn [nm] (satisfies? r/MultivariateDistributionProto (r/distribution nm)))))

(defmacro make-model 
  [priors & r]
  (assert (vector? priors) "Priors should be a vector!")
  (assert (even? (count priors)) "Odd number of elements in priors.")
  (let [symbols (take-nth 2 priors) 
        ks (map keyword symbols)
        distr (map #(if (keyword? (first %))
                      (conj % 'r/distribution)
                      %) (take-nth 2 (rest priors)))
        params (with-meta (symbol "params") {:tag 'clojure.lang.PersistentVector})
        priords (symbol "prior-distributions")
        priorns (symbol "prior-names")
        parameters-map (symbol "parameters-map")] 
    `(let [~priords (list ~@distr)
           ~priorns (list ~@ks)]
       {:parameter-names ~priorns
        :prior-distributions ~priords
        :model (fn local-model#
                 ([] (mapv r/sample ~priords))
                 ([~params]
                  (let [~@(interleave symbols (map-indexed (fn [id [_ k]]
                                                             (cond
                                                               (is-multidimensional? k) `(.nth ~params ~id)
                                                               (is-discrete? k) `(m/floor (.nth ~params ~id)) 
                                                               :else `(double (.nth ~params ~id)))) distr))]
                    (let [~parameters-map (hash-map ~@(interleave ks symbols))
                          ^ModelResult mr# (or (do ~@r) (model-result))
                          llsum# (log-likelihood-sum mr#)]
                      (->ModelResult (.ll mr#)
                                     (assoc (if-let [r# (.result mr#)]
                                              (if (map? r#)
                                                (merge r# ~parameters-map)
                                                (assoc ~parameters-map :model-result r#))
                                              ~parameters-map) :LL llsum#)
                                     llsum#)))))})))

(defmacro defmodel
  [nm priors & r]
  `(def ~nm (make-model ~priors ~@r)))

(defmulti infer (fn [k & r] k))

(defn- finalizer
  [^long max-iters ^long samples ^double max-time]
  (let [start-time (System/currentTimeMillis)]
    (fn [a ^long iter]
      (let [reason (cond
                     (== iter max-iters) :max-iters
                     (== samples (count a)) :samples
                     (> (/ (- (System/currentTimeMillis) start-time) 1000.0) max-time) :max-time
                     :else false)]
        (when reason
          (assoc {:accepted (persistent! a)} :stop-reason reason))))))

(defmethod infer :rejection-sampling
  ([_ model] (infer :rejection-sampling model {}))
  ([_ {:keys [parameter-names prior-distributions model]} {:keys [^long max-iters ^long samples ^double max-time ^double log-bound]
                                                           :or {max-iters 100000 samples 1000 max-time 30.0 log-bound 0.0}}]
   (let [finalize? (finalizer max-iters samples max-time)
         accepted  (transient [])] 
     (loop [iter (long 0)]
       (if-let [finalized (finalize? accepted iter)]
         finalized
         (let [params (model)
               ^ModelResult model-res (model params)
               result (.result model-res)
               diff (- (.LL model-res) log-bound)]
           (when (and (m/valid-double? diff)
                      (or (>= diff 0.0) (< (m/log (r/drand)) diff)))
             (conj! accepted result))
           (recur (inc iter))))))))

(defmethod infer :forward-sampling
  ([_ model] (infer :forward-sampling model {}))
  ([_ {:keys [parameter-names prior-distributions model]} {:keys [^long max-iters ^long samples ^double max-time ^double log-bound]
                                                           :or {max-iters 100000 samples 1000 max-time 10.0}}]
   (let [finalize? (finalizer max-iters samples max-time)
         accepted (transient [])] 
     (loop [iter (long 0)]
       (if-let [finalized (finalize? accepted iter)]
         finalized
         (let [params (model)
               ^ModelResult model-res (model params)
               result (.result model-res)]
           (when-not (some m/invalid-double? (.ll model-res))
             (conj! accepted result))
           (recur (inc iter))))))))

(defn initial-point-calc
  ([initial-point model priors pnames] (initial-point-calc initial-point model priors pnames 10000))
  ([initial-point model priors pnames iter]
   (let [initial-point (or initial-point (model))
         ^ModelResult mr (model initial-point)
         s (unnormalized-posterior-sum mr initial-point priors)] 
     (if (m/valid-double? s)
       [initial-point s (.result mr)]
       (if (pos? ^long iter)
         (recur (model) model priors pnames (dec ^long iter))
         (throw (Exception. "Can't find valid initial point. Check priors.")))))))

(defn- stddev-without-outliers
  ^double [data]
  (let [data (m/seq->double-array data)
        q1 (stats/percentile data 25)
        q3 (stats/percentile data 75)
        iqr15 (* 1.5 (- q3 q1))
        l (- q1 iqr15)
        u (+ q3 iqr15)]
    (stats/stddev (filter (fn [^double v] (< l v u)) data))))

(defn gaussian-step-calc
  [steps ^double step-scale priors]
  (if (and steps (== (count steps) (count priors)))
    (vec steps)
    (mapv #(if (satisfies? r/MultivariateDistributionProto %)
             (let [vs (map-indexed (fn [id r] (m/sqrt (get r id))) (r/covariance %))]
               (mapv (fn [^double v] (* step-scale v)) vs))
             (let [v (r/variance %)]
               (* step-scale ^double (if (m/valid-double? v)
                                       (m/sqrt v)
                                       (stddev-without-outliers (r/->seq % 5000)))))) priors)))
(defn gaussian-next-step-fn
  [steps]
  (let [fns (mapv (fn [step]
                    (if (vector? step)
                      (fn [means]
                        (mapv #(r/grand %1 %2) means step))
                      (fn [mean]
                        (r/grand mean step)))) steps)]
    (fn [params]
      (mapv #(%1 %2) fns params))))

(defmethod infer :metropolis-hastings
  ([_ model] (infer :metropolis-hastings model {}))
  ([_ {:keys [parameter-names prior-distributions model]}
    {:keys [^long max-iters ^long samples ^double max-time ^long burn
            initial-point steps ^double step-scale ^long thin]
     :or {max-iters 100000 samples 10000 max-time 30.0 burn 500
          step-scale 0.7 thin 1}}]
   (let [finalize? (finalizer max-iters samples max-time)
         accepted (transient [])
         [init-params init-lp init-result] (initial-point-calc initial-point model prior-distributions parameter-names)
         step-vals (gaussian-step-calc steps step-scale prior-distributions)
         step-fn (gaussian-next-step-fn step-vals)]
     (loop [iter (long 0)
            accepted-cnt (long 0)
            out-of-prior (long 0)
            params init-params
            ^double lp init-lp
            result init-result]
       (if-let [finalized (finalize? accepted iter)] 
         (assoc finalized
                :acceptance-ratio (/ accepted-cnt (double iter))
                :out-of-prior out-of-prior
                :steps step-vals)
         (let [new-params (step-fn params)
               priors-lp (prior-sum new-params prior-distributions)]
           (if (m/valid-double? priors-lp)
             (let [^ModelResult model-res (model new-params)
                   new-result (.result model-res)
                   new-lp (+ (.LL model-res) priors-lp)
                   reject? (or (m/invalid-double? new-lp) ;; reject 
                               (let [diff (- new-lp lp)]
                                 (and (neg? diff) (> (m/log (r/drand)) diff))))]

               ;; store results
               (when (and (>= iter burn) (zero? (mod iter thin)))
                 (conj! accepted (if reject? result new-result)))
               
               (if reject?
                 (recur (inc iter) accepted-cnt out-of-prior params lp result)
                 (recur (inc iter) (inc accepted-cnt) out-of-prior new-params new-lp new-result)))
             
             (do (when (and (>= iter burn) (zero? (mod iter thin)))
                   (conj! accepted result))
                 (recur (inc iter) accepted-cnt (inc out-of-prior) params lp result)))))))))

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
(defmacro sample [& r] `(r/sample ~@r))
(defmacro randval [& r] `(r/randval ~@r))

(defmacro condition
  ([predicate]
   `(if ~predicate 0.0 ##-Inf))
  ([predicate true-value false-value]
   `(if ~predicate ~true-value ~false-value)))

(defmacro observe1
  [distr v]
  `(r/lpdf ~distr ~v))

(defmacro observe
  [distr data]
  `(r/log-likelihood ~distr ~data))

(defmacro call
  ([model] `((:model ~model)))
  ([model parameters]
   `(let [^ModelResult mr# ((:model ~model) ~parameters)]
      (assoc (.result mr#)
             :LL (:LL mr#)))))

(defmacro find-initial-point
  ([model inference-result]
   `((apply juxt (:parameter-names ~model))
     (first (sort-by :LL clojure.core/> (:accepted ~inference-result)))))
  ([model]
   `(find-initial-point ~model (forward-sampling ~model))))

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
  [inferred selector]
  (map #(% selector) (:accepted inferred)))

(defn traces
  [inferred & r]
  (map (apply juxt (map #(fn [v]
                           (v %)) r)) (:accepted inferred)))
