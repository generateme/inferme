(ns inferme.multi
  "Artificial multivariate distribution based on other independent distributions."
  (:require [fastmath.random :as r]
            [fastmath.protocols :as prot]
            [fastmath.core :as m]))

(defmethod r/distribution :multi
  [_ {:keys [dims distribution parameters] :as all}]
  (let [distrs (vec (repeat dims (r/distribution distribution parameters)))
        distr-id (keyword (str "multi-iid-" (name distribution)))
        m (delay (mapv prot/mean distrs))
        cv (delay (mapv prot/variance distrs))
        r (or (:rng all) (r/rng :jvm))]
    (reify
      prot/DistributionProto
      (pdf [d v] (m/exp (prot/lpdf d v)))
      (lpdf [_ v] (reduce m/fast+ (map #(prot/lpdf %1 %2) distrs v)))
      (probability [d v] (m/exp (prot/lpdf d v)))
      (sample [_] (mapv prot/sample distrs))
      (dimensions [_] dims)
      (source-object [this] this)
      (continuous? [_] (prot/continuous? (first distrs)))
      prot/DistributionIdProto
      (distribution-id [_] distr-id)
      (distribution-parameters [_] (prot/distribution-parameters (first distrs)))
      prot/MultivariateDistributionProto
      (means [_] @m)
      (covariance [_] @cv)
      prot/RNGProto
      (drandom [d] (prot/sample d))
      (frandom [d] (mapv unchecked-float (prot/sample d)))
      (lrandom [d] (mapv unchecked-long (prot/sample d)))
      (irandom [d] (mapv unchecked-int (prot/sample d)))
      (->seq [d] (repeatedly #(prot/sample d)))
      (->seq [d n] (repeatedly n #(prot/sample d)))
      (set-seed! [d seed] (prot/set-seed! r seed) d))))
