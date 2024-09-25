(ns simple
  (:require [fastmath.core :as m]
            [fastmath.random :as r]
            [fastmath.stats :as stats]
            [inferme.core :refer :all] 
            [inferme.plot :as plot]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(m/use-primitive-operators)

;;

;; Looking for mean (should be 3)

(defmodel normal
  [mu (:normal {:sd 1000})]
  (model-result [(observe (distr :normal {:mu mu}) (repeatedly 3 #(r/grand 3 1)))]))

(def res (infer :metropolis-hastings normal {:steps [1.0]
                                             ;; :kernel (jump/bactrian-kernel (distr :laplace) 0.8)
                                             ;; :kernel (jump/bactrian-kernel (distr :normal) 0.9)
                                             :samples 10000
                                             :max-iters 1e7
                                             :thin 20
                                             :burn 5000
                                             ;; :initial-point [3]
                                             }))


(:acceptance-ratio res)
(count (trace res :mu))
(:steps res)

(stats/mean (trace res :mu))
;; => 3.017771819797229

(plot/lag (trace res :mu))

(plot/histogram (trace res :mu))

(plot/histogram (m/rank (trace res :mu)))

;; Let's find optimal step size for given model

(defn find-step
  [step]
  (:acceptance-ratio (infer :metropolis-hastings normal {:steps [step]
                                                         ;; :kernel (jump/bactrian-kernel (distr :laplace) 0.9)
                                                         ;; :kernel (jump/bactrian-kernel (distr :normal) 0.5)
                                                         :samples 1000
                                                         :initial-point [0]
                                                         })))

(plot/scatter (pmap #(vector % (find-step %)) (range 0.0001 2 0.001)))


;;

(defmodel daslu-example
  [mu (:normal)]
  (model-result [(observe1 (distr :normal {:mu mu}) 10.0)]))

(def res (infer :metropolis-hastings daslu-example {:steps [1.0]
                                                    :samples 10000
                                                    :max-iters 1e7
                                                    :thin 5
                                                    :burn 5000}))

(:acceptance-ratio res)
;; => 0.6087853430337914
(count (trace res :mu))
;; => 10000

(stats/mean (trace res :mu))
;; => 4.997783637857927

(stats/variance (trace res :mu))
;; => 0.49403256673031765

(plot/lag (trace res :mu))

(plot/histogram (trace res :mu))
(plot/histogram (m/rank (trace res :mu)))


;; bad samples, jump to big:

(def res (infer :metropolis-hastings daslu-example {:steps [12.5]
                                                  :samples 2000
                                                  :max-iters 1e7}))
(plot/lag (trace res :mu))
(plot/histogram (trace res :mu))
(plot/histogram (m/rank (trace res :mu)))


;;

(defmodel log-normal
  [scale (:cauchy {:scale 1000})
   shape (:half-cauchy {:scale 1000})]
  (model-result [(observe (distr :log-normal {:scale scale
                                              :shape shape})
                          #_[21 35 55 100 134 13 15 17 16 30 34 45 60 34 55]
                          #_[16.5
                             48.5
                             2.5
                             41.25
                             9.5
                             27.75
                             17
                             2.5
                             8.25
                             9.75
                             35.25
                             8.5
                             3
                             14.5
                             11.5
                             12.25
                             7.5
                             2
                             25
                             17.5
                             33
                             5
                             9
                             4
                             27
                             29
                             19.5
                             4.75
                             3
                             32.25
                             10.5
                             3.5
                             15
                             6.75
                             15.25
                             16
                             2.25
                             1
                             1
                             1.5
                             4
                             6.75
                             9.5
                             7
                             7.5
                             6.5
                             9
                             25.25
                             3
                             1
                             1.5
                             1.25
                             4
                             3
                             3.25
                             16.75
                             6
                             11.25
                             4
                             7.75
                             16.25
                             15.25
                             2.75
                             1.25
                             15.5
                             4.75
                             16.75
                             7.25
                             7.25
                             10.75
                             6.5
                             49.25
                             10
                             5.25
                             8.5
                             1]
                          #_[24
                             16
                             12
                             12
                             32
                             12
                             8
                             12
                             4
                             8
                             4
                             12
                             16
                             8
                             4
                             16
                             16
                             12
                             8
                             8
                             16
                             4
                             12
                             8
                             8
                             2
                             24
                             8
                             12
                             4
                             4
                             32
                             8
                             4
                             16
                             1
                             12
                             16
                             3
                             2
                             1
                             1
                             1
                             2
                             4
                             8
                             8
                             24
                             4
                             8
                             24
                             1
                             1
                             1
                             4
                             2
                             4
                             8
                             8
                             12
                             20
                             4
                             8
                             40
                             16
                             16
                             8
                             5
                             1
                             1
                             16
                             8
                             8
                             8
                             8
                             12
                             8
                             24
                             4
                             12
                             16
                             8
                             20
                             12
                             4
                             4
                             8
                             1
                             5
                             16
                             2
                             8
                             6
                             2
                             8
                             16
                             8
                             16
                             12
                             3
                             8
                             12]
                          [13.0
                           2.0
                           5.0
                           8.0
                           13.0
                           3.0
                           3.0
                           3.0
                           3.0
                           3.0
                           13.0
                           3.0
                           3.0
                           5.0
                           3.0
                           3.0
                           3.0
                           3.0
                           3.0
                           2.0
                           5.0
                           3.0
                           3.0
                           5.0
                           5.0
                           3.0
                           3.0
                           3.0
                           2.0
                           3.0
                           8.0
                           13.0
                           5.0
                           8.0
                           8.0
                           1.0
                           13.0
                           13.0
                           3.0])]))

(def res-log-normal (infer :metropolis-hastings log-normal {:step [0.5 0.5]
                                                          :initial-point [1 1]
                                                          :samples 10000
                                                          :max-iters 1e7
                                                          :thin 30
                                                          :burn 10000}))

(plot/histogram (trace res-log-normal :shape))
(plot/histogram (trace res-log-normal :scale))

(stats/mean (trace res-log-normal :shape))
(stats/mean (trace res-log-normal :scale))

(plot/lag (trace res-log-normal :shape))
(plot/lag (trace res-log-normal :scale))

