(defproject generateme/inferme "0.0.3-SNAPSHOT"
  :description "MCMC based Bayesian inference toolkit"
  :url "https://github.com/generateme/inferme"
  :license {:name "The Unlicense"
            :url "http://unlicense.org"}
  :scm {:name "git"
        :url "https://github.com/generateme/inferme/"}  
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [cljplot "0.0.3"]]
  :profiles {:dev-old {:dependencies [[anglican "1.1.0"]
                                      [metaprob "0.1.0-SNAPSHOT"]]}})
