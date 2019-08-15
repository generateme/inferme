(defproject generateme/inferme "0.0.1-SNAPSHOT"
  :description "MCMC based Bayesian inference toolkit"
  :url "https://github.com/generateme/inferme"
  :license {:name "The Unlicense"
            :url "http://unlicense.org"}
  :scm {:name "git"
        :url "https://github.com/generateme/inferme/"}  
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [generateme/fastmath "1.4.0-SNAPSHOT"]
                 [clojure2d "1.2.0-SNAPSHOT"]
                 [cljplot "0.0.2-SNAPSHOT"]])
