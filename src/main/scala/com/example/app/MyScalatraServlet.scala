package com.example.app

import org.scalatra._
import scalate.ScalateSupport
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.{HashingTF, IDF}

case class Classify(label: Double)
case class Query(word: String)

class MyScalatraServlet extends MyScalatraWebAppStack {
  val conf = new SparkConf().setAppName("simple application").setMaster("local")
  val sc = new SparkContext(conf)
  val labels = sc.objectFile("model/labels.txt")
  val texts = sc.objectFile("model/texts.txt")
  val htf = new HashingTF(1000)
  val tf = htf.transform(texts)
  val model = NaiveBayesModel.load(sc, "model")
  // val words = List("武将"," 大名", "天下人", "関白", "太閤")
  // val test_tf = htf.transform(words)
  // val test = model.predict(test_tf)

  // how to use curl
  // curl --noproxy 127.0.0.1,get.this -H "Content-Type: application/json" -d '{"word":"関白"}' http://127.0.0.1:8080/classify
  post("/classify") {
    parsedBody match {
      case json => {
        val words: Query = json.extract[Query]
        val test_tf = htf.transform(words.word.split(","))
        val test = model.predict(test_tf)
        Classify(test)
      }
      case _ => halt(400, "unknown format")
    }
  }
}
