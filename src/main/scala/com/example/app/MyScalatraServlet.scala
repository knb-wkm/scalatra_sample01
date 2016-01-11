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

class MyScalatraServlet extends MyScalatraWebAppStack {
  val conf = new SparkConf().setAppName("simple application").setMaster("local")
  val sc = new SparkContext(conf)
  val labels = sc.objectFile("model/labels.txt")
  val texts = sc.objectFile("model/texts.txt")
  val htf = new HashingTF(1000)
  val tf = htf.transform(texts)
  val model = NaiveBayesModel.load(sc, "model")
  val words = List("武将"," 大名", "天下人", "関白", "太閤")
  val test_tf = htf.transform(words)
  val test = model.predict(test_tf)

  get("/") {
    <html>
      <body>
        <h1>Hello, World!</h1>
        <p>{test}</p>
      </body>
    </html>
  }
}
