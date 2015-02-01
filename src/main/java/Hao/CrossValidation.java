package Hao;

import java.util.Random;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.LinearForwardSelection;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

public class CrossValidation {
	


	  /**
	   * uses the meta-classifier
	   */
	  protected static void useClassifier(Instances data) throws Exception {
	    System.out.println("\n1. Meta-classfier");
	    AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
	    CfsSubsetEval eval = new CfsSubsetEval();
//	    GreedyStepwise search = new GreedyStepwise();
//	    search.setSearchBackwards(true);
//	    J48 base = new J48();
	    NaiveBayes base = new NaiveBayes();
	    classifier.setClassifier(base);
	    classifier.setEvaluator(eval);
//	    classifier.setSearch(search);
	    Evaluation evaluation = new Evaluation(data);
	    evaluation.crossValidateModel(classifier, data, 10, new Random(1));
	    System.out.println(evaluation.toSummaryString());
	  }


	  /**
	   * takes a dataset as first argument
	   *
	   * @param args        the commandline arguments
	   * @throws Exception  if something goes wrong
	   */
	  public static void main(String[] args) throws Exception {
	    // load data
	    System.out.println("\n0. Loading data");
//	    String path = System.getProperty("user.dir") + File.separator;
//		String file = "oldfeatures.txt";
	    
	    DataSource source = new DataSource(
	    		"C:\\Program Files\\Weka-3-6\\gw_training.arff");  //training006_204.png //traing006_Select
	 //   "C:\\Program Files\\Weka-3-6\\data\\datasets_UCI"
//		+ "\\uci-20050214\\My_experiment\\arrhythmia\\arrhythmia.arff");
	    Instances data = source.getDataSet();
	    if (data.classIndex() == -1)
	      data.setClassIndex(data.numAttributes() - 1);

	    // 1. meta-classifier
	    useClassifier(data);
	  }

}
