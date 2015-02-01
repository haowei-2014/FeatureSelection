package Hao;

import weka.attributeSelection.*;
import weka.core.*;
import weka.core.converters.ConverterUtils.*;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.*;
import weka.classifiers.trees.*;
import weka.filters.*;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * performs attribute selection using CfsSubsetEval and GreedyStepwise
 * (backwards) and trains J48 with that. Needs 3.5.5 or higher to compile.
 * 
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class AttributeSelectionTest {

	/**
	 * uses the meta-classifier
	 */
	protected static void useClassifier(Instances data) throws Exception {
		System.out.println("\n1. Meta-classfier");
		AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
		// CfsSubsetEval eval = new CfsSubsetEval();
		ClassifierSubsetEval eval = new ClassifierSubsetEval();
		eval.setClassifier(new NaiveBayes());
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(false);
		// J48 base = new J48();
		NaiveBayes base = new NaiveBayes();
		classifier.setClassifier(base);
		classifier.setEvaluator(eval);
		classifier.setSearch(search);
		Evaluation evaluation = new Evaluation(data);
		evaluation.crossValidateModel(classifier, data, 10, new Random(1));
		System.out.println(evaluation.toSummaryString());
	}

	/**
	 * uses the filter
	 */
	protected static void useFilter(Instances data) throws Exception {
		System.out.println("\n2. Filter");
		weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
		ClassifierSubsetEval eval = new ClassifierSubsetEval();
		eval.setClassifier(new NaiveBayes());
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(false);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(data);
		Instances newData = Filter.useFilter(data, filter);
		System.out.println(newData);
	}

	/**
	 * uses the low level approach
	 */
	protected static void useLowLevel(Instances data) throws Exception {
		System.out.println("\n3. Low-level");
		AttributeSelection attsel = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(false);
		// BestFirst search = new BestFirst();
		// search.setDirection(new SelectedTag(2, BestFirst.TAGS_SELECTION));
		// GeneticSearch search = new GeneticSearch();
		// search.setMaxGenerations(8000);
		// search.setReportFrequency(20);
		// LinearForwardSelection search = new LinearForwardSelection();
		// GeneticSearchHao search = new GeneticSearchHao();
		attsel.setEvaluator(eval);
		attsel.setSearch(search);
		attsel.SelectAttributes(data);
		int[] indices = attsel.selectedAttributes();
		System.out.println("selected attribute indices (starting with 0):\n"
				+ Utils.arrayToString(indices));
		// begin modification Hao
		int nbAttributes = 0;
		for (int i = 0; i < indices.length; i++) {
			indices[i] += 1;
			nbAttributes++;
		}
		System.out.println("selected attribute indices (starting with 1):\n"
				+ Utils.arrayToString(indices));
		System.out.println("Number of selected attributes: " + nbAttributes);
		// end modification
	}

	protected static int[] useWrapper(Instances data) throws Exception {
		// System.out.println("\n4. Use wrppper. NN is used as classifier.");
		// System.out.println("parz data set, genetic selection.");
		AttributeSelection attsel = new AttributeSelection();
		ClassifierSubsetEval eval = new ClassifierSubsetEval();
		eval.setClassifier(new MultilayerPerceptron());
		// eval.setClassifier(new NaiveBayes());
		// GreedyStepwise search = new GreedyStepwise();
		// search.setSearchBackwards(false);
//		 BestFirst search = new BestFirst();
//		 search.setDirection(new SelectedTag(1, BestFirst.TAGS_SELECTION));
		GeneticSearch search = new GeneticSearch();
		search.setMaxGenerations(2000);
		search.setReportFrequency(20);
//	  LinearForwardSelection search = new LinearForwardSelection();
		// GeneticSearchHao search = new GeneticSearchHao();
		attsel.setEvaluator(eval);
		attsel.setSearch(search);
		attsel.SelectAttributes(data);
		int[] indices = attsel.selectedAttributes();
		System.out.println("selected attribute indices (starting with 0):\n"
				+ Utils.arrayToString(indices));
		// begin modification Hao
		int nbAttributes = 0;
		int[] indicesShow = new int[indices.length];
		for (int i = 0; i < indices.length; i++) {
			indicesShow[i] = indices[i] + 1;
			nbAttributes++;
		}
		System.out.println("selected attribute indices (starting with 1):\n"
				+ Utils.arrayToString(indicesShow));
		System.out.println("Number of selected attributes: " + nbAttributes);
		return indices;
		// end modification
	}

	/**
	 * uses the filter
	 */
	protected static Instances useFilter_Hao(Instances data) throws Exception {
		System.out.println("\n2. Filter");
		weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
		ClassifierSubsetEval eval = new ClassifierSubsetEval();
		eval.setClassifier(new NaiveBayes());
		BestFirst search = new BestFirst();
		search.setDirection(new SelectedTag(1, BestFirst.TAGS_SELECTION));
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(data);
		Instances newData = Filter.useFilter(data, filter);
		// System.out.println(newData);
		
		return newData;
	}

	/**
	 * uses classifier to train and test
	 */
	protected static void traintest_Hao(String path, String trainingFile,
			String testingFile, int[] score) throws Exception {
		System.out.println("===========================================\ntrain and test.");
		
		// read training and testing data sets.
		DataSource trainingSource = new DataSource(path + trainingFile);
		Instances trainingData = trainingSource.getDataSet();
		if (trainingData.classIndex() == -1)
			trainingData.setClassIndex(trainingData.numAttributes() - 1);
		DataSource testingSource = new DataSource(path + testingFile);
		Instances testingData = testingSource.getDataSet();
		if (testingData.classIndex() == -1)
			testingData.setClassIndex(testingData.numAttributes() - 1);
		
		// generate the indices of selected attributes.
		int numSelectedAttributes = 0;
		for (int i = 0; i < score.length; i++)
		{
			if (score[i] > 2)
				numSelectedAttributes++;
		}
		int[] selectedAttributes = new int[numSelectedAttributes];
		int j = 0;
		for (int i = 0; i < score.length; i++)
		{
			if (score[i] > 2){
				selectedAttributes[j] = i;
				j++;
			}
		}
		int[] selectedAttributesShow = new int[numSelectedAttributes];
		for (int i = 0; i < numSelectedAttributes; i++)
			selectedAttributesShow[i] = selectedAttributes[i] + 1;
		System.out.println("selected attribute indices (starting with 0):\n"
				+ Utils.arrayToString(selectedAttributes));
		System.out.println("selected attribute indices (starting with 1):\n"
				+ Utils.arrayToString(selectedAttributesShow));
		
		// remove unselected attributes.
		for (int i = score.length - 1; i >= 0; i--) {
			boolean remove = true;
			for (int k = 0; k < selectedAttributes.length; k++) {
				if (i == selectedAttributes[k]) {
					remove = false;
					break;
				}
			}
			if (remove == true) {
				trainingData.deleteAttributeAt(i);
				testingData.deleteAttributeAt(i);
			}
		}
		// train and test
		 Classifier mlp = new MultilayerPerceptron();
		 mlp.buildClassifier(trainingData);
		 Evaluation eval = new Evaluation(trainingData);
		 eval.evaluateModel(mlp, testingData);
		 System.out.println(eval.toSummaryString("\nResults\n======\n", false));
	}

	/**
	 * takes a dataset as first argument
	 * 
	 * @param args
	 *            the commandline arguments
	 * @throws Exception
	 *             if something goes wrong
	 */
	public static void main(String[] args) throws Exception {
		// load data
		final long startTime = System.nanoTime();
		System.out.println("Use wrppper. MLP is used as classifier. Five random"
				+ "subfiles are used to select attributes.");
		System.out.println("gw data set, genetic selection.");
//		System.out.println("\n0. Loading data");
//		String path = System.getProperty("user.dir") + File.separator + "gw" + File.separator;
		String path = "C:\\Program Files\\Weka-3-6\\data\\";
//		String file = "parzival_training.arff";
		// DataSource source = new DataSource(path + file);

		/*String[] files = { "parz_trainingRan1.arff", "parz_trainingRan2.arff",
				"parz_trainingRan3.arff", "parz_trainingRan4.arff",
				"parz_trainingRan5.arff"};*/
		String[] files = { "iris.arff", "gw_trainingRan2.arff",
				"gw_trainingRan3.arff", "gw_trainingRan4.arff",
				"gw_trainingRan5.arff"};
		int[] score = new int[205];
		for (int i = 0; i < files.length; i++) {
			System.out.println("===========================================");
			System.out.println(files[i] + " is being processed.");
			DataSource source = new DataSource(path + files[i]);
			Instances data = source.getDataSet();
			if (data.classIndex() == -1)
				data.setClassIndex(data.numAttributes() - 1);
			int[] indices = useWrapper(data);
			for (int j = 0; j < score.length; j++){
				for (int k = 0; k < indices.length; k++)
				{
					if (j == indices[k]){
						score[j]++;
						break;
					}
				}
			}
		}
		System.out.println("===========================================");
		final long duration = System.nanoTime() - startTime;
		System.out.println(duration + " nanoseconds is used for feature selection on all data sets.");
		System.out.println(duration / 1000000000 + " seconds is used for feature selection on all data sets.");

		// read the training and testing datasets, filter them and do training and testing.
		traintest_Hao(path, "gw_training.arff", "gw_testing.arff", score);
		
		/*
		 * DataSource source = new DataSource(
		 * "C:\\Program Files\\Weka-3-6\\Parzival\\breast-cancer.arff");
		 * //training006_204.png //traing006_Select //
		 * "C:\\Program Files\\Weka-3-6\\data\\datasets_UCI" // +
		 * "\\uci-20050214\\My_experiment\\arrhythmia\\arrhythmia.arff"); //
		 * gw_training.arff Instances data = source.getDataSet(); if
		 * (data.classIndex() == -1) data.setClassIndex(data.numAttributes() -
		 * 1);
		 * 
		 * // 1. meta-classifier // useClassifier(data);
		 * 
		 * // 2. filter // useFilter(data);
		 * 
		 * // 3. low-level // useLowLevel(data);
		 * 
		 * // 4. use wrapper useWrapper(data);
		 * 
		 * // 5. use wrapper and classify // Instances newData =
		 * useFilter_Hao(data); // useClassifier_Hao(newData);
		 */
	}
}
