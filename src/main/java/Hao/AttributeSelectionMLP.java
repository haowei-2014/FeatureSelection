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

import Hao.AttributeSelectionNB_CrossValidation_Me.FS;
import Hao.AttributeSelectionNB_CrossValidation_Me.Selection;

/**
 * performs attribute selection using CfsSubsetEval and GreedyStepwise
 * (backwards) and trains J48 with that. Needs 3.5.5 or higher to compile.
 * 
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class AttributeSelectionMLP {

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
	
	protected static int[] wrapperFS(Instances data) throws Exception {
		AttributeSelection attsel = new AttributeSelection();
		ClassifierSubsetEval eval = new ClassifierSubsetEval();
		eval.setClassifier(new MultilayerPerceptron());	
/*	protected static int[] filterFS(Instances data) throws Exception {
		AttributeSelection attsel = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();	*/	
		
		 GreedyStepwise search = new GreedyStepwise();
		 search.setSearchBackwards(false);
//		BestFirst search = new BestFirst();
//		search.setDirection(new SelectedTag(0, BestFirst.TAGS_SELECTION));
//		 GeneticSearch search = new GeneticSearch();
//		 search.setMaxGenerations(2000);
//		 search.setReportFrequency(20);
//		 LinearForwardSelection search = new LinearForwardSelection();
//		 search.setForwardSelectionMethod(new SelectedTag(1, LinearForwardSelection.TAGS_SEARCH_METHOD));
		// GeneticSearchHao search = new GeneticSearchHao();
		attsel.setEvaluator(eval);
		attsel.setSearch(search);
		attsel.SelectAttributes(data);
		int[] indices = attsel.selectedAttributes();
//		System.out.println("selected attribute indices (starting with 0):\n" + Utils.arrayToString(indices));
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
	 * @return 
	 */
	protected static double traintest_Hao(Instances trainingData,
			int[] indices, boolean fs) throws Exception {
		if (fs) {
			System.out.println("train and test on the reduced features.");
			// remove unselected attributes.
			for (int i = trainingData.numAttributes() - 1; i >= 0; i--) {
				boolean remove = true;
				for (int k = 0; k < indices.length; k++) {
					if (i == indices[k]) {
						remove = false;
						break;
					}
				}
				if (remove == true) {
					trainingData.deleteAttributeAt(i);
				}
			}
		} else 
			System.out.println("train and test on the full features.");
		// cross validation
		Classifier mlp = new MultilayerPerceptron();
		Evaluation evaluation = new Evaluation(trainingData);
		evaluation.crossValidateModel(mlp, trainingData, 10, new Random(1));
		System.out.println(evaluation.toSummaryString());
		return evaluation.errorRate();
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
		System.out.println("Use wrapper. MLP is used as classifier.");
		System.out.println("Mathias data set. wrapper (greedystepwise forward )");
		// System.out.println("\n0. Loading data");
		 String path = System.getProperty("user.dir") + File.separator + "Mathias" + File.separator;
//		String path = "E:\\HisDoc project\\Mathias\\";
		String[] files = { "set_3000_1.arff", "set_10000_1.arff", "set_10000_5.arff",
				"set_20000_1.arff", "set_20000_4.arff"};

		for (int i = 0; i < files.length; i++) {
			System.out.println("========================================");
			System.out.println(files[i] + " is used.");
			DataSource source = new DataSource(path + files[i]);
			Instances data = source.getDataSet();
			if (data.classIndex() == -1)
				data.setClassIndex(data.numAttributes() - 1);
			final long startTime1 = System.nanoTime();
			int[] indices = wrapperFS(data);
			traintest_Hao(data, indices, false);
			traintest_Hao(data, indices, true);
			final long duration1 = System.nanoTime() - startTime1;
		}
	}
}
