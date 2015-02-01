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
import weka.core.converters.ArffSaver;
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
public class FS_NB_MLP_Hybird_CrossV_Me {
	
	public enum Selection {
		GREEDYSTEPWISE, BFFORWARD, BFBACKWARD, LINEARFORWARD, GENETIC
	}
	public enum FS {
	    FILTER, WRAPPER_NB, WRAPPER_MLP
	}

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
	
	protected static int[] filterWrapper(Instances data, FS fs, Selection selection ) throws Exception {
		AttributeSelection attsel = new AttributeSelection();
		ASEvaluation eval = null;
		switch (fs) {
		case FILTER:
			eval = new CfsSubsetEval();
			System.out.println("Filter is used.");
			break;
		case WRAPPER_NB:
			eval = new ClassifierSubsetEval();
			((ClassifierSubsetEval) eval).setClassifier(new NaiveBayes());	
			System.out.println("Wrapper NB is used.");
			break;	
		case WRAPPER_MLP:
			eval = new ClassifierSubsetEval();
			((ClassifierSubsetEval) eval).setClassifier(new MultilayerPerceptron());	
			System.out.println("Wrapper MLP is used.");
			break;
		default:
			System.out.println("Specify the search type!");
			break;
		}	
		 
		ASSearch search = null;
		switch (selection) {
		case GREEDYSTEPWISE:
			search = new GreedyStepwise();
			((GreedyStepwise) search).setSearchBackwards(false);
			break;
		case BFFORWARD:
			search = new BestFirst();
			((BestFirst) search).setDirection(new SelectedTag(1,
					BestFirst.TAGS_SELECTION));
			break;
		case BFBACKWARD:
			search = new BestFirst();
			((BestFirst) search).setDirection(new SelectedTag(0,
					BestFirst.TAGS_SELECTION));
			break;
		case LINEARFORWARD:
			search = new LinearForwardSelection();
			break;
		case GENETIC:
			search = new GeneticSearch();
			((GeneticSearch) search).setMaxGenerations(2000);
			((GeneticSearch) search).setReportFrequency(20);
			break;			
		default:
			System.out.println("Specify the search method!");
			break;
		}	 
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
	
	protected static Instances getSelectedData(String path, Instances data,
			int[] indices) throws Exception {
		// remove unselected attributes.
		for (int i = data.numAttributes() - 1; i >= 0; i--) {
			boolean remove = true;
			for (int k = 0; k < indices.length; k++) {
				if (i == indices[k]) {
					remove = false;
					break;
				}
			}
			if (remove == true) {
				data.deleteAttributeAt(i);
			}
		}
		return data;
	}

	/**
	 * uses classifier to train and test
	 * @return 
	 */
	protected static double traintest_Hao(Instances trainingData) throws Exception {
		System.out
				.println("===========================================\ntrain and test.");
		// remove unselected attributes.
/*		for (int i = trainingData.numAttributes() - 1; i >= 0; i--) {
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
		}*/
		// cross validation
		System.out.println("Cross validation. The number of features is: " + trainingData.numAttributes());
		Classifier mlp = new MultilayerPerceptron();
		Evaluation evaluation = new Evaluation(trainingData);
		evaluation.crossValidateModel(mlp, trainingData, 10, new Random(1));
		System.out.println(evaluation.toSummaryString());
		return evaluation.errorRate();
	}
	
	public static void save(String path, Instances data, String name) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File(path + name));
//		saver.setDestination(new File("./data/test.arff")); // **not** necessary
		saver.writeBatch();
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
		System.out.println("Use wrapper. Navie Bayes is used as classifier.");
		System.out.println("UCI data set, proposed selection. wrapper (greedystepwise forward )"
				+ " + wrapper (genetic).");
		// System.out.println("\n0. Loading data");
//		String path = System.getProperty("user.dir") + File.separator + "sg" + File.separator;
		String path = "C:\\Program Files\\Weka-3-6\\data\\datasets_UCI\\"
	    		+ "uci-20050214\\NB_MLP_experiment\\audiology\\";
		String file = "audiology.arff";
		Boolean phase2 = false;
		double errorRate;
		int[] indices;
		Instances trainingSelectedData;
		final long startTime1, duration1, startTime2, duration;

		DataSource source = new DataSource(path + file);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		// phase 1
		startTime1 = System.nanoTime();
		indices = filterWrapper(data, FS.WRAPPER_NB, Selection.GREEDYSTEPWISE);
		duration1 = System.nanoTime() - startTime1;
		trainingSelectedData = getSelectedData(path, data, indices);
		errorRate = traintest_Hao(trainingSelectedData);
		startTime2 = System.nanoTime();
		save(path, trainingSelectedData, "adaptedGreedy_NB.arff");

		// phase 2
		if (phase2) {
			indices = filterWrapper(trainingSelectedData, FS.WRAPPER_MLP,
					Selection.BFBACKWARD);
			trainingSelectedData = getSelectedData(path, trainingSelectedData,
					indices);
			final long duration2 = System.nanoTime() - startTime2;
			duration = duration1 + duration2;
			// read the training and testing datasets, filter them and do
			// training
			// and testing.
			errorRate = traintest_Hao(trainingSelectedData);
		}
		System.out.println("Result of " + file + ": ");
		System.out.println("Dimensionality of selected features: "
				+ indices.length);
//		System.out.println("Running time for feature selection: " + duration
//				/ 1000000000 + " seconds");
		System.out.println("Error rate: " + errorRate);
		System.out.println("====================================");
	}
}
