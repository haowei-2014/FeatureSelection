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

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

import javax.imageio.ImageIO;

/**
 * performs attribute selection using CfsSubsetEval and GreedyStepwise
 * (backwards) and trains J48 with that. Needs 3.5.5 or higher to compile.
 * 
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class AttributeSelectionNBGabor {

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
		eval.setClassifier(new NaiveBayes());	
	/*protected static int[] filterFS(Instances data) throws Exception {
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
//		System.out.println("selected attribute indices (starting with 0):\n"
//				+ Utils.arrayToString(indices));
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
	protected static double traintest_Hao(String path, String trainingFile,
			String testingFile, int[] indices) throws Exception {
		System.out
				.println("===========================================\ntrain and test.");

		// read training and testing data sets.
		DataSource trainingSource = new DataSource(path + trainingFile);
		Instances trainingData = trainingSource.getDataSet();
		if (trainingData.classIndex() == -1)
			trainingData.setClassIndex(trainingData.numAttributes() - 1);
		DataSource testingSource = new DataSource(path + testingFile);
		Instances testingData = testingSource.getDataSet();
		if (testingData.classIndex() == -1)
			testingData.setClassIndex(testingData.numAttributes() - 1);

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
				testingData.deleteAttributeAt(i);
			}
		}
		// train and test
		Classifier nb = new NaiveBayes();
		nb.buildClassifier(trainingData);
		Evaluation eval = new Evaluation(trainingData);
		eval.evaluateModel(nb, testingData);
		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		return eval.errorRate();
	}

	
	public static void trainandtest(String path, String trainingFile,
			String testingFile) throws Exception {
		// read training and testing data sets.
		DataSource trainingSource = new DataSource(path + trainingFile);
		Instances trainingData = trainingSource.getDataSet();
		if (trainingData.classIndex() == -1)
			trainingData.setClassIndex(trainingData.numAttributes() - 1);
		
		DataSource testingSource = new DataSource(path + testingFile);
		Instances testingData = testingSource.getDataSet();
		if (testingData.classIndex() == -1)
			testingData.setClassIndex(testingData.numAttributes() - 1);
		
		
		Classifier nb = new NaiveBayes();
		nb.buildClassifier(trainingData);
		System.out.println("# of training dimension: " + trainingData.numAttributes());
		System.out.println("# of testing dimension: " + testingData.numAttributes());
		System.out.println("Training is finished.");

		// evaluate classifier and print some statistics
//		Evaluation eval = new Evaluation(trainingData);
//		eval.evaluateModel(nb, testingData);
//		eval.printClassifications(nb, trainingData, testingSource, 0, null, null);

		try {
			File f2 = new File(path + "prediction.txt");
			BufferedWriter out = new BufferedWriter(new FileWriter(f2));
			String s = "";
			int lineCount = 0;
			// label instances
			for (int i = 0; i < testingData.numInstances(); i++) {
				double clsLabel = nb.classifyInstance(testingData.instance(i));
//				System.out.println(clsLabel);
				out.write(Double.toString(clsLabel));
				out.newLine();
				// System.out.println("Line " + lineCount + ": " + s);
				lineCount++;
			}
			System.out.println("Number of lines read = " + lineCount);
			out.close();
		}
		catch (IOException e) {
			System.out.println("I/O Error occurred");
		}		
//		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
	}
	
	public static void start() throws Exception {
		// load data

		System.out.println("Use wrapper NB.");
		System.out.println("Gabor data set, adapted greedy forward selection.");
		// System.out.println("\n0. Loading data");
		// String path = System.getProperty("user.dir") + File.separator +
		// "gabor" + File.separator;
		String path = "E:\\HisDoc project\\Gabor_filter\\A_Course_Experiment_FS\\wekadata\\";
		// String[] files = {"gw_trainingRan10percent1.arff",
		// "gw_trainingRan10percent2.arff",
		// "gw_trainingRan10percent3.arff"};
		String file = "gaborTr.286.0.arff";
		double averageDim = 0;
		double averageTime = 0;
		double averageError = 0;

		final long startTime = System.nanoTime();
		DataSource source = new DataSource(path + file);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		int[] indices = wrapperFS(data);
		final long duration = System.nanoTime() - startTime;
		// read the training and testing datasets, filter them and do training
		// and testing.
		/*
		 * double errorRate = traintest_Hao(path, "gw_training.arff",
		 * "gw_testing.arff", indices); System.out.println("Result of " +
		 * files[i] + ": ");
		 * System.out.println("Dimensionality of selected features: " +
		 * indices.length);
		 * System.out.println("Running time for feature selection: " + duration
		 * / 1000000000 + " seconds"); System.out.println("Error rate: " +
		 * errorRate);
		 * System.out.println("====================================");
		 * 
		 * averageDim += indices.length; averageTime += duration / 1000000000;
		 * averageError += errorRate;
		 */

		System.out.println("Average dimensionality: ");
		System.out.println("Average running time: ");
		System.out.println("Average error rate: ");
	}
	
	public static void drawResult(String path){
		int width = 482;
		int height = 2319;
		int[] result1D = new int[1117758]; 
		int[][] result2D = new int[height][width];
		BufferedImage bufferedImage = new BufferedImage(width, height,
				BufferedImage.TYPE_BYTE_BINARY);
		Color clr = null;
		try {
			int count = 0;
			String strLine;		
			FileInputStream fstream = new FileInputStream(path + "prediction.txt");
			// Get the object of DataInputStream
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));

			// Read File Line By Line
			while ((strLine = br.readLine()) != null) {
				// Print the content on the console
				result1D[count] = Integer.parseInt(strLine.substring(0, 1));
				count++;
			}
			System.out.println("Count =" + count);
			// Close the input stream
			in.close();
			br.close();
			fstream.close();
		} catch (Exception e) {// Catch exception if any
			System.err.println("Error: " + e.getMessage());
		}	
		
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				result2D[i][j] = result1D[j*height + i];
				if (result2D[i][j] == 0)
					clr = Color.black;
				else
					clr = Color.white;
				bufferedImage.setRGB(j, i, clr.getRGB());
			}
		}
		
		try {
			File file = new File(path + "prediction.png");
			if (file.exists()) {
				file.delete();
			}
			ImageIO.write(bufferedImage, "png", file);
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Done!");
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
		String path = "E:\\HisDoc project\\Gabor_filter\\A_Course_Experiment_FS\\";
		String trainingFile = "gaborTr.286.0_Greedy.arff";
		String testingFile = "gaborTe.007.0_Greedy.arff";
//		String trainingFile = "iris.arff";
//		String testingFile = "irisCopy.arff";
		trainandtest(path, trainingFile, testingFile);
		drawResult(path);
	}
}
