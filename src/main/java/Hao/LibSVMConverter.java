package Hao;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;

import weka.core.Instances;
import weka.core.converters.*;

public class LibSVMConverter {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		prepareGaborData();
		String path = "E:\\HisDoc project\\Gabor_filter\\A_Course_Experiment_FS\\";
	//			+ "old_files\\d-263\\";
		String nameLibsvm = "gabor.libsvm";
		String nameWeka = "gabor.arff";
		generateTemp(path, nameLibsvm);
		modify(path, nameLibsvm, nameWeka);    
	    System.out.println("Done!");
	}
	
	public static void generateTemp(String path, String nameLibsvm) throws IOException
	{
	    LibSVMLoader loader = new LibSVMLoader();
	    try {
			loader.setSource(new File(path + nameLibsvm));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    Instances data = null;
		try {
			data = loader.getDataSet();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    ArffSaver saver = new ArffSaver();
	    saver.setInstances(data);
	    saver.setFile(new File(path + nameLibsvm + ".tmp"));
	    saver.writeBatch();
	}
	
	public static void modify(String path, String nameLibsvm, String nameWeka)
	{
		try {
			File f1 = new File(path + nameLibsvm + ".tmp");
			BufferedReader in = new BufferedReader(new FileReader(f1));
			File f2 = new File(path + nameWeka);
			BufferedWriter out = new BufferedWriter(new FileWriter(f2));
			String s = "";
			int lineCount = 0;
			while ((s = in.readLine()) != null) {
				if (s.equals("@attribute class numeric")){
//					out.write("@attribute class {1,2,3,4}");
					out.write("@attribute class {1,2}");
					System.out.println("Line " + lineCount + ": " + s);
					}
				else
					out.write(s);
				out.newLine();
//				System.out.println("Line " + lineCount + ": " + s);
				lineCount++;
			}
			System.out.println("Number of lines read = " + lineCount);
			in.close();
			out.close();
			f1.delete();
		}
         
        catch(IOException e){
            System.out.println("I/O Error occurred");
        }
	}
	
	public static void prepareGaborData(){

		// TODO Auto-generated method stub
		try {
			// Open the file that is the first
			// command line parameter
			int count;
			int i = 1;
			int j = 0;
			int lineelementindex;

			// String trainingname;
			String testingname;
			File ft;

			File fi;

			FileInputStream fstream_feature;
			FileInputStream fstream_target;
			// BufferedWriter bwtraining;
			BufferedWriter bwtesting;

			// Get the object of DataInputStream
			DataInputStream in_feature;
			DataInputStream in_target;
			BufferedReader br_feature;
			BufferedReader br_target;
			String strFeature;
			String newstrFeature;
			String strTarget;
			String newstrTarget = "";
			String newStr;
			String target;
			String input;
			String yesorno;
			// Scanner readUserInput;
			BufferedReader dataIn;
			String[] lineElement;

			count = 0;
			String path = "E:\\HisDoc project\\Gabor_filter\\A_Course_Experiment_FS\\";
//					+ "old_files\\d-263\\";

			testingname = path + "gabor.libsvm";
			ft = new File(testingname);
			if (ft.exists())
				ft.delete();
			bwtesting = new BufferedWriter(new FileWriter(
					new File(testingname), true));

			fstream_feature = new FileInputStream(path + "gaborFeatures.txt");
			in_feature = new DataInputStream(fstream_feature);
			br_feature = new BufferedReader(new InputStreamReader(in_feature));

			fstream_target = new FileInputStream(path + "gaborLabels.txt");
			in_target = new DataInputStream(fstream_target);
			br_target = new BufferedReader(new InputStreamReader(in_target));

			// Read File Line By Line
			while ((strFeature = br_feature.readLine()) != null
					&& (strTarget = br_target.readLine()) != null) {
				// Print the content on the console

				newstrFeature = "";
				lineElement = strFeature.split(",");
				for (lineelementindex = 0; lineelementindex < lineElement.length; lineelementindex++) {
					lineElement[lineelementindex] = Integer
							.toString((lineelementindex + 1))
							+ ":"
							+ lineElement[lineelementindex];
					// System.out.println(lineElement[lineelementindex]);
					newstrFeature = newstrFeature
							+ lineElement[lineelementindex] + " ";
				}
				// System.out.println(newstrFeature);
				newstrFeature = newstrFeature.substring(0,
						newstrFeature.length() - 1);
				// System.out.println(newstrFeature);

/*				switch (strTarget) {
				case "1 0 0 0":
					newstrTarget = "+1 ";
					break;
				case "0 1 0 0":
					newstrTarget = "+2 ";
					break;
				case "0 0 1 0":
					newstrTarget = "+3 ";
					break;
				case "0 0 0 1":
					newstrTarget = "+4 ";
					break;
				}*/
				newstrTarget = strTarget + " ";

				newStr = newstrTarget + newstrFeature;
//				System.out.println(newStr);

				// bwtraining.write(newStr);
				// bwtraining.newLine();

				bwtesting.write(newStr);
				bwtesting.newLine();

				count++;
//				System.out.println("Count =" + count);
			}
			// System.out.println("The dataset analysis of" +
			// testingfilenames[j] +"is completed.");
			// System.out.println("The dataset analysis of" +
			// trainingfilenames[j] +"is completed.");
			System.out.println("Count =" + count);
			// Close the input stream

			in_feature.close();
			in_target.close();
			// bwtraining.close();
			bwtesting.close();			
		} catch (Exception e) {// Catch exception if any
			System.err.println("Error: " + e.getMessage());
		}
	}
}
