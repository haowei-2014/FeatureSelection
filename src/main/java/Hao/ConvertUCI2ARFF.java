package Hao;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class ConvertUCI2ARFF {

	public static void main(String[] args) {
		int nbFeatures = 310;
		try {			 
			File file = new File("E:\\HisDoc project\\FeatureSelectionForJournal"
					+ "\\datasets\\_arff\\LSVT\\LSVT.arff");
			if (!file.exists()) {
				file.createNewFile();
			}
 
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			// @RELATION
			bw.write("@RELATION LSVT");
			bw.newLine();
			bw.newLine();
			
			// @ATTRIBUTE	
			for (int i = 1; i <= nbFeatures; i++){
				String attribute = "@ATTRIBUTE feature" + i + " numeric";
				bw.write(attribute);
				bw.newLine();
			}
			bw.write("@ATTRIBUTE class {1, 2}");
			bw.newLine();
			bw.newLine();
			
			// read .data
		    BufferedReader br = new BufferedReader(new FileReader(
		    		"E:\\HisDoc project\\FeatureSelectionForJournal"
		    		+ "\\datasets\\_data\\LSVT.txt"));
		    try {
		        String line = br.readLine();
		        // @DATA
		        bw.write("@DATA");
		        bw.newLine();
		        while (line != null) {	            
		            String[] values = line.split("\t");   //\t
		            System.out.println(values.length);
		            for (int j = 0; j < values.length-1; j++){
		     //       for (int j = 1; j < values.length; j++){	
		            	bw.write(values[j]);
		            	bw.write(",");
		            }
		            bw.write(values[310]);
		            bw.newLine();     
		            line = br.readLine();
//		            System.out.println(line);
		        }
		    } finally {
		        br.close();
		    }
			bw.close();
			System.out.println("Done");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
