/* Ido Cohen	Guy Cohen	203516992	304840283 */
package InputOutput;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;


public class Output {
	
	public static boolean writeToConsole = false;

	//TODO: delete path!
	public static String folderPath = "";

	public int countOutput = 1;
	public String outputFile;
	public static int vocabulary_size = 300000;
	
	public Output(String outputFile){
		this.outputFile = folderPath + outputFile;
	}
	
	public void writeNames(){
		writeOutputFile("#Students\tIdo Cohen\tGuy Cohen\t203516992\t304840283",false);
	}
	
	public void writeOutput(String output){
		writeOutputFile("\n#Output" + countOutput++ + "\t" + output);
	}
	
	public void writeOutput(Number output){
		writeOutput(output.toString());
	}
	
	public void writeOutputFile(String outputLine) {
		writeOutputFile(outputLine,true);
	}
	
	public void writeOutputFile(String outputLine, Boolean toAppend) {

		try{
			FileWriter fileWriter = new FileWriter(outputFile,toAppend);

			PrintWriter out1 = new PrintWriter(fileWriter);
			out1.write(outputLine);

			writeConsoleWhenTrue(outputLine);

			out1.flush();
			out1.close();
			fileWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void writeConsoleWhenTrue(Object Line){

		if(writeToConsole){
			System.out.println(Line);
		}
	}
}
