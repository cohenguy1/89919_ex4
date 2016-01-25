/* Ido Cohen	Guy Cohen	203516992	304840283 */
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import InputOutput.DataClass;
import InputOutput.Document;
import InputOutput.Output;

public class Ex4 
{
	public final static int NUM_OF_CLUSTERS = 9;

	public static void main(String[] args) 
	{
		String devl_inputFile = args[0];
		String outputFile = args[1];

		Output outputClass = new Output(outputFile);

		//Output init
		outputClass.writeNames();
		outputClass.writeOutput(devl_inputFile);
		outputClass.writeOutput(outputFile);
		outputClass.writeOutput(Output.vocabulary_size);

		try 
		{
			DataClass devData = new DataClass();
			devData.readInputFile(devl_inputFile);

			outputClass.writeOutput(devData.WordsMap.size());
			
			devData.removeRareWords();
			
			outputClass.writeOutput(devData.WordsMap.size());
			
			List<Cluster> clusters = new ArrayList<Cluster>();
			
			for (int i = 0; i < NUM_OF_CLUSTERS; i++)
			{
				clusters.add(new Cluster());
			}
			
			List<Document> docList = devData.getDocsList();
			
			outputClass.writeOutput(docList.size());
			
			for (int i = 0; i < docList.size(); i++)
			{
				clusters.get(i % NUM_OF_CLUSTERS).AddDocument(docList.get(i));
			}
			
			EMAlgorithm emAlgorithm = new EMAlgorithm();
			emAlgorithm.RunAlgorithm(devData, clusters);
		}
		catch (IOException e) 
		{
			e.printStackTrace();
		}			
	}	
}
