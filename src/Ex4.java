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
	public final static int NumOfClusters = 9;

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
			
			for (int i = 0; i < NumOfClusters; i++)
			{
				clusters.add(new Cluster());
			}
			
			List<Document> docList = devData.getDocsList();
			
			outputClass.writeOutput(docList.size());
			
			for (int i = 0; i < docList.size(); i++)
			{
				clusters.get(i % NumOfClusters).AddDocument(docList.get(i));
			}
		}
		catch (IOException e) 
		{
			e.printStackTrace();
		}			
	}	

	/*
	 * Returns model perplexity 
	 */
	/*private static double calculatePerplexityByBackOff(double bigramLambda, Map<String, Map<String, Integer>> validationMap, long validationSizeWithoutBeginArticle) 
	{		
		double sumPWords = 0;

		BackOff.CalculateAlphaValues(bigramLambda, validationMap.keySet());

		for (String word : validationMap.keySet())
		{			
			if (word == DataClass.BEGIN_ARTICLE)
			{
				continue;
			}

			// for each of the prev words of the word in the validation map
			for (String prevWord : validationMap.get(word).keySet())
			{
				long wordAfterPrevOccurences = DataClass.getWordOccurrences(validationMap, word, prevWord);

				double pWord = BackOff.calcBigramBackOff(bigramLambda, DataClass.TrainingMap, word,
						prevWord, BackOff.GetAlphaValue(prevWord));

				// adds the probability to the sum occurrences time (as the number of sequential occurrences in the validation map)
				sumPWords += wordAfterPrevOccurences * Math.log(pWord)/Math.log(2);
			}
		}

		double perplexity = Math.pow(2,(-1.0/validationSizeWithoutBeginArticle) * sumPWords); 
		return perplexity;
	}

	private static double getBestLambda(Map<String, Map<String, Integer>> validationMap, long validationSize)
	{
		double bestLambdaIndex = 0.0001;
		double bestPerplexityValue = calculatePerplexityByBackOff(bestLambdaIndex, validationMap, validationSize);

		double perplexity;

		// iterate over the lambdas from 0.0001 to 0.02 (1 to 200 divided by 10,000, for accuracies) 
		double DIVIDE_LAMDA = 10000.0;

		for (int lambdaIndex = 2; lambdaIndex <= 200; lambdaIndex++)
		{
			// calculate the perplexity by this lambda
			perplexity = calculatePerplexityByBackOff(lambdaIndex/DIVIDE_LAMDA, validationMap, validationSize);

			// compare to the best lambda perplexity value thus far
			if (perplexity < bestPerplexityValue)
			{
				bestLambdaIndex = lambdaIndex;
				bestPerplexityValue = perplexity;
			}
		}

		// return the best lambda
		return bestLambdaIndex/DIVIDE_LAMDA;
	}
	 */
}
