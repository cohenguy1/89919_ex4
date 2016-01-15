import java.util.List;
import java.util.*;

import InputOutput.DataClass;
import InputOutput.Document;

public class EMAlgorithm 
{
	public final static int NumOfClusters = 9;
	
	double clustersProb[];
	
	Map<Document, Double[]> wti; 
	
	public EMAlgorithm()
	{
		clustersProb = new double[NumOfClusters];
		wti = new HashMap<Document, Double[]>();
	}
	
	public void RunAlgorithm(DataClass devData, List<Cluster> clusters)
	{
		List<Document> docsList = devData.getDocsList(); 
		
		CalcClustersProb(clusters, docsList.size());
		
		CalcEStep(docsList, clusters);
	}

	private void CalcClustersProb(List<Cluster> clusters, int docsListSize)
	{
		for (int i = 0; i < NumOfClusters; i++)
		{
			clustersProb[i] = clusters.get(i).documents.size() / (double)docsListSize;
		}
	}
	
	private void CalcEStep(List<Document> docList, List<Cluster> clusters)
	{
		for (Document doc : docList)
		{
			Double[] clusterProbForDoc = new Double[NumOfClusters];
			
			for (int i = 0; i < NumOfClusters; i++)
			{				
				clusterProbForDoc[i] = GetClassificationProb(doc, clusters, i); 
			}
			
			
			wti.put(doc, clusterProbForDoc);
		}
	}

	private double GetClassificationProb(Document doc, List<Cluster> clusters, int clusterIndex)
	{
		double prob1 = GetProbByCluster(doc, clusters, clusterIndex);
		double prob2 = GetProbForAllClusters(doc, clusters);
		
		return clustersProb[clusterIndex] * prob1 / prob2;
	}
	
	private double GetProbByCluster(Document doc, List<Cluster> clusters, int clusterIndex)
	{
		double prob1 = 1;
		
		for (String word : doc.WordsMap.keySet())
		{
			for (int j = 0; j < doc.WordsMap.get(word); j++)
			{
				double probForWordInCluster = GetNumOfOccursInCluster(clusterIndex, clusters, word);
				if (probForWordInCluster > 0)
				{
					prob1 *= probForWordInCluster;
				}
			}
		}
		
		return prob1;
	}
	
	private double GetProbForAllClusters(Document doc, List<Cluster> clusters)
	{	
		double prob2 = 0;
		
		for (int clusterIndex = 0; clusterIndex < NumOfClusters; clusterIndex++)
		{
			double prob1 = GetProbByCluster(doc, clusters, clusterIndex);
			prob2 = clustersProb[clusterIndex] * prob1;
		}
		
		return prob2;
	}
	
	private double GetNumOfOccursInCluster(int i, List<Cluster> clusters, String word) 
	{
		Cluster cluster = clusters.get(i);
		return GetNumOfOccursInCluster(cluster, word);
	}

	private double GetNumOfOccursInCluster(Cluster cluster, String word) 
	{
		int totalWordCount = 0;
		int wordCount = 0;
		
		for (Document doc : cluster.documents)
		{
			wordCount += doc.getWordOccurrences(word);
			totalWordCount += doc.words.length;
		}
		
		return wordCount / (double)totalWordCount;
	}
}
