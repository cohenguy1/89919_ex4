/* Ido Cohen	Guy Cohen	203516992	304840283 */
import java.io.IOException;
import java.util.*;

import InputOutput.DataClass;
import InputOutput.Document;
import InputOutput.Topics;

public class EMAlgorithm 
{
	public final static int NumOfClusters = Ex4.NUM_OF_CLUSTERS;

	double clustersProb[];
	long numberOfRelevantWords; //This is the new vocabulary size

	long relevantWordsCount;
	List<Document> docsList; 
	List<Set<Topics>> docsTopicList;

	Map<Document, Double[]> Wti = new TreeMap<Document, Double[]>(); 
	Map<Document, Double[]> Zti = new TreeMap<Document, Double[]>(); 
	Map<Document, Double> Mt = new TreeMap<Document, Double>(); 

	Map<String, Double[]> Pik = new TreeMap<String, Double[]>();

	//ASUMING:
	private double lidstonLambda = 1;
	private double paramK = 10;
	private double threshold = 0.000001;
	private double stopThreshold = 10;


	public EMAlgorithm()
	{
		clustersProb = new double[NumOfClusters];
		Wti = new HashMap<Document, Double[]>();
	}

	public void RunAlgorithm(DataClass devData, List<Cluster> clusters)
	{
		docsList = devData.getDocsList(); 
		numberOfRelevantWords = devData.WordsMap.size();
		docsTopicList = devData.getDocsTopicList();

		CountRelevantWordsCount(devData.WordsMap);
		System.out.println("Relevant words " + relevantWordsCount);

		InitialEStep(devData.WordsMap, clusters, devData.getDocsList().size());
		calcMStep(devData,clusters);

		double lastLikelihood = -999999999;
		double likelihood = -999999999+2*stopThreshold;
		List<Double> likelihoodList = new ArrayList<Double>();
		double perplexity = 0;
		List<Double> perplexityList = new ArrayList<Double>();
		int iteration = 0;

		while (likelihood- lastLikelihood > stopThreshold ){
			calcEStep(devData,clusters);

			calcMStep(devData,clusters);

			lastLikelihood = likelihood;
			likelihood = calcLikelihood();
			likelihoodList.add(likelihood);
			System.out.println("Likelihood- " + likelihood);

			perplexity = calcPerplexity(likelihood);
			perplexityList.add(perplexity);
			System.out.println("Perplexity- " + perplexity);

			iteration++;
			System.out.println("iteration " + iteration);
		}

		System.out.println("All Likelihood- " + likelihoodList);
		System.out.println("All Perplexity- " + perplexityList);

		Map<Integer,List<Document>> docsInCluster = new TreeMap<Integer,List<Document>>();
		List<Topics> mainClusterTopic = new ArrayList<Topics>();
		Integer[][] confusionMatrix = calcConfusionMatrix(docsInCluster, mainClusterTopic);

		System.out.println("Confusion Matrix:");
		System.out.println("==========================");
		for (int i=0; i<NumOfClusters; i++){
			for(int j=0; j <= Topics.values().length; j++) {
				System.out.print(confusionMatrix[i][j] + " ");
			}
			System.out.println();
		}

		ClassifyDocs(docsInCluster, mainClusterTopic);

		double accuracy = CalcAccuracy(docsInCluster);
		System.out.println("accuracy- " + accuracy);
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void ClassifyDocs(Map<Integer,List<Document>> docsInCluster, List<Topics> mainClusterTopic)	{
		// Iterate over each document in each cluster
		for (Integer clusterIndex : docsInCluster.keySet())	{
			for (Document doc : docsInCluster.get(clusterIndex)) {
				doc.Classify(mainClusterTopic.get(clusterIndex));
			}
		}
	}

	private double CalcAccuracy(Map<Integer,List<Document>> docsInCluster) {
		double correctClassifications = 0;
		for (Integer clusterIndex : docsInCluster.keySet())	{
			for (Document doc : docsInCluster.get(clusterIndex)) {
				if (doc.getTopics().contains(doc.getClassification()))
				{
					correctClassifications++;
				}
			}
		}

		return correctClassifications/docsList.size();
	}

	private Integer[][] calcConfusionMatrix(Map<Integer,List<Document>> docsInCluster, List<Topics> mainClusterTopic) {
		//Find Cluster of each document
		int maxCluster;
		for (Document doc : docsList){
			Double maxWt = Wti.get(doc)[0];
			maxCluster = 0;
			for (int i=1; i<NumOfClusters; i++){
				Double wti = Wti.get(doc)[i];
				if (wti > maxWt){
					maxWt = wti;
					maxCluster = i;
				}
			}
			if (docsInCluster.get(maxCluster) == null){
				docsInCluster.put(maxCluster, new ArrayList<Document>());
			}
			docsInCluster.get(maxCluster).add(doc);
		}

		//Calculate Confusion Matrix
		Integer[][] confustionMatrix = new Integer[NumOfClusters][Topics.getNumberOfTopcis()+1];
		for (int i=0; i<NumOfClusters; i++){
			int j=0;
			for(Topics topic : Topics.values()){
				confustionMatrix[i][j]=0;
				if (docsInCluster.get(i) != null){
					for(Document doc : docsInCluster.get(i)){
						if(doc.getTopics().contains(topic)){
							confustionMatrix[i][j] += 1;
						}
					}
				}
				j++;
			}
			confustionMatrix[i][j] = docsInCluster.get(i).size(); //add number of documents in cluster in the last column
		}

		//Find main topic in each cluster
		for (int i=0; i<NumOfClusters; i++){
			int j=0;
			Topics mainTopic = null;
			Integer mainTopicAmount = 0;
			for(Topics topic : Topics.values()){
				if (confustionMatrix[i][j] > mainTopicAmount){
					mainTopicAmount = confustionMatrix[i][j];
					mainTopic = topic;
				}
				j++;
			}
			mainClusterTopic.add(mainTopic);
		}

		System.out.println("Main Topics- " + mainClusterTopic);

		return confustionMatrix;
	}

	private double calcPerplexity(double likelihood) {
		return Math.pow(2, -1.0/relevantWordsCount * likelihood);
	}

	private double calcLikelihood() {
		double likelihood = 0;
		double sumZt = 0;
		double maxZt = 0;
		double newZti = 0;
		for (Document doc : Mt.keySet()){
			sumZt = 0;
			maxZt = Mt.get(doc);
			if (Zti.get(doc) != null){
				for (Double zti: Zti.get(doc)){
					newZti = zti - maxZt;
					if (-1*paramK <= newZti ){
						sumZt += Math.exp(newZti);
					}
				}
			}

			likelihood += maxZt + Math.log(sumZt);			
		}

		return likelihood;
	}

	private void CountRelevantWordsCount(Map<String, Integer> wordsMap)
	{
		relevantWordsCount = 0;

		for (Integer wordCount : wordsMap.values())
		{
			relevantWordsCount += wordCount;
		}
	}

	private void calcEStep(DataClass devData, List<Cluster> clusters) {
		for (Document doc : docsList ){
			//			count++;
			Double[] Zi = calcZti(devData.WordsMap,doc); 
			double maxZ = getMaxZ(Zi);
			Zti.put(doc, Zi);
			Mt.put(doc, maxZ);

			Double[] clusterProbForDoc = new Double[NumOfClusters];

			double sumZi = 0;
			for (int i = 0; i < NumOfClusters; i++)
			{	
				if (Zi[i]-maxZ < -1*paramK){
					clusterProbForDoc[i] = 0.0;
				}
				else{
					double denumeratorPow = Math.exp(Zi[i]-maxZ);
					clusterProbForDoc[i] = denumeratorPow;
					sumZi += denumeratorPow;
				}
			}
			for (int i = 0; i < NumOfClusters; i++)
			{	
				clusterProbForDoc[i] /= sumZi;

			}

			Wti.put(doc, clusterProbForDoc);
		}
	}

	private double getMaxZ(Double[] Zi) {
		double maxZ=Double.NEGATIVE_INFINITY;
		for (double zti : Zi){
			if (zti>maxZ){
				maxZ = zti;
			}
		}
		return maxZ;
	}

	private Double[] calcZti(Map<String, Integer> wordsMap, Document doc) {

		Double[] Zt = new Double[NumOfClusters];

		for (int i = 0; i < NumOfClusters; i++)
		{	
			double sumFrequncy = 0;
			for (String word : doc.WordsMap.keySet()){
				sumFrequncy += doc.getWordOccurrences(word) * Math.log(Pik.get(word)[i]);  //natural log
			}
			Zt[i] = Math.log(clustersProb[i]) + sumFrequncy; 
		}		

		return Zt;
	}

	private void calcMStep (DataClass devData, List<Cluster> clusters) {

		double sumWti = 0;

		Double[] denominatorI = new Double[NumOfClusters];
		Double[] pLidstone;

		for (int i = 0; i < NumOfClusters; i++) {
			sumWti = 0;
			for (Document doc : docsList) {
				if (Wti.get(doc) == null){
					System.out.println("print");
				}
				sumWti += Wti.get(doc)[i] * doc.getNumberOfRelevantWordsInDoc();
			}
			denominatorI[i] = sumWti;
		}

		for (String word : devData.WordsMap.keySet()) {
			pLidstone = new Double[NumOfClusters];

			for (int i = 0; i < NumOfClusters; i++) {
				double numerator = 0;	

				for (Document doc : docsList) {
					if (doc.hasWord(word) && Wti.get(doc)[i] != 0) {
						numerator += Wti.get(doc)[i] * doc.getWordOccurrences(word);
					}
				}

				pLidstone[i] = CalcUnigramPLidstone(numerator, denominatorI[i]); 
			}

			Pik.put(word, pLidstone);
		}

		for (int i = 0; i < NumOfClusters; i++) {
			clustersProb[i] = 0;

			for (Document doc : docsList) {
				clustersProb[i] += Wti.get(doc)[i];
			}

			clustersProb[i] /= docsList.size();
		}

		for (int i = 0; i < NumOfClusters; i++) {
			if (clustersProb[i] < threshold) {
				clustersProb[i] = threshold;
			}
		}

		double alphaSum = 0;
		for (int i = 0; i < NumOfClusters; i++) {
			alphaSum += clustersProb[i];
		}

		for (int i = 0; i < NumOfClusters; i++) {
			clustersProb[i] /= alphaSum;
		}
	}

	/**
	 * Initialize Pik - the probability for each word Wk to be in cluster i
	 * @param wordsMap
	 * @param clusters
	 * @param numberOfDocs 
	 */
	private void InitialEStep(Map<String, Integer> wordsMap, List<Cluster> clusters, int numberOfDocs) {

		for (int i=0; i<NumOfClusters; i++){
			for (Document doc : clusters.get(i).getDocuments()){
				Double[] clusterProbForDoc = new Double[NumOfClusters];
				for(int j=0; j<NumOfClusters; j++){
					clusterProbForDoc[j] = (i==j ? 1.0 : 0.0);
				}				
				Wti.put(doc, clusterProbForDoc);
			}
		}
	}

	public double CalcUnigramPLidstone(Double totalWordordOccurences, Double trainSize) {
		//		C(X)+ LAMBDA / |S| + LAMBDA*|X|
		return (totalWordordOccurences + lidstonLambda)
				/ (trainSize + lidstonLambda * numberOfRelevantWords);
	}
}
