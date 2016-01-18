import java.util.*;

import InputOutput.DataClass;
import InputOutput.Document;

public class EMAlgorithm 
{
	public final static int NumOfClusters = Ex4.NUM_OF_CLUSTERS;

	double clustersProb[];
	long numberOfRelevantWords; //This is the new vocabulary size
	List<Document> docsList; 

	Map<Document, Double[]> Wti = new TreeMap<Document, Double[]>(); 
	Map<Document, Double[]> Zti = new TreeMap<Document, Double[]>(); 
	Map<Document, Double> Mt = new TreeMap<Document, Double>(); 

	Map<String, Double[]> Pik = new TreeMap<String, Double[]>();

	//ASUMING:
	private double lidstonLambda = 1.1;
	private double paramK = 10; 


	public EMAlgorithm()
	{
		clustersProb = new double[NumOfClusters];
		Wti = new HashMap<Document, Double[]>();
	}

	public void RunAlgorithm(DataClass devData, List<Cluster> clusters)
	{
		docsList = devData.getDocsList(); 
		numberOfRelevantWords = devData.WordsMap.size();

		InitialEStep(devData.WordsMap, clusters, devData.getDocsList().size());

		//		likelihood_array = []
		//	    perplexity_array = []
		double prev_likelihood = -10000001;
		double curr_likelihood = -10000000;
		int epoch = 0;
		while (curr_likelihood >= prev_likelihood){
			calcEStep(devData,clusters);
			
			calcMStep(devData,clusters);
			
		}

		//		likelihood_array = []
		//			    perplexity_array = []
		//			    prev_likelihood = -10000001
		//			    curr_likelihood = -10000000
		//			    epoch = 0
		//			    while curr_likelihood >= prev_likelihood:
		//			        w, z_list, m_list = e_step(all_words_with_all_freq, articles_with_their_words_freqs, alpha, probabilities,
		//			                                   number_of_clusters, k_param)
		//			        alpha, probabilities = m_step(w, articles_with_their_words_freqs, all_words_with_all_freq, number_of_clusters,
		//			                                      lambda_val, v_size, words_clusters)
		//			        prev_likelihood = curr_likelihood
		//			        curr_likelihood = calc_likelihood(m_list, z_list, k_param)
		//			        curr_perplexity = calc_perplexity(curr_likelihood, v_size)
		//			        print "likelihood per curr epoch (", epoch, ") - ", curr_likelihood
		//			        likelihood_array.append(curr_likelihood)
		//			        perplexity_array.append(curr_perplexity)
		//			        epoch += 1
		//			    plot_graph(epoch, likelihood_array, "likelihood")
		//			    plot_graph(epoch, perplexity_array, "perplexity")
		//			    return w
	}

	private void calcEStep(DataClass devData, List<Cluster> clusters) {
//		int count=0;
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
		
		//	    for t, doc_with_freq in articles_with_their_words_freqs.iteritems():
		//	        w[t] = {}
		//	        curr_z, max_zi = calc_z_values(all_relevant_words, number_of_clusters, alpha, probabilities, doc_with_freq, k_param)
		//	        sum_zi = 0
		//	        for i in range(0, number_of_clusters):
		//	            if curr_z[i] - max_zi < (-1.0) * k_param:
		//	                w[t][i] = 0
		//	            else:
		//	                w[t][i] = math.exp(curr_z[i] - max_zi)
		//	                sum_zi += w[t][i]
		//	        for i in range(0, number_of_clusters):
		//	            w[t][i] /= sum_zi
		//
		//	        z_list.append(curr_z)
		//	        m_list.append(max_zi)
		//	    return w, z_list, m_list

	}

	private double getMaxZ(Double[] Zi) {
		double maxZ=Double.MIN_VALUE;
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

		//		z = []
		//	    for i in range(0, number_of_clusters):
		//	        sum_of_freq_ln = 0
		//	        for word in curr_article_with_t:
		//	            sum_of_freq_ln += curr_article_with_t[word] * np.log(probabilities[word][i])
		//	        z.append(np.log(alpha[i]) + sum_of_freq_ln)
		//	    max_z = max(z)
		//	    return z, max_z

	}

	private void calcMStep (DataClass devData, List<Cluster> clusters) {
		
		double threshold = 0.000001;
		double sumWti = 0;

		Double[] denominatorI = new Double[NumOfClusters];
		Double[] pLidstone;
		
		for (int i = 0; i < NumOfClusters; i++) {
			for (Document doc : docsList) {
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
		
//		threshold = 0.000001
//			    number_of_docs = len(articles_with_their_words_frequencies)
//			    probabilities = {}
//			    denominator = []
//			    for i in range(0, number_of_clusters):
//			        denom_i = 0
//			        for t in articles_with_their_words_frequencies:
//			            len_of_t = sum(articles_with_their_words_frequencies[t].values())
//			            denom_i += weights[t][i] * len_of_t
//			        denominator.append(denom_i)
//			    for word in relevant_words_with_freq:
//			        probabilities[word] = {}
//			        for i in range(0, number_of_clusters):
//			            numerator = 0
//			            for t in articles_with_their_words_frequencies:
//			                if word in articles_with_their_words_frequencies[t] and weights[t][i] != 0:
//			                    numerator += weights[t][i] * articles_with_their_words_frequencies[t][word]
//			            probabilities[word][i] = calc_lidstone_for_unigram(numerator, denominator[i], v_size, lambda_val)
//			    # If alpha is smaller then a threshold we will scale it to the threshold to not get ln(alpha) = error
//
//			    alpha = [0] * number_of_clusters
//			    for i in range(0, number_of_clusters):
//			        for t in articles_with_their_words_frequencies:
//			            alpha[i] += weights[t][i]
//			        alpha[i] /= number_of_docs
//			    # alpha = [sum(i) / number_of_docs for i in zip(*weights)]
//			    for i in range(0, len(alpha)):
//			        if alpha[i] < threshold:
//			            alpha[i] = threshold
//			    sum_of_alpha = sum(alpha)
//			    # Normalize alpha for it to sum to 1
//			    alpha = [x / sum_of_alpha for x in alpha]
//			    return alpha, probabilities

	}
	
	/**
	 * Initialize Pik - the probability for each word Wk to be in cluster i
	 * @param wordsMap
	 * @param clusters
	 * @param numberOfDocs 
	 */
	private void InitialEStep(Map<String, Integer> wordsMap, List<Cluster> clusters, int numberOfDocs) {

		//Init Ai
		InitialClustersProb(clusters, numberOfDocs);

		//Init Pik
		for (String word : wordsMap.keySet())
		{			
			boolean[] isWordInCluster = isWordInClusters(clusters,word);

			//for every doc t - has a list of probabilities (for each cluster i))
			Pik.put(word, InitialPikWithLidstone(isWordInCluster));
		}

	}

	/**
	 * Check for each cluster if the word occurs in it
	 * @param clusters
	 * @param word
	 * @return
	 */
	private boolean[] isWordInClusters(List<Cluster> clusters, String word) {
		boolean[] isWordInCluster = new boolean[NumOfClusters];
		for (int i = 0; i < NumOfClusters; i++)
		{				
			isWordInCluster[i] = clusters.get(i).hasWord(word); 
		}

		return isWordInCluster;
	}

	/**
	 * Smooth Pik with lidstone - using the number of clusturs the word occurs in
	 * ASUMING: we can pick any way to initialize them, and this works.
	 * @param isWordInCluster
	 * @return
	 */
	private Double[] InitialPikWithLidstone(boolean[] isWordInCluster) {		
		Double[] clusterProbForDoc = new Double[NumOfClusters];

		//count in how many clusters the word oocurs
		int numberOfClustersWithWord=0;
		for (boolean isInCluster : isWordInCluster){
			if (isInCluster){
				numberOfClustersWithWord++;
			}		
		}

		for (int i = 0; i < NumOfClusters; i++)
		{		
			if(isWordInCluster[i]){
				clusterProbForDoc[i] = CalcUnigramPLidstone(numberOfClustersWithWord , NumOfClusters);
			}
			else{
				clusterProbForDoc[i] = CalcUnigramPLidstone(NumOfClusters-numberOfClustersWithWord , NumOfClusters); 
			}

		}

		return clusterProbForDoc;

	}
	
	public double CalcUnigramPLidstone(long totalWordordOccurences, long trainSize) {
		//		C(X)+ LAMBDA / |S| + LAMBDA*|X|
		return (totalWordordOccurences + lidstonLambda)
				/ (trainSize + lidstonLambda * numberOfRelevantWords);
	}
	
	public double CalcUnigramPLidstone(Double totalWordordOccurences, Double trainSize) {
		//		C(X)+ LAMBDA / |S| + LAMBDA*|X|
		return (totalWordordOccurences + lidstonLambda)
				/ (trainSize + lidstonLambda * numberOfRelevantWords);
	}

	/**
	 * Calculates Ai (ASUMING: using initial Wti=1/NumOfClusters - uniform probability)  
	 * @param clusters
	 * @param docsListSize
	 */
	private void InitialClustersProb(List<Cluster> clusters, int docsListSize)
	{
		for (int i = 0; i < NumOfClusters; i++)
		{
			clustersProb[i] = clusters.get(i).documents.size() / (double)docsListSize;
		}
	}

	private double GetClassificationProb(Document doc, List<Cluster> clusters, int clusterIndex)
	{
		double numerator_prob = GetProbByCluster(doc, clusters, clusterIndex);
		double denominator_prob = GetProbForAllClusters(doc, clusters);

		return numerator_prob / denominator_prob;
	}

	/**
	 * Calculate sigma(Pik^Ntk) for doc t
	 * @param doc
	 * @param clusters
	 * @param clusterIndex
	 * @return
	 */
	private double GetProbByCluster(Document doc, List<Cluster> clusters, int clusterIndex)
	{
		double prob1 = 1;

		for (String word : doc.WordsMap.keySet())
		{
			double probForWordInCluster = GetNumOfOccursInCluster(clusterIndex, clusters, word);
			if (probForWordInCluster > 0)
			{
				prob1 *= Math.pow(probForWordInCluster,doc.WordsMap.get(word));
			}
		}

		return clustersProb[clusterIndex] * prob1;
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


	/**
	 * Calculate Pik for cluster i and word k
	 * ASUMING: using initial Wti=1/NumOfClusters - uniform probability, so they don't change Pik
	 * ASUMING: Pik is calculates only by the documents in the cluster. //TODO:check if right
	 * @param cluster
	 * @param word
	 * @return
	 */
	private double GetNumOfOccursInCluster(Cluster cluster, String word) 
	{
		int totalWordCountInClusterDocuments = 0;
		int wordCountInClusterDocuments = 0;

		for (Document doc : cluster.documents)
		{
			wordCountInClusterDocuments += doc.getWordOccurrences(word);
			totalWordCountInClusterDocuments += doc.getNumberOfRelevantWordsInDoc();
		}

		return wordCountInClusterDocuments / (double)totalWordCountInClusterDocuments;
	}

	private double GetNumOfOccursInCluster(int i, List<Cluster> clusters, String word) 
	{
		Cluster cluster = clusters.get(i);
		return GetNumOfOccursInCluster(cluster, word);
	}
}
