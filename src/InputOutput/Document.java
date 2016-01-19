package InputOutput;

import java.util.*;

public class Document implements Comparable<Document>
{
	public int id;

	public String[] words; //should be private - check TODO in Data Class

	public Map<String, Integer> WordsMap;

	Set<Topics> topics;

	public Document(String[] docContent, int docId, Set<Topics> topics)
	{
		id = docId;
		words = docContent;
		WordsMap = new TreeMap<String, Integer>();
		this.topics = topics;
	}

	public void AddWordToMap(String word)
	{
		word = word.toLowerCase();

		WordsMap.put(word, getWordOccurrences(word) + 1);
	}

	/*
	 * Gets the Total number of occurrences of word in map
	 */
	public int getWordOccurrences(String word)
	{
		return WordsMap.get(word) == null ? 0 : WordsMap.get(word);
	}

	public void removeWordsFromMap(List<String> wordsToRemove) 
	{
		for (String word : wordsToRemove)
		{
			WordsMap.remove(word);			
		}
	}

	public boolean hasWord(String word) {
		return getWordOccurrences(word)>0;
	}

	public long getNumberOfRelevantWordsInDoc(){
		long sum=0;
		for (String word : WordsMap.keySet())
		{
			sum += WordsMap.get(word);		
		} 
		return sum;

	}

	public void setTopics(Set<Topics> topics) {
		this.topics = topics;
	}

	public Set<Topics> getTopics() {
		return topics;
	}

	@Override
	public int compareTo(Document oDoc) {
		return id-oDoc.id;
	}
}
