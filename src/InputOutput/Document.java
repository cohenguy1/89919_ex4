package InputOutput;

import java.util.*;

public class Document 
{
	public String[] words;
	
	public Map<String, Integer> WordsMap;
	
	public Document(String[] docContent)
	{
		words = docContent;
		WordsMap = new TreeMap<String, Integer>();
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
}
