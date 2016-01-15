package InputOutput;

import java.util.*;

public class Document 
{
	public String content;
	
	public Map<String, Integer> WordsMap;
	
	public Document(String docContent)
	{
		content = docContent;
		WordsMap = new TreeMap<String, Integer>();
	}

	public void AddWordToMap(String word)
	{
		word = word.toLowerCase();

		WordsMap.put(word, getWordOccurrences(WordsMap, word) + 1);
	}
	
	/*
	 * Gets the Total number of occurrences of word in map
	 */
	public static int getWordOccurrences(Map<String, Integer> map, String word)
	{
		return map.get(word) == null ? 0 : map.get(word);
	}
	
	public void removeWordsFromMap(List<String> wordsToRemove) 
	{
		for (String word : wordsToRemove)
		{
			WordsMap.remove(word);
		}
	}
}
