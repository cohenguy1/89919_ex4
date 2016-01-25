/* Ido Cohen	Guy Cohen	203516992	304840283 */
/* Ido Cohen	Guy Cohen	203516992	304840283 */
package InputOutput;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

public class DataClass 
{
	private static final int RAREÉ_WORD_OCCURS = 3;
	public static String UNSEEN_WORD = "unseen-word";
	private boolean skipLine = true;

	private List<Set<Topics>> docsTopicList;

	private List<Document> docsList;

	public Map<String, Integer> WordsMap = new TreeMap<String, Integer>();

	public DataClass(){
		this.docsTopicList = new ArrayList<Set<Topics>>();
		this.docsList = new ArrayList<Document>();
	}

	/*
	 * Parse the input file
	 */
	public void readInputFile(String inputFile) throws IOException
	{
		Output.writeConsoleWhenTrue(Output.folderPath+inputFile);

		FileReader fileReader = new FileReader(Output.folderPath+inputFile);
		BufferedReader bufferedReader = new BufferedReader(fileReader);

		String docTopicLine;
		int countDocs=0;
		Set<Topics> topics;

		while ((docTopicLine = bufferedReader.readLine()) != null) {
			countDocs++;
			
			topics = setTopicFromLine(docTopicLine);
			docsTopicList.add(topics);

			skipEmptyLine(bufferedReader);

			String[] docWords = bufferedReader.readLine().split(" ");
			
			Document doc = new Document(docWords,countDocs,topics);
			docsList.add(doc);
			
			mapWordCount(doc);
			
			skipEmptyLine(bufferedReader);
		}
		
		fileReader.close();
	}

	private void skipEmptyLine(BufferedReader bufferedReader) {
		try {
			if(skipLine)
				bufferedReader.readLine();
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
	public void removeRareWords()
	{
		Map<String, Integer> newWordsMap = new TreeMap<String, Integer>();
		List<String> wordsToRemove = new ArrayList<String>();
		
		for (String word : WordsMap.keySet())
		{
		
			if (WordsMap.get(word) > RAREÉ_WORD_OCCURS)
			{
				newWordsMap.put(word, WordsMap.get(word));
			}
			else
			{
				wordsToRemove.add(word);
			}
		}
		
		for (Document doc : docsList)
		{
			doc.removeWordsFromMap(wordsToRemove);
		}
		
		WordsMap = newWordsMap;
	}

	private Set<Topics> setTopicFromLine(String docTopicLine) {
		
		Set<Topics> topics = new TreeSet<Topics>();
		Topics topic;
		String[] lineNoTrainHeader = docTopicLine.replace("<TRAIN	", "").replace(">", "").split("\t");
		for (int i=1; i<lineNoTrainHeader.length; i++){
			topic = Topics.fromString(lineNoTrainHeader[i]); 
			topics.add(topic);
		}
		Output.writeConsoleWhenTrue(docTopicLine);
		Output.writeConsoleWhenTrue(topics);
		return topics;

	}

	/*
	 * Adds each word of the line read to the word mapping 
	 */
	private void mapWordCount(Document doc) 
	{	
		for(String word : doc.words)
		{
			AddWordToMap(WordsMap, word);
			
			doc.AddWordToMap(word);
		}
	}

	private void AddWordToMap(Map<String, Integer> wordsMap, String word)
	{
		word = word.toLowerCase();

		wordsMap.put(word, getWordOccurrences(wordsMap, word) + 1);
	}
	
	/*
	 * Returns the number of words in the map
	 */
	public static long wordsTotalAmount(Map<String, Map<String, Integer>> mapTotalDocsWords)
	{
		int count = 0;

		for(Map<String, Integer> map :  mapTotalDocsWords.values())
		{
			for (int value : map.values())
			{
				count += value;	
			}
		}

		return count;
	}

	/*
	 * Gets the Total number of occurrences of word in map
	 */
	public static int getWordOccurrences(Map<String, Integer> map, String word)
	{
		return map.get(word) == null ? 0 : map.get(word);
	}
	
	public List<Set<Topics>> getDocsTopicList() {
		return docsTopicList;
	}

	public void setDocsTopicList(List<Set<Topics>> docsTopicList) {
		this.docsTopicList = docsTopicList;
	}

	public List<Document> getDocsList() 
	{
		return docsList;
	}
}
