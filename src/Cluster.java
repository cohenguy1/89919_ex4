import java.util.ArrayList;
import java.util.List;

import InputOutput.Document;

public class Cluster 
{
	public List<Document> documents;

	public Cluster()
	{
		documents = new ArrayList<Document>();
	}
	
	public void AddDocument(Document document) 
	{
		documents.add(document);
	}

	public boolean hasWord(String word) {
		for (Document doc : documents){
			if (doc.hasWord(word))
				return true;
		}
		return false;
	}

}
