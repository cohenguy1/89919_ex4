/* Ido Cohen	Guy Cohen	203516992	304840283 */
import java.util.*;

import InputOutput.Document;

public class Cluster 
{
	public List<Document> documents;

	public Cluster()
	{
		documents = new ArrayList<Document>();
	}
	
	public List<Document> getDocuments() {
		return documents;
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
