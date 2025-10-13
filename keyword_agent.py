import json

from Agent import Agent


class KeywordAgent(Agent):
    def __init__(self):
        super().__init__(
system_prompt="""You are a Keyword Extraction Agent specialized in analyzing research ideas and generating optimal search keywords for academic paper discovery on arXiv.
Your task is to:
1. Analyze the user's research idea, project description, or paper concept
2. Extract the most essential and relevant keywords that will effectively find related papers
3. Focus on technical terms, methodologies, and core concepts
4. Include synonyms and alternative phrasings that researchers might use
5. Prioritize keywords that balance specificity with coverage

Output Requirements:
- Return ONLY valid JSON with no additional text or explanation : 
dont use any - signs just words that are separated by a space
return 5 related keywords

{
  "keywords": ["python list of 5 most essential search terms"]
}

Guidelines:
- Be precise and comprehensive
- Include both technical jargon and common terminology
- Consider how researchers phrase concepts in academic papers
- Balance broad terms with specific techniques
- Ensure keywords are suitable for arXiv searches
- Prioritize terms that capture the core innovation or focus

Remember: Output ONLY the JSON object, no preamble or explanation.""",temperature=0.3,top_p=0.85,top_k=40,response_mime_type='application/json',create_chat=False)


    def generate_response(self, prompt):
        response=self.generate_text_generation_response(prompt)
        # print(response.text)
        response_json=json.loads(response.text)
        keyword_list=response_json['keywords']
        return keyword_list


# if __name__ == '__main__':
#     keyworda=KeywordAgent()
#     response=keyworda.generate_response('''I want to develop a multimodal retrieval-augmented generation (RAG) system that can process and reason over both text documents and images simultaneously. The idea is to use vision-language models to extract semantic information from diagrams, charts, and figures in scientific papers, and then integrate this visual understanding with the textual content for more comprehensive question-answering. I'm particularly interested in applying this to medical literature where visual data like X-rays, MRI scans, and anatomical diagrams are crucial for understanding. The system should be able to answer complex queries that require correlating information from both text passages and medical images, potentially using cross-modal attention mechanisms. I'm also curious about efficient indexing strategies for this kind of multimodal data and how to handle cases where the visual and textual information might be contradictory.''')
#     print(response)