import json

from Agent import Agent
from heading_extractor import HeadingExtractor


class HeadingSelectorAgent(Agent):
    def __init__(self):
        super().__init__(system_prompt='''
## Role
You are a specialized Heading Selector Agent designed to identify the most relevant section headings within academic papers based on limited metadata (title, abstract, and headings list).

## Input
You will receive:
1. **User's Research Idea**: A description of the research concept, question, or hypothesis the user is exploring
2. **Paper Title**: The title of the academic paper
3. **Paper Abstract**: A summary of the paper's objectives, methods, and findings
4. **List of Headings**: All section headings from the paper (without the actual content)
5. **Selection Target**: Select exactly 3 heading intervals
## Task
Based solely on the title, abstract, and heading names, identify the 3 headings most likely to contain content that would help determine if the user's research idea is present in or related to this paper.    
Do not Select Conclusion or Acknowledgement when selecting headings     
## Selection Strategy

### Step 1: Analyze the Context
- Review the user's research idea to identify key concepts, methods, and domains
- Read the paper's title and abstract to understand the paper's scope and focus
- Assess the relevance match between the user's idea and the paper        
  ### Step 2: Select Heading intervals Based On
1. **Semantic Alignment**: Headings containing keywords or concepts from the user's research idea
2. **Core Content Sections**: Headings likely to contain substantive research content:
   - Methods/Methodology/Approach
   - Results/Findings/Experiments
   - Discussion/Analysis
   - Related Work/Literature Review
   - Specific technical sections named after key concepts
3. **Abstract Clues**: Headings that correspond to topics or methods mentioned in the abstract
4. **Research Depth**: Sections where novel contributions or detailed exploration typically appear
5. ** Heading Type**: Only select main headings like 4. EXPERIMENTS etc dont select subheadings and write only the headings name exactly like given to you
If it is 4. EXPERIMENTS you will return it as EXPERIMENTS
### Step 3: Avoid
- Administrative sections: Acknowledgments, Funding, Author Contributions, Declarations
- Generic/vague sections: Introduction (unless highly specific), Conclusion (unless critical)
- Sections clearly outside the scope of the user's idea


## Output Format
Provide your response as a JSON object with exactly 3 selected heading intervals:
```json
{
  "selected_intervals": [
    {
      "from_heading": "METHODOLOGY",
      "to_heading": "EXPERIMENTAL SETUP",
    },
    {
      "from_heading": "EXPERIMENTAL SETUP",
      "to_heading": "RESULTS",
    },
    {
      "from_heading": "RESULTS",
      "to_heading": "DISCUSSION",
    }
  ]
}
##EXAMPLE OUTPUT
{
  "selected_intervals": [
    {
      "from_heading": "METHODOLOGY",
      "to_heading": "EXPERIMENTAL SETUP",
      "rationale": "Contains detailed description of the proposed approach directly relevant to the user's research methods"
    },
    {
      "from_heading": "EXPERIMENTAL SETUP",
      "to_heading": "RESULTS",
      "rationale": "Describes experimental configuration and parameters that align with the user's research design"
    },
    {
      "from_heading": "RESULTS",
      "to_heading": "DISCUSSION",
      "rationale": "Presents empirical findings that validate techniques related to the research idea"
    }
  ]
}


        ''',top_p=0.85,top_k=40,temperature=0.1,response_mime_type='application/json',create_chat=False)

    def generate_response(self,users_idea,headings,abstract):
        response=self.generate_text_generation_response('users idea:'+f'{users_idea}'+'abstract'+f'{abstract}'+'headings'+f'{headings}')
        return json.loads(response.text)




if __name__ == '__main__':
     extractor = HeadingExtractor()
     source = "/Users/harun/Documents/GitHub/Hypothetica/1708.04776v1.pdf"
     markdown = extractor.convert_to_markdown(source)
     headings = extractor.extract_headings(markdown)
     headings_json = extractor.get_headings_json(headings)
     abstract="""
    Nowadays, cross-modal retrieval plays an indispens-
able role to flexibly find information across different modalities
of data. Effectively measuring the similarity between different
modalities of data is the key of cross-modal retrieval. Differ-
ent modalities such as image and text have imbalanced and
complementary relationships, which contain unequal amount of
information when describing the same semantics. For example,
images often contain more details that cannot be demonstrated
by textual descriptions and vice versa. Existing works based on
Deep Neural Network (DNN) mostly construct one common space
for different modalities to find the latent alignments between
them, which lose their exclusive modality-specific characteristics.
Different from the existing works, we propose modality-specific
cross-modal similarity measurement (MCSM) approach by con-
structing independent semantic space for each modality, which
adopts end-to-end framework to directly generate modality-
specific cross-modal similarity without explicit common represen-
tation. For each semantic space, modality-specific characteristics
within one modality are fully exploited by recurrent attention
network, while the data of another modality is projected into
this space with attention based joint embedding to utilize the
learned attention weights for guiding the fine-grained cross-
modal correlation learning, which can capture the imbalanced
and complementary relationships between different modalities.
Finally, the complementarity between the semantic spaces for dif-
ferent modalities is explored by adaptive fusion of the modality-
specific cross-modal similarities to perform cross-modal retrieval.
Experiments on the widely-used Wikipedia and Pascal Sentence
datasets as well as our constructed large-scale XMediaNet dataset
verify the effectiveness of our proposed approach, outperforming
9 state-of-the-art methods.
     
     
     """
     project_idea="""
     Project: I want to do a Multi-Modal Medical Image-Report Retrieval System
Develop a retrieval system that allows healthcare professionals to search for medical images (X-rays, CT scans, MRI) 
using text queries or find relevant clinical reports using image queries.
 Following the paper's modality-specific approach, the system would construct independent semantic spaces for medical 
 images and clinical reports, using recurrent attention networks to focus on pathological regions in images and critical diagnostic keywords in text reports.
  The attention mechanism would capture the imbalanced relationship between modalities—where
   images show visual symptoms but reports contain additional patient history and lab values—and use
    adaptive fusion to combine both semantic spaces for accurate retrieval. This would help radiologists find similar cases, 
    enable second opinions by retrieving relevant past diagnoses, and assist in clinical decision-making by bridging 
    visual and textual medical data.
     
     
     
     """
     agent=HeadingSelectorAgent()
     print(agent.generate_response(project_idea,headings_json,abstract))



