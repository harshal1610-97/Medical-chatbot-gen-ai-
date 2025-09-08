# Medical-chatbot-gen-ai-


# How to run?
### STEPS:

Clone the repository

'''bash
project repo: https://github.com/harshal1610-97/Medical-chatbot-gen-ai-.git
'''

### STEP 01- Create a conda environment after opening the repository

'''
conda create -n medibot python=3.10

'''bash
conda activate medibot

### STEP 02- install the requirements
'''bash
pip install -r requirements.txt
'''

### Create a '.env' file in the root directory and add your Pinecone and Generative ai credentials as follows:

'''ini
PINECONE_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GOOGLE_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
'''

'''bash
# run the following command to store embeddings to pinecone 
python store_index.py
'''

'''bash
#Finally run the following command 
python app.py
'''

Now
'''bash
open up localhost:
'''


### Techstack Used:

- Python
- Langchain
-Flask
- Generative ai
- Pinecone
