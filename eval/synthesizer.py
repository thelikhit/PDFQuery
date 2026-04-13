from deepeval.synthesizer import Synthesizer
from deepeval.models import AzureOpenAIModel
import pdfplumber
from dotenv import load_dotenv
import os

load_dotenv()

model = AzureOpenAIModel(
    model=os.getenv("AZURE_OPENAI_LANGUAGE_MODEL_NAME"),
    deployment_name=os.getenv("AZURE_OPENAI_LANGUAGE_MODEL_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0
)


chunks = []
with pdfplumber.open('synthesized_data/monopolyinstructions.pdf') as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text and text.strip():
            chunks.append(text.strip())

print(f"Extracted {len(chunks)} chunks")


synthesizer = Synthesizer(model=model)
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=[[chunk] for chunk in chunks],
    include_expected_output=True
)

print(goldens)

synthesizer.save_as(
    file_type='json',
    directory="",
    file_name="synthesized_dataset_monopoly"
)