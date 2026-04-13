import os

from deepeval.models import AzureOpenAIModel
from dotenv import load_dotenv
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
  FaithfulnessMetric,
  ContextualRelevancyMetric,
)
from deepeval import evaluate
from src.app.core.rag_service import rag

load_dotenv()

model = AzureOpenAIModel(
    model=os.getenv("AZURE_OPENAI_LANGUAGE_MODEL_NAME"),
    deployment_name=os.getenv("AZURE_OPENAI_LANGUAGE_MODEL_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0
)

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(
    file_path=r"D:\ProgramFiles\PycharmProjects\PDFQuery\eval\synthesized_data\synthesized_dataset_monopoly_short.json",
    input_key_name="input",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    context_key_name="context",
)

test_cases = []

for golden in dataset.goldens:
    res, text_chunks = rag(golden.input, return_context=True)
    test_case = LLMTestCase(input=golden.input, actual_output=res, retrieval_context=text_chunks)
    test_cases.append(test_case)

evaluate(
    test_cases=test_cases,
    metrics=[
        ContextualRelevancyMetric(
            model=model,
        ),
        FaithfulnessMetric(
            model=model,
        ),
    ]
)
