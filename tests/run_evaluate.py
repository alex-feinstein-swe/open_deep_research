from dotenv import load_dotenv
load_dotenv()  # Load .env from project root before any other imports

from langsmith import Client
from tests.evaluators import eval_overall_quality, eval_relevance, eval_structure, eval_groundedness, eval_completeness
import asyncio
import time
import argparse
from typing import Optional, List
from open_deep_research.deep_researcher import deep_researcher_builder
from langgraph.checkpoint.memory import MemorySaver
import uuid


client = Client()

# NOTE: Configure the right dataset and evaluators
dataset_name = "Deep Research Bench"
evaluators = [eval_overall_quality, eval_relevance, eval_structure, eval_groundedness, eval_completeness]
# NOTE: Configure the right parameters for the experiment, these will be logged in the metadata
max_structured_output_retries = 3
allow_clarification = False
max_concurrent_research_units = 10
search_api = "youdeepsearch"
max_researcher_iterations = 6
max_react_tool_calls = 10
summarization_model = "openai:gpt-4.1-mini"
summarization_model_max_tokens = 8192
research_model = "openai:gpt-5" # "anthropic:claude-sonnet-4-20250514"
research_model_max_tokens = 10000
compression_model = "openai:gpt-4.1"
compression_model_max_tokens = 10000
final_report_model = "openai:gpt-4.1"
final_report_model_max_tokens = 10000

async def target(
    inputs: dict,
):
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    # NOTE: Configure the right dataset and evaluators
    config["configurable"]["max_structured_output_retries"] = max_structured_output_retries
    config["configurable"]["allow_clarification"] = allow_clarification
    config["configurable"]["max_concurrent_research_units"] = max_concurrent_research_units
    config["configurable"]["search_api"] = search_api
    config["configurable"]["max_researcher_iterations"] = max_researcher_iterations
    config["configurable"]["max_react_tool_calls"] = max_react_tool_calls
    config["configurable"]["summarization_model"] = summarization_model
    config["configurable"]["summarization_model_max_tokens"] = summarization_model_max_tokens
    config["configurable"]["research_model"] = research_model
    config["configurable"]["research_model_max_tokens"] = research_model_max_tokens
    config["configurable"]["compression_model"] = compression_model
    config["configurable"]["compression_model_max_tokens"] = compression_model_max_tokens
    config["configurable"]["final_report_model"] = final_report_model
    config["configurable"]["final_report_model_max_tokens"] = final_report_model_max_tokens
    # NOTE: We do not use MCP tools to stay consistent

    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["input"]}]},
        config
    )
    return final_state

def get_evaluation_data(
    client: Client,
    dataset_name: str,
    item_ids: Optional[List[str]] = None
):
    """
    Get evaluation data, optionally filtered by LangSmith example IDs.
    
    Args:
        client: LangSmith client instance
        dataset_name: Name of the dataset to evaluate
        item_ids: Optional list of LangSmith example IDs to filter. If None, returns dataset name for full evaluation.
    
    Returns:
        Either a list of filtered Example objects or the dataset name string.
    
    Raises:
        ValueError: If item_ids are provided but no matching examples are found.
    """
    if item_ids is None:
        return dataset_name
    
    # Fetch the dataset to get examples
    dataset = client.read_dataset(dataset_name=dataset_name)
    
    # Get all examples from the dataset
    examples_gen = client.list_examples(dataset_id=dataset.id)
    all_examples = list(examples_gen)
    
    # Filter examples by LangSmith example IDs only
    item_ids_str = [str(id) for id in item_ids]
    filtered_examples = [ex for ex in all_examples if str(ex.id) in item_ids_str]
    
    if not filtered_examples:
        raise ValueError(f"No examples found matching the provided LangSmith example IDs: {item_ids}")
    
    print(f"Filtered to {len(filtered_examples)} examples from {len(all_examples)} total")
    return filtered_examples

async def main(item_ids: Optional[List[str]] = None):
    data = get_evaluation_data(client, dataset_name, item_ids)
    
    return await client.aevaluate(
        target,
        data=data,
        evaluators=evaluators,
        experiment_prefix=f"ODR GPT-5, You Deep Search",
        max_concurrency=10,
        metadata={
            "max_structured_output_retries": max_structured_output_retries,
            "allow_clarification": allow_clarification,
            "max_concurrent_research_units": max_concurrent_research_units,
            "search_api": search_api,
            "max_researcher_iterations": max_researcher_iterations,
            "max_react_tool_calls": max_react_tool_calls,
            "summarization_model": summarization_model,
            "summarization_model_max_tokens": summarization_model_max_tokens,
            "research_model": research_model,
            "research_model_max_tokens": research_model_max_tokens,
            "compression_model": compression_model,
            "compression_model_max_tokens": compression_model_max_tokens,
            "final_report_model": final_report_model,
            "final_report_model_max_tokens": final_report_model_max_tokens,
        }
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run evaluation on Deep Research Bench dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run on entire dataset
            python tests/run_evaluate.py
            
            # Run on specific LangSmith example IDs
            python tests/run_evaluate.py --item-ids abc123 def456
        """
    )
    parser.add_argument(
        '--item-ids',
        nargs='+',
        help='Specific LangSmith example IDs to evaluate (space-separated).'
    )
    args = parser.parse_args()
    
    start_time = time.perf_counter()
    results = asyncio.run(main(item_ids=args.item_ids))
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    print(results)
    print(f"\nTotal execution time: {elapsed_seconds:.2f} seconds")