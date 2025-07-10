# front_end_llm/front_end_llm.py
from typing import List, Dict
from openai import OpenAI
from pymongo.collection import Collection

# Import models
from front_end_llm.pydantic_models import AskInput, Message

# Import prompts from the canonical prompts.py
from front_end_llm.prompts import SYSTEM_PROMPT, RETRY_PROMPT_SUFFIX, NEXT_QUESTION_PROMPT # NEXT_QUESTION_PROMPT is here

# Import LLM-specific utilities
from front_end_llm.shared import is_forbidden, is_duplicate, get_chat_messages, get_qa_history_for_llm

def _generate_response_from_llm(
    openai_client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float = 0.1
) -> str:
    """Internal helper to call the OpenAI API."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", # Or whatever model you prefer
            messages=messages,
            max_tokens=30, # Keep this to enforce conciseness
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise

async def ask_openai(
    user_message_content: str,
    chat_id: str,
    user_id: str,
    openai_client: OpenAI,
    messages_collection: Collection
) -> str:
    history_for_llm = get_qa_history_for_llm(messages_collection, chat_id)
    current_chat_messages_models = get_chat_messages(messages_collection, chat_id)

    # 1. Combine SYSTEM_PROMPT and NEXT_QUESTION_PROMPT for the primary system instruction.
    # This creates a comprehensive set of rules for the AI's behavior.
    full_system_instruction = SYSTEM_PROMPT + "\n\n" + NEXT_QUESTION_PROMPT

    # 2. Construct the messages list for the LLM call
    messages_to_send = [
        {"role": "system", "content": full_system_instruction}
    ]
    
    # 3. Add all prior turns (history)
    messages_to_send.extend(history_for_llm)

    # 4. Add the current user's message as the final piece of context for the LLM to respond to.
    # The LLM is expected to respond to *this* message by asking the *next* question.
    messages_to_send.append({"role": "user", "content": user_message_content})


    print("\n--- Messages sent to OpenAI (Initial Attempt) ---")
    for msg in messages_to_send:
        print(f"Role: {msg['role']}, Content: {msg['content'][:200]}...") # Print more chars for full_system_instruction
    print("---------------------------------------------------\n")

    # First attempt to generate question
    question = _generate_response_from_llm(openai_client, messages_to_send, temperature=0.1)

    # Check for forbidden or duplicate questions and retry if necessary
    if is_forbidden(question) or is_duplicate(question, current_chat_messages_models):
        # ⭐ For retry, reinforce SYSTEM_PROMPT and add RETRY_PROMPT_SUFFIX.
        # It's crucial to give the LLM ALL the context again, plus the retry specific instruction.
        retry_system_instruction = SYSTEM_PROMPT + "\n\n" + RETRY_PROMPT_SUFFIX + "\n\n" + NEXT_QUESTION_PROMPT # Reinforce all instructions

        retry_messages = [
            {"role": "system", "content": retry_system_instruction} # Stronger system prompt for retry
        ]
        retry_messages.extend(history_for_llm) # Original history including last user message
        retry_messages.append({"role": "user", "content": user_message_content}) # Re-add the last user message

        # Add a clear, concise instruction for the retry attempt, telling it what to output.
        retry_messages.append({"role": "user", "content": "Your previous response was invalid. Please generate ONLY the next most relevant question from the APPROVED LIST. Do NOT add any conversational preambles or explanations. Just the question."})


        print("\n--- Messages sent to OpenAI (RETRY Attempt) ---")
        for msg in retry_messages:
            print(f"Role: {msg['role']}, Content: {msg['content'][:200]}...")
        print("------------------------------------------------\n")

        question = _generate_response_from_llm(openai_client, retry_messages, temperature=0.1)

        # Final check after retry
        if is_forbidden(question) or is_duplicate(question, current_chat_messages_models):
            question = "Thank you. That’s all the questions we needed for now. If you have more information, feel free to share."

    return question