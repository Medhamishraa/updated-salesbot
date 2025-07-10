import logging
from uuid import uuid4
from back_end_llm.prompts import get_application_extraction_prompt, get_google_search_prompt
from back_end_llm.utils import (
    fetch_latest_session_from_mongo,
    json_to_chatml,
    extract_user_location,
    get_lat_lng_from_location,
    search_google_places,
    get_mongo_collection, get_write_collection
)
from back_end_llm.pydantic_models import (
    ConversationLog, PredictionResult, SearchQueryEntry, SearchQueryResults, Place, SearchTerms
)
from pydantic_ai import Agent
from datetime import datetime


def main(user_id: str, session_uuid: str, chat_id: str):
    conversation_entries = fetch_latest_session_from_mongo(session_uuid, user_id, chat_id)
    if not conversation_entries:
        print("No valid session or QA items found.")
        return

    conv_log = ConversationLog(conversation=conversation_entries)
    chatml_conversation = json_to_chatml(conv_log)

    user_location = extract_user_location(conversation_entries)
    coords = get_lat_lng_from_location(user_location) if user_location else None
    if coords:
        logging.info(f"User location: {user_location} → {coords}")

    agent = Agent("openai:gpt-3.5-turbo")

    result = agent.run_sync(get_application_extraction_prompt(chatml_conversation), output_type=PredictionResult)
    applications = result.output.predicted_interests

    search_results = []
    for app in applications:
        search_terms = []
        PROMPT = get_google_search_prompt(app)
        try:
            search_result = agent.run_sync(PROMPT, output_type=SearchTerms)
            search_terms = search_result.output.search_terms
        except Exception as e:
            logging.error("Search term generation failed for '%s': %s", app, str(e))

        all_places = []
        final_status = "ZERO_RESULTS"
        for term in search_terms:
            places, status = search_google_places(term, location=coords)
            if status == "OK" and places:
                final_status = "OK"
            elif status == "ERROR":
                final_status = "ERROR"
            all_places.extend(places)

        unique_places = {
            p.get("id"): p
            for p in all_places
            if p.get("businessStatus") != "CLOSED_PERMANENTLY" and p.get("id")
        }

        search_results.append(SearchQueryEntry(
            application=app,
            google_search_terms=search_terms,
            matched_places=[Place(**place) for place in unique_places.values()],
            status=final_status
        ))

    final_output = SearchQueryResults(
        extracted_applications=applications,
        targeting_keywords=search_results
    )

    # Save result to fixed file name
    with open('output.json', 'w', encoding='utf-8') as f:
        f.write(final_output.model_dump_json(indent=2))

    print(" Results saved to: output.json")

    # -----------------  MongoDB Storage ------------------

    # Dummy session + user IDs (replace with real ones if needed)
        # -----------------  MongoDB Storage with Provided Chat ID ------------------


    # Prepare messages from conversation
    messages = [
        {
            "question": entry.question,
            "answer": entry.answer
        }
        for entry in conversation_entries
    ]

    # Format the output for this chat
    output_data = [
        {
            "application": entry.application,
            "search_terms": entry.google_search_terms,
            "companies": [
                {
                    "name": p.displayName.text if p.displayName else None,
                    "address": p.formattedAddress,
                    "location": {
                        "latitude": p.location.latitude if p.location else None,
                        "longitude": p.location.longitude if p.location else None
                    },
                    "phone": {
                        "national": p.nationalPhoneNumber,
                        "international": p.internationalPhoneNumber
                    },
                    "website": p.websiteURL,
                    "google_maps_url": p.googleMapsURL,
                    "rating": p.rating,
                    "user_rating_count": p.userRatingCount,
                    "types": p.types or [],
                    "status": p.businessStatus
                } for p in entry.matched_places
            ]
        }
        for entry in final_output.targeting_keywords
    ]

    # Compose and upsert the full document with this chat_id

    collection = get_write_collection()

    collection.update_one(
        {"session_uuid": session_uuid, "userId": user_id},
        {
            "$set": {
                f"chats.{chat_id}.output": output_data
            }
        },
        upsert=True
    )


    print(f" Chat data inserted under chat ID '{chat_id}' in MongoDB.")

if __name__ == "__main__":
    main()

