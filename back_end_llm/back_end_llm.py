import logging
from back_end_llm.prompts import get_application_extraction_prompt, get_google_search_prompt
from back_end_llm.utils import (
    fetch_latest_session_from_mongo,
    json_to_chatml,
    extract_user_location,
    get_lat_lng_from_location,
    search_google_places,
    get_mongo_collection
)
from back_end_llm.pydantic_models import (
    ConversationLog, PredictionResult, SearchQueryEntry, SearchQueryResults, Place, SearchTerms
)
from pydantic_ai import Agent


def main():
    conversation_entries = fetch_latest_session_from_mongo()
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
    # print(applications)
    search_results = []
    for app in applications:
        search_terms = []
        PROMPT = get_google_search_prompt(app)
        try:
            search_result = agent.run_sync(PROMPT, output_type=SearchTerms)
            # print(SearchTerms)
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

    print("📝 Results saved to: output.json")

    # Insert into MongoDB
    collection = get_mongo_collection()
    for entry in final_output.targeting_keywords:
        doc = {
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

        collection.update_one(
            {"application": entry.application},
            {"$set": doc},
            upsert=True
        )

    print(" Data successfully inserted/updated into MongoDB Atlas.")



