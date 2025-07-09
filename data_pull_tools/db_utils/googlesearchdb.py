import json
from pymongo import MongoClient

uri="mongodb+srv://ayushsinghbasera:YEJTg3zhMwXJcTXm@cluster0.fmzrdga.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "application_db1"
COLLECTION_NAME = "applications1"

client = MongoClient(uri)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)


for app_block in data["targeting_keywords"]:
    application_name = app_block["application"]
    search_terms = app_block["google_search_terms"]
    matched_places = app_block.get("matched_places", [])


    doc = {
        "application": application_name,
        "search_terms": search_terms,
        "companies": []
    }

    for company in matched_places:
        company_info = {
            "name": company["displayName"]["text"],
            "address": company.get("formattedAddress"),
            "location": company.get("location"),
            "phone": {
                "national": company.get("nationalPhoneNumber"),
                "international": company.get("internationalPhoneNumber"),
            },
            "website": company.get("websiteUri"),
            "google_maps_url": company.get("googleMapsUri"),
            "rating": company.get("rating"),
            "user_rating_count": company.get("userRatingCount"),
            "types": company.get("types", []),
            "status": company.get("businessStatus")
        }
        doc["companies"].append(company_info)

    
    collection.update_one(
        {"application": application_name},
        {"$set": doc},
        upsert=True
    )

print(" Data successfully inserted/updated into MongoDB Atlas.")
