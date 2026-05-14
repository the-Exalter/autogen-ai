import os
import re

from pymongo import MongoClient
from pymongo.errors import PyMongoError


def get_rag_context(make: str, year: int | None) -> str:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        return ""

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client.get_default_database()
        collection = db["vehicles"]

        query: dict = {
            "make": {"$regex": f"^{re.escape(make)}$", "$options": "i"},
            "source": "db",
        }
        if year is not None:
            query["year"] = {"$gte": year - 3, "$lte": year + 3}

        projection = {
            "_id": 0,
            "make": 1,
            "model": 1,
            "year": 1,
            "price_pkr": 1,
            "body_type": 1,
            "engine_capacity": 1,
        }

        docs = list(collection.find(query, projection).limit(5))
        client.close()

        if not docs:
            return ""

        lines = ["Similar vehicles from database:"]
        for d in docs:
            lines.append(
                f"- {d.get('make', '')} {d.get('model', '')} {d.get('year', '')}, "
                f"{d.get('body_type', '')}, {d.get('engine_capacity', '')}, "
                f"PKR {d.get('price_pkr', '')}"
            )
        return "\n".join(lines)
    except PyMongoError:
        return ""
    except Exception:
        return ""
