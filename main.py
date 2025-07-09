#all the main code and pipeline has to be merged compile
import uvicorn
from api import app  # This app includes everything: front_end_llm + backend trigger

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8006)
