from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
AVIATIONEDGE_API_KEY = os.getenv("AVIATIONEDGE_API_KEY")

AVIATIONSTACK_BASE_URL = "http://api.aviationstack.com/v1"
AVIATIONEDGE_BASE_URL = "https://aviation-edge.com/v2/public"

if not AVIATIONSTACK_API_KEY:
    print("Warning: Missing AVIATIONSTACK_API_KEY in .env")
if not AVIATIONEDGE_API_KEY:
    print("Warning: Missing AVIATIONEDGE_API_KEY in .env")