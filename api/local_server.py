"""Local development server — runs api/recommend.py on port 8000."""
from http.server import HTTPServer
from recommend import handler

if __name__ == "__main__":
    server = HTTPServer(("localhost", 8000), handler)
    print("API running at http://localhost:8000")
    server.serve_forever()
