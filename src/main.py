import uuid
from chatbot_logic.graph import app

from dotenv import load_dotenv
load_dotenv()

# Database initialization (if required)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id,
    }
}

def run_chatbot():
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        try:
            events = app.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values")
            for event in events:
                for value in event.values():
                    print("Assistant:", value["messages"][-1]["content"])
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_chatbot()