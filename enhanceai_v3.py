import os
import json
import logging
from typing import List, Optional, Union
from pydantic import BaseModel
import sympy as sp
from datetime import datetime

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Tools:
    MAX_MEMORY_SIZE = 100  # Limit for memory entries

    def __init__(self, memory_file: str = "memory.json", encryption: bool = False):
        self.citation = True
        self.memory_file = memory_file  # Path for the memory file
        self.memory: List[str] = []  # List to store chat history
        self.encryption = encryption  # Option for memory encryption
        self.load_memory()  # Load existing memory from the specified JSON file

    def message_handler(self, message: str) -> str:
        """
        Handle incoming messages and determine the appropriate action.

        :param message: The message from the user.
        :return: The response based on the message.
        """
        # Normalize the message for easier processing
        message = message.strip().lower()

        # Check for existing memory and respond appropriately
        response = ""

        command_mapping = {
            "calculate": self.calculator,
            "current date": self.get_current_date,
            "read memory": self.read_memory,
            "clear memory": self.clear_memory,
        }

        for command, action in command_mapping.items():
            if message.startswith(command):
                if command == "calculate":
                    equation = message[len(command) :].strip()  # Extract the equation
                    response = action(equation)
                else:
                    response = action()  # Call action without parameters
                break
        else:
            response = self.handle_unrecognized_message(message)

        # Reset to initial state if needed
        self.reset_state()
        return response

    def handle_unrecognized_message(self, message: str) -> str:
        """
        Handle unrecognized messages with context.

        :param message: The unrecognized message.
        :return: A contextual response.
        """
        if message in ["ok", "uh", "no way"]:
            return self.generate_response_based_on_input(message)
        return "I didn't understand that. Please try a different command."

    def generate_response_based_on_input(self, message: str) -> str:
        """
        Generate a contextual response based on specific inputs.

        :param message: The input message.
        :return: A contextual response.
        """
        responses = {
            "ok": "It seems we're back to square one! What would you like to talk about?",
            "uh": "You're still unsure about something, aren't you? Don't worry, I'm here to help. What's on your mind?",
            "no way": "I remember! Your response was just 'NO WAY.', which seemed a bit unusual, but we went with it!",
        }
        return responses.get(message, "I didn't understand that.")

    def calculator(self, equation: str) -> str:
        """
        Calculate the numeric result of an equation safely.

        :param equation: The equation to calculate.
        :return: The result of the equation or an error message.
        """
        try:
            expr = sp.sympify(equation)  # Parse the equation into a sympy expression
            result = expr.evalf()  # Evaluate the expression numerically
            result_str = f"{equation} = {result}"
            self.add_to_memory(result_str)  # Store the calculation result in memory
            return result_str
        except (sp.SympifyError, ValueError, TypeError) as e:
            logging.error(f"Error in calculator: {e}")
            return f"Invalid equation: {e}. Please provide a valid mathematical expression."

    def get_current_date(self) -> str:
        """
        Get the current date in a formatted string.

        :return: The current date as a formatted string.
        """
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        logging.info("Retrieved current date.")
        return f"Today's date is {current_date}"

    def add_to_memory(self, message: str) -> None:
        """
        Add a message to the memory and save it to a JSON file.

        :param message: The message to remember.
        """
        if len(self.memory) >= self.MAX_MEMORY_SIZE:
            self.memory.pop(0)  # Remove the oldest entry if memory exceeds the limit
        self.memory.append(message)
        self.save_memory()
        logging.info(f"Added message to memory: {message}")

    def read_memory(self) -> str:
        """
        Read all stored messages in memory.

        :return: A string representation of all messages.
        """
        return "\n".join(self.memory) if self.memory else "Memory is empty."

    def clear_memory(self) -> None:
        """
        Clear all stored messages in memory.
        """
        self.memory.clear()
        self.save_memory()
        logging.info("Memory has been cleared.")

    def save_memory(self) -> None:
        """
        Save the current memory to a JSON file, with optional encryption.
        """
        try:
            data = json.dumps(self.memory)
            if self.encryption:
                data = self._encrypt(data)  # Encrypt if encryption is enabled
            with open(self.memory_file, "w") as f:
                f.write(data)
            logging.info("Memory saved successfully.")
        except IOError as e:
            logging.error(f"Error saving memory: {e}")

    def load_memory(self) -> None:
        """
        Load memory from a JSON file if it exists.
        """
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    data = f.read()
                    if self.encryption:
                        data = self._decrypt(data)  # Decrypt if encryption is enabled
                    self.memory = json.loads(data)
                logging.info("Memory loaded successfully.")
            except (IOError, json.JSONDecodeError) as e:
                logging.error(f"Error loading memory: {e}")
                self.memory = []

    def reset_state(self) -> None:
        """
        Reset any necessary states or flags after processing a message.
        This can be expanded based on requirements.
        """
        # Reset any internal states if necessary
        pass

    def _encrypt(self, data: str) -> str:
        """
        Placeholder encryption method.
        :param data: The data to encrypt.
        :return: Encrypted data string.
        """
        # Implement encryption logic here
        return data

    def _decrypt(self, data: str) -> str:
        """
        Placeholder decryption method.
        :param data: The data to decrypt.
        :return: Decrypted data string.
        """
        # Implement decryption logic here
        return data


# Example usage
if __name__ == "__main__":
    tools = Tools(encryption=True)

    # Simulated user inputs
    user_inputs = [
        "Hello.",
        "What is the date today?",
        "Nice.",
        "Uh.",
        "What is 94350349578 + 3450734075?",
        "WOAH.",
        "How did you do that well?",
        "What is that provided context?",
        "Oh. A calculator context.",
        "Ok.",
        "So uh. What is the date today?",
        "And you also remember our convo.",
        "What's the first message I made?",
        "What was your reply?",
        "After that?",
        "NO WAY.",
    ]

    for user_input in user_inputs:
        response = tools.message_handler(user_input)
        print(response)

    print("\nChat History:")
    print(tools.read_memory())
