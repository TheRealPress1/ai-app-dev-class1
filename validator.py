"""
Session 7: Class Structure, Encapsulation, and Abstraction.

This module provides a class to validate user-submitted text against 
length and security requirements.
"""

import string

"""
STUDENT CHANGE LOG & AI DISCLOSURE:
----------------------------------
1. Did you use an LLM (ChatGPT/Claude/etc.)? No
2. If yes, what was your primary prompt? N/A
----------------------------------
"""


class InputValidator:
    """
    A class used to represent an input validation engine.

    Attributes:
        text (str): The sanitized (stripped) version of the input text.
    """

    DANGEROUS_KEYWORDS = ["SELECT", "DELETE", "INSERT", "UPDATE", "DROP", "--", ";"]

    def __init__(self, text):
        """
        Initializes the validator and performs basic encapsulation.

        Args:
            text (str): The raw string to be validated.
        """
        self.text = text.strip()

    def is_long_enough(self, min_chars=20):
        """
        Internal logic to verify the length of the string.

        Args:
            min_chars (int): The threshold for a valid string. Defaults to 20.

        Returns:
            bool: True if length is >= min_chars, False otherwise.
        """
        return len(self.text) >= min_chars

    def is_safe(self):
        """
        Internal logic to check for dangerous SQL keywords and punctuation.

        Security Criteria:
        1. No SQL keywords: SELECT, DELETE, INSERT, UPDATE, DROP, --, ;
        2. No characters from string.punctuation.

        Returns:
            bool: True if no dangerous elements are found, False otherwise.
        """
        upper_text = self.text.upper()
        for keyword in self.DANGEROUS_KEYWORDS:
            if keyword in upper_text:
                return False

        for char in self.text:
            if char in string.punctuation:
                return False

        return True

    def validate_all(self):
        """
        The Public Interface: Abstracts the internal validation logic.

        This method coordinates the execution of is_long_enough and is_safe.

        Returns:
            tuple: (bool, str)
                   - The bool indicates success/failure.
                   - The str provides the specific success or error message.
        """
        if not self.is_long_enough():
            return (False, "Error: Input is too short. Minimum 20 characters required.")

        if not self.is_safe():
            return (False, "Error: Input contains unsafe keywords or punctuation.")

        return (True, "Input validated successfully.")


if __name__ == "__main__":
    # Test 1: Fails due to a dangerous keyword (DROP)
    test1 = InputValidator("DROP TABLE users and all their data right now")
    print("Test 1 (dangerous keyword):", test1.validate_all())

    # Test 2: Fails due to being too short
    test2 = InputValidator("Hi there")
    print("Test 2 (too short):        ", test2.validate_all())

    # Test 3: Passes all validation checks
    test3 = InputValidator("Tell me about the history of artificial intelligence")
    print("Test 3 (valid input):      ", test3.validate_all())
