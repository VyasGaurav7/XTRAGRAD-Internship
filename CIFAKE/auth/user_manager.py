"""
CIFAKE Authentication and User Management Module
Implements user registration, credential verification, and session management.
Matches requirements from Figure 3 (Registration) and Figure 4 (Login).
"""

import os
import json
import hashlib
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")


def _hash_password(password: str) -> str:
    """Hash password with SHA-256 for secure local storage."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _ensure_user_store():
    """Ensure data directory and users.json exist with default demo credentials."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        default_users = {
            "admin": {
                "username": "admin",
                "name": "Administrator",
                "email": "admin@cifake.ai",
                "mobile": "+1-800-555-0199",
                "password_hash": _hash_password("admin123")
            },
            "researcher": {
                "username": "researcher",
                "name": "Dr. AI Forensics",
                "email": "research@cifake.org",
                "mobile": "+1-555-432-8765",
                "password_hash": _hash_password("cifake2024")
            }
        }
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=2)


def load_users() -> dict:
    """Load existing users from the JSON store."""
    _ensure_user_store()
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def register_user(username, name, email, mobile, password):
    """
    Registers a new user matching Figure 3 specifications.
    Returns (success: bool, message: str).
    """
    username = username.strip().lower()
    name = name.strip()
    email = email.strip()
    mobile = mobile.strip()

    if not username or not name or not email or not mobile or not password:
        return False, "All fields are required. Please fill in all registration inputs."

    if len(username) < 3:
        return False, "Username must be at least 3 characters long."

    email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not re.match(email_regex, email):
        return False, "Invalid email address format (e.g., user@example.com)."

    if len(mobile) < 7:
        return False, "Please enter a valid mobile phone number."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    users = load_users()
    if username in users:
        return False, f"Username '{username}' is already registered. Please choose another username."

    users[username] = {
        "username": username,
        "name": name,
        "email": email,
        "mobile": mobile,
        "password_hash": _hash_password(password)
    }

    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
        return True, "Registration successful! You can now log in with your credentials."
    except Exception as e:
        return False, f"Failed to save user record: {str(e)}"


def authenticate_user(username, password):
    """
    Authenticates user credentials matching Figure 4 specifications.
    Returns (success: bool, user_dict_or_error: dict/str).
    """
    username = username.strip().lower()
    users = load_users()

    if username not in users:
        return False, "Username not found. Please register an account first."

    user = users[username]
    if user["password_hash"] == _hash_password(password):
        return True, user
    else:
        return False, "Incorrect password. Please try again."
