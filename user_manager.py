"""
user_manager.py — User Enrollment Manager

Handles saving and loading enrolled user data:
  - Voice embeddings stored as .pt (PyTorch tensor) files
  - User metadata stored in a JSON file

Data is stored in the 'enrolled_users/' folder.
"""

import os
import json
import torch

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ENROLLED_DIR = "enrolled_users"
USERS_FILE = os.path.join(ENROLLED_DIR, "users.json")


# ──────────────────────────────────────────────
# Step 1: Ensure the Storage Directory Exists
# ──────────────────────────────────────────────
def _ensure_dir():
    """Create the enrolled_users directory if it doesn't exist."""
    os.makedirs(ENROLLED_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Step 2: Enroll a New User
# ──────────────────────────────────────────────
def enroll_user(name, embedding, passphrase):
    """
    Save a user's name, voice embedding, and custom passphrase to disk.

    Args:
        name (str): User's name
        embedding (torch.Tensor): Voice embedding vector
        passphrase (str): User's custom passphrase
    """
    _ensure_dir()

    # Save embedding as a .pt file
    embedding_path = os.path.join(ENROLLED_DIR, f"{name}.pt")
    torch.save(embedding, embedding_path)

    # Update the users JSON file
    users = _load_users_json()
    users[name] = {
        "embedding_file": f"{name}.pt",
        "passphrase": passphrase.lower().strip()
    }
    _save_users_json(users)


def get_user_passphrase(name):
    """
    Get the enrolled passphrase for a specific user.

    Args:
        name (str): User's name

    Returns:
        str: Enrolled passphrase, or None if not found
    """
    users = _load_users_json()
    if name in users:
        return users[name].get("passphrase", "")
    return None



# ──────────────────────────────────────────────
# Step 3: Get All Enrolled Users (with embeddings)
# ──────────────────────────────────────────────
def get_enrolled_users():
    """
    Load all enrolled users and their embeddings from disk.

    Returns:
        dict: {name: embedding_tensor, ...}
    """
    users = _load_users_json()
    enrolled = {}

    for name, info in users.items():
        embedding_path = os.path.join(ENROLLED_DIR, info["embedding_file"])
        if os.path.exists(embedding_path):
            enrolled[name] = torch.load(embedding_path, weights_only=True)

    return enrolled


# ──────────────────────────────────────────────
# Step 4: Get List of Enrolled User Names
# ──────────────────────────────────────────────
def get_user_names():
    """
    Returns a list of enrolled user names.

    Returns:
        list: List of user name strings
    """
    users = _load_users_json()
    return list(users.keys())


# ──────────────────────────────────────────────
# Step 5: Delete an Enrolled User
# ──────────────────────────────────────────────
def delete_user(name):
    """
    Remove a user's enrollment data from disk.

    Args:
        name (str): User's name to delete
    """
    users = _load_users_json()

    if name in users:
        # Delete embedding file
        embedding_path = os.path.join(ENROLLED_DIR, users[name]["embedding_file"])
        if os.path.exists(embedding_path):
            os.remove(embedding_path)

        # Remove from JSON
        del users[name]
        _save_users_json(users)


# ──────────────────────────────────────────────
# Step 6: Update User (Rename / Change Passphrase)
# ──────────────────────────────────────────────
def update_user(old_name, new_name=None, new_passphrase=None):
    """
    Update a user's name and/or passphrase.

    Args:
        old_name (str): Current user name
        new_name (str, optional): New name (None = keep current)
        new_passphrase (str, optional): New passphrase (None = keep current)

    Returns:
        bool: True if update succeeded
    """
    users = _load_users_json()

    if old_name not in users:
        return False

    user_data = users[old_name]
    target_name = new_name.strip() if new_name and new_name.strip() else old_name

    # Update passphrase if provided
    if new_passphrase and new_passphrase.strip():
        user_data["passphrase"] = new_passphrase.lower().strip()

    # If name changed, rename the embedding file and update dict key
    if target_name != old_name:
        old_emb_path = os.path.join(ENROLLED_DIR, user_data["embedding_file"])
        new_emb_file = f"{target_name}.pt"
        new_emb_path = os.path.join(ENROLLED_DIR, new_emb_file)

        if os.path.exists(old_emb_path):
            os.rename(old_emb_path, new_emb_path)

        user_data["embedding_file"] = new_emb_file

        # Remove old key, add new key
        del users[old_name]
        users[target_name] = user_data
    else:
        users[old_name] = user_data

    _save_users_json(users)
    return True


# ──────────────────────────────────────────────
# Helper Functions (Internal)
# ──────────────────────────────────────────────
def _load_users_json():
    """Load the users JSON file, return empty dict if not found."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_users_json(users):
    """Save users dict to the JSON file."""
    _ensure_dir()
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)
