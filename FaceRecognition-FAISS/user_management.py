import json
import os
import uuid
from datetime import datetime
import threading

class UserManagement:
    def __init__(self, user_data_file='user_data.json'):
        self.user_data_file = user_data_file
        self.users = {}
        self.next_id = 0
        self._lock = threading.Lock() # Add a lock for thread safety
        self.load_data()

    def load_data(self):
        with self._lock:
            if os.path.exists(self.user_data_file):
                try:
                    # Check for empty file before trying to load
                    if os.path.getsize(self.user_data_file) > 0:
                        with open(self.user_data_file, 'r') as file:
                            data = json.load(file)
                            
                            # Check if it's the new format (has "users" and "next_id" keys)
                            if "users" in data and "next_id" in data:
                                # New format: {"next_id": 0, "users": {"0": {...}}}
                                self.users = data["users"]
                                self.next_id = data["next_id"]
                                print(f"✅ Loaded NEW format: {len(self.users)} users, next_id={self.next_id}")
                            else:
                                # Old format: {"name1": {"timestamp": "..."}, "name2": {...}}
                                # Migrate to new format
                                print(f"⚠️ Detected OLD format. Migrating {len(data)} users to new format...")
                                self.users = {}
                                self.next_id = 0
                                
                                for old_name, old_data in data.items():
                                    new_uuid = str(uuid.uuid4())
                                    self.users[str(self.next_id)] = {
                                        'uuid': new_uuid,
                                        'name': old_name,
                                        'timestamp': old_data.get('timestamp', str(datetime.now()))
                                    }
                                    self.next_id += 1
                                
                                # Save migrated data immediately (without lock - already locked)
                                with open(self.user_data_file, 'w') as file:
                                    data_to_save = {
                                        "next_id": self.next_id,
                                        "users": self.users
                                    }
                                    json.dump(data_to_save, file, indent=4)
                                print(f"✅ Migration complete: {len(self.users)} users migrated")
                    else:
                        self.users = {}
                        self.next_id = 0
                except (json.JSONDecodeError, IOError) as e:
                    print(f"❌ Error loading user data: {e}")
                    # This can happen in a race condition, though lock reduces risk.
                    # To be safe, we reset.
                    self.users = {}
                    self.next_id = 0
            else:
                self.users = {}
                self.next_id = 0

    def save_data(self):
        with self._lock:
            with open(self.user_data_file, 'w') as file:
                data_to_save = {
                    "next_id": self.next_id,
                    "users": self.users
                }
                json.dump(data_to_save, file, indent=4)

    def is_name_registered(self, name):
        """Checks if a username is already taken."""
        self.load_data()  # Force reload from file to ensure state is synchronized
        for user_data in self.users.values():
            if user_data['name'] == name:
                return True
        return False

    def register_user(self, name):
        """
        Registers a new user with a unique integer ID and a UUID.
        Returns a tuple: (success, message, int_id, uuid_str)
        """
        if self.is_name_registered(name):
            return False, f"Failed! Username '{name}' is already registered.", None, None

        new_int_id = self.next_id
        new_uuid = str(uuid.uuid4())
        
        self.users[str(new_int_id)] = {
            'uuid': new_uuid,
            'name': name,
            'timestamp': str(datetime.now())
        }
        
        self.next_id += 1
        self.save_data()
        
        return True, f"'{name}' registered successfully.", new_int_id, new_uuid

    def reset_users(self):
        self.users.clear()
        self.next_id = 0
        self.save_data()

    def get_all_user_data(self):
        """Returns the entire dictionary of users."""
        self.load_data() # Ensure freshest data is returned
        return self.users

    def get_user_by_id(self, int_id):
        """Gets user data by their integer ID."""
        self.load_data() # Ensure freshest data is returned
        return self.users.get(str(int_id))
