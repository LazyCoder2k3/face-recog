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
        self._lock = threading.Lock() # Lock cho luồng (Thread safety)
        self.load_data()

    def load_data(self):
        """Đọc dữ liệu từ file JSON vào bộ nhớ."""
        with self._lock:
            if os.path.exists(self.user_data_file):
                try:
                    if os.path.getsize(self.user_data_file) > 0:
                        with open(self.user_data_file, 'r') as file:
                            data = json.load(file)
                            
                            # Kiểm tra format mới
                            if "users" in data and "next_id" in data:
                                self.users = data["users"]
                                self.next_id = data["next_id"]
                                # print(f"✅ UserData Loaded: {len(self.users)} users.")
                            else:
                                # Migrate format cũ sang mới
                                print(f"⚠️ Migrating old format...")
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
                                self._save_data_no_lock() # Lưu ngay format mới
                    else:
                        self.users = {}
                        self.next_id = 0
                except (json.JSONDecodeError, IOError) as e:
                    print(f"❌ Error loading user data: {e}")
                    self.users = {}
                    self.next_id = 0
            else:
                self.users = {}
                self.next_id = 0

    def save_data(self):
        """Lưu dữ liệu bộ nhớ xuống file (có khóa)."""
        with self._lock:
            self._save_data_no_lock()

    def _save_data_no_lock(self):
        """Hàm hỗ trợ lưu file không dùng lock (để gọi bên trong các hàm đã có lock)."""
        try:
            with open(self.user_data_file, 'w') as file:
                data_to_save = {
                    "next_id": self.next_id,
                    "users": self.users
                }
                json.dump(data_to_save, file, indent=4)
        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def is_name_registered(self, name):
        """Kiểm tra tên đã tồn tại chưa."""
        # Reload để đảm bảo dữ liệu mới nhất từ các tiến trình khác
        self.load_data() 
        for user_data in self.users.values():
            if user_data.get('name') == name:
                return True
        return False

    def register_user(self, name):
        """Đăng ký user mới."""
        # Không cần gọi load_data() ở đây vì is_name_registered đã gọi rồi
        if self.is_name_registered(name):
            return False, f"Failed! Username '{name}' is already registered.", None, None

        with self._lock:
            new_int_id = self.next_id
            new_uuid = str(uuid.uuid4())
            
            # Key của dict là string của int_id
            self.users[str(new_int_id)] = {
                'uuid': new_uuid,
                'name': name,
                'timestamp': str(datetime.now())
            }
            
            self.next_id += 1
            self._save_data_no_lock()
            
        return True, f"'{name}' registered successfully.", new_int_id, new_uuid

    # --- MỚI: Hàm xóa 1 user theo UUID (faceId) ---
    def delete_user_by_uuid(self, target_uuid):
        """
        Xóa user dựa trên UUID.
        Return: (Success: bool, Message: str, Deleted_Int_ID: int)
        """
        self.load_data() # Cập nhật dữ liệu mới nhất trước khi xóa
        
        key_to_remove = None
        user_name = "Unknown"
        deleted_int_id = -1

        with self._lock:
            # Tìm user có uuid khớp
            for key, user_data in self.users.items():
                if user_data.get('uuid') == target_uuid:
                    key_to_remove = key
                    user_name = user_data.get('name', 'Unknown')
                    deleted_int_id = int(key)
                    break
            
            if key_to_remove:
                del self.users[key_to_remove]
                self._save_data_no_lock()
                print(f"🗑️ Deleted user '{user_name}' (ID: {deleted_int_id})")
                return True, f"User '{user_name}' deleted.", deleted_int_id
            else:
                return False, "User not found.", -1

    def reset_users(self):
        """Xóa toàn bộ user."""
        with self._lock:
            self.users.clear()
            self.next_id = 0
            self._save_data_no_lock()
        return True, "All users reset."

    def get_all_user_data(self):
        """Lấy toàn bộ dữ liệu (Reload trước khi lấy)."""
        self.load_data()
        return self.users

    def get_user_by_id(self, int_id):
        """Lấy thông tin user theo ID số nguyên."""
        self.load_data()
        return self.users.get(str(int_id))