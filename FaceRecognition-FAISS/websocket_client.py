import asyncio
import websockets
import json
import threading
import requests
import cv2
import numpy as np
import face_recognition_pybind as FR
import faiss
import time
from functools import partial # Dùng để wrap hàm

class WebSocketClient:
    def __init__(self, uri, status_queue, user_management, faiss_index):
        self._uri = uri
        self._status_queue = status_queue
        self._user_management = user_management
        self._faiss_index = faiss_index
        self._thread = None
        self._loop = None
        self._ws = None
        self._main_task = None
        self._running = False # Biến kiểm soát vòng lặp chính
        self._send_queue = None # Queue sẽ được khởi tạo trong loop

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._thread_target, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._main_task.cancel)
        if self._thread:
            self._thread.join(timeout=2.0)

    def send_recognize_message(self, deviceId, faceId):
        """Public method to queue a recognize message"""
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(self._enqueue_recognize(deviceId, faceId), self._loop)

    async def _enqueue_recognize(self, deviceId, faceId):
        msg = {
            "TOPIC": "VMXSys/VAIPLF2CtrlBox/accesscontrol",
            "PAYLOAD": {
                "action": "recognize",
                "deviceId": deviceId,
                "faceId": faceId
            }
        }
        await self._send_queue.put(json.dumps(msg))

    def _thread_target(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._send_queue = asyncio.Queue()
        try:
            self._main_task = self._loop.create_task(self._connect_and_listen())
            self._loop.run_until_complete(self._main_task)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                # Cleanup tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.close()
            except Exception as e:
                print(f"DEBUG WS: Cleanup error: {e}")

    async def _connect_and_listen(self):
        while self._running: # Vòng lặp reconnect tự động
            self._status_queue.put("Connecting...")
            print("DEBUG: Attempting WebSocket connection...")

            try:
                # CẤU HÌNH PING/PONG: Nới lỏng để tránh disconnect
                async with websockets.connect(
                    self._uri, 
                    ping_interval=30, 
                    ping_timeout=60,
                    close_timeout=10
                ) as websocket:
                    self._ws = websocket
                    self._status_queue.put("Connect successfully")
                    print("DEBUG: WebSocket connected successfully.")
                    
                    # Chạy song song 2 task: Nhận và Gửi
                    receive_task = asyncio.create_task(self._receive_loop(websocket))
                    send_task = asyncio.create_task(self._send_loop(websocket))
                    
                    done, pending = await asyncio.wait(
                        [receive_task, send_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for task in pending:
                        task.cancel()

            except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as e:
                self._status_queue.put(f"Connection failed: {e}")
                print(f"DEBUG: Connection failed/dropped: {e}")
            
            # Chờ 5 giây trước khi reconnect để tránh spam server
            if self._running:
                await asyncio.sleep(5)

    async def _receive_loop(self, websocket):
        try:
            async for message in websocket:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"DEBUG: Connection closed in receive loop: {e}")
            self._status_queue.put("Disconnected. Reconnecting...")
            raise e

    async def _send_loop(self, websocket):
        try:
            while True:
                message = await self._send_queue.get()
                await websocket.send(message)
                self._send_queue.task_done()
        except Exception as e:
            print(f"DEBUG: Error in send loop: {e}")
            raise e

    async def _handle_message(self, message):
        try:
            data = json.loads(message)
            topic = data.get("TOPIC")
            payload = data.get("PAYLOAD", {})
            action = payload.get("action")

            if topic == "VMXSys/VMXSLSMobile/face" and action == "register":
                username = payload.get("username")
                face_urls = payload.get("face_urls")

                if username and face_urls:
                    self._status_queue.put(f"Registering user '{username}'...")
                    # Thay đổi cách gọi hàm xử lý đăng ký
                    await self._register_user_async(username, face_urls)
                else:
                    print("DEBUG WS: Invalid registration payload")

            elif topic == "VMXSys/CtrlBox2Device/accesscontrol" and action == "remove":
                command = payload.get("command")
                if command == "all":
                    deviceId = payload.get("deviceId", 1)
                    print("DEBUG WS: Received command to remove ALL users.")
                    self._status_queue.put("CMD:RESET_USERS")
                    await self._send_remove_response(self._ws, command="all", result="completed", deviceId=deviceId)
                elif command == "user":
                    faceId = payload.get("faceId")
                    if faceId:
                         print(f"DEBUG WS: Received command to remove user {faceId}")
                         self._status_queue.put(f"CMD:DELETE_USER:{faceId}")
                         # Phản hồi completed ngay lập tức (hoặc có thể đợi App xóa xong rồi gửi, 
                         # nhưng ở đây ta gửi luôn để confirm đã nhận lệnh)
                         await self._send_remove_response(self._ws, command="user", result="completed", faceId=faceId)

        except Exception as e:
            print(f"DEBUG WS: Error handling message: {e}")

    # ====================================================================
    # PHẦN QUAN TRỌNG NHẤT: Tách xử lý nặng ra khỏi Async Loop
    # ====================================================================

    def _heavy_registration_task(self, username, face_urls):
        """Hàm này chứa toàn bộ logic 'nặng' (Blocking code)"""
        print(f"DEBUG THREAD: Start heavy processing for {username}")
        
        # 1. Check User Exist (Idempotent Handling)
        if self._user_management.is_name_registered(username):
            print(f"DEBUG THREAD: User '{username}' already exists. Returning success (Idempotent).")
            # Find existing UUID to return
            existing_uuid = ""
            self._user_management.load_data() # Ensure fresh data
            for uid, udata in self._user_management.users.items():
                if udata.get('name') == username:
                    existing_uuid = udata.get('uuid', "")
                    break
            
            return {"status": "completed", "faceId": existing_uuid, "msg": "User already exists"}

        # 2. Download & Decode Image
        if not face_urls:
             return {"status": "failed", "faceId": "", "msg": "No URLs"}

        first_url = face_urls[0]
        feature_vector = None

        try:
            # Tác vụ mạng (Blocking)
            response = requests.get(first_url, timeout=10)
            response.raise_for_status()
            image_array = np.frombuffer(response.content, np.uint8)
            # Tác vụ CPU (Blocking)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is not None:
                # Tác vụ AI siêu nặng (Blocking)
                result, facial_feature = FR.register_user(image, image.shape[0], image.shape[1], image.shape[2])
                if result == 0 and len(facial_feature) == 128:
                    feature_vector = np.array(facial_feature, dtype='float32')
        except Exception as e:
            print(f"DEBUG THREAD: Error in heavy task: {e}")
            return {"status": "failed", "faceId": "", "msg": str(e)}

        if feature_vector is None:
             return {"status": "failed", "faceId": "", "msg": "No face found"}

        # 3. Save to DB & FAISS
        try:
            final_feature = feature_vector.reshape(1, 128)
            faiss.normalize_L2(final_feature)

            # Tác vụ I/O Database
            success, message, new_int_id, new_uuid = self._user_management.register_user(username)
            
            if success:
                # Lưu ý: faiss_index thường không thread-safe tuyệt đối, 
                # nhưng index add/search cơ bản thường ổn.
                self._faiss_index.add_with_ids(final_feature, np.array([new_int_id]).astype('int64'))
                faiss.write_index(self._faiss_index, "facial_faiss_index.bin")
                return {"status": "completed", "faceId": new_uuid, "msg": "Success"}
            else:
                return {"status": "failed", "faceId": "", "msg": message}

        except Exception as e:
            return {"status": "failed", "faceId": "", "msg": str(e)}

    async def _register_user_async(self, username, face_urls):
        """Wrapper bất đồng bộ để gọi hàm nặng trong Executor"""
        loop = asyncio.get_running_loop()
        
        # CHẠY HÀM NẶNG TRONG THREAD POOL ĐỂ KHÔNG CHẶN LOOP
        # run_in_executor(None, ...) sẽ dùng Default ThreadPoolExecutor
        result_data = await loop.run_in_executor(
            None, 
            partial(self._heavy_registration_task, username, face_urls)
        )

        # Sau khi xử lý xong, gửi kết quả lại qua WebSocket (Non-blocking)
        if result_data["status"] == "completed":
            self._status_queue.put(f"Successfully registered '{username}'.")
            self._status_queue.put("CMD:RELOAD_DATA")
            await self._send_register_response(self._ws, username, face_urls, "completed", faceId=result_data["faceId"])
        else:
            self._status_queue.put(f"Registration failed: {result_data['msg']}")
            await self._send_register_response(self._ws, username, face_urls, "failed", faceId="")

    async def _send_remove_response(self, ws, command, result, faceId=None, deviceId=None):
        payload = {"action": "remove", "command": command, "result": result}
        if faceId: payload["faceId"] = faceId
        if deviceId: payload["deviceId"] = deviceId
        response = {"TOPIC": "VMXSys/VAIPLF2CtrlBox/accesscontrol", "PAYLOAD": payload}
        try: await ws.send(json.dumps(response))
        except: pass

    async def _send_register_response(self, ws, username, face_urls, result, faceId):
        response = {
            "TOPIC": "VMXSys/VAIPLF2CtrlBox/accesscontrol",
            "PAYLOAD": {
                "action": "register", "username": username, "face_urls": face_urls,
                "faceId": faceId, "result": result
            }
        }
        try: await ws.send(json.dumps(response))
        except: pass