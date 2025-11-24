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

    def start(self):
        self._thread = threading.Thread(target=self._thread_target, daemon=True)
        self._thread.start()

    def stop(self):
        if self._main_task and self._loop:
            self._loop.call_soon_threadsafe(self._main_task.cancel)
        if self._thread:
            self._thread.join()

    def _thread_target(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._main_task = self._loop.create_task(self._connect_and_listen())
            self._loop.run_until_complete(self._main_task)
        except asyncio.CancelledError:
            pass
        finally:
            all_tasks = asyncio.all_tasks(loop=self._loop)
            group = asyncio.gather(*all_tasks, return_exceptions=True)
            self._loop.run_until_complete(group)
            self._loop.close()

    async def _connect_and_listen(self):
        self._status_queue.put("Connecting...")
        print("DEBUG: Attempting WebSocket connection...")

        try:
            async with websockets.connect(self._uri) as websocket:
                self._ws = websocket
                self._status_queue.put("Connect successfully")
                print("DEBUG: WebSocket connected successfully.")
                async for message in websocket:
                    print(f"DEBUG: Received message from WebSocket: {message}")
                    self._status_queue.put(f"Received message: {message[:100]}...")
                    await self._handle_message(message)
        except Exception as e:
            self._status_queue.put(f"Connection failed: {e}")
            print(f"DEBUG: WebSocket connection failed: {e}")

    async def _handle_message(self, message):
        try:
            data = json.loads(message)
            topic = data.get("TOPIC")
            payload = data.get("PAYLOAD", {})
            action = payload.get("action")

            if topic == "VMXSys/VMXSLSMobile/face" and action == "register":
                username = payload.get("username")
                face_urls = payload.get("face_urls")
                request_id = payload.get("request_id")

                if username and face_urls:
                    self._status_queue.put(f"Registering user '{username}'...")
                    await self._register_from_urls(self._ws, username, face_urls, request_id)
                else:
                    self._status_queue.put("Invalid registration payload.")
            
            elif topic == "VMXSys/VAIPLF2CtrlBox/accesscontrol" and action == "remove":
                command = payload.get("command")
                if command == "all":
                    print("DEBUG WS: Received command to reset all users. Notifying app and stopping communication.")
                    self._status_queue.put("CMD:RESET_USERS")
                    # Per requirement, do not send a confirmation message back to STB.

        except json.JSONDecodeError:
            self._status_queue.put("Received non-JSON message.")
            print("DEBUG WS: Received non-JSON message.")
        except Exception as e:
            self._status_queue.put(f"Error handling message: {e}")
            print(f"DEBUG WS: Error handling message: {e}")

    async def _register_from_urls(self, ws, username, face_urls, request_id):
        print(f"DEBUG WS: Starting registration process for '{username}'.")
        if self._user_management.is_name_registered(username):
            message = f"User '{username}' already exists."
            print(f"DEBUG WS: {message}")
            self._status_queue.put(message)
            return

        # =================================================================================
        # NEW LOGIC (2025-11-21): Only process the first URL from the list
        # =================================================================================
        feature_vector = None
        if not face_urls:
            print("DEBUG WS: ERROR - No face URLs provided.")
            self._status_queue.put("No face URLs provided for registration.")
            return

        first_url = face_urls[0]
        print(f"DEBUG WS: Processing only the first URL: {first_url}")

        try:
            response = requests.get(first_url, timeout=10)
            response.raise_for_status()
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"DEBUG WS: WARNING - Failed to decode image from {first_url}.")
                self._status_queue.put(f"Failed to decode image from URL.")
                return

            result, facial_feature = FR.register_user(image, image.shape[0], image.shape[1], image.shape[2])
            
            if result == 0 and len(facial_feature) == 128:
                print(f"DEBUG WS: Successfully extracted features from the first URL.")
                feature_vector = np.array(facial_feature, dtype='float32')
            else:
                print(f"DEBUG WS: WARNING - Failed to get valid features from the first URL. Result: {result}, Length: {len(facial_feature)}")

        except Exception as e:
            print(f"DEBUG WS: ERROR - Exception while processing image from {first_url}: {e}")
            self._status_queue.put(f"Error processing image from URL: {e}")
            return
        
        if feature_vector is None:
            message = "No valid face found in the first image."
            print(f"DEBUG WS: {message} Aborting registration.")
            self._status_queue.put(message)
            return

        try:
            # Reshape and normalize the single feature vector
            final_feature = feature_vector.reshape(1, 128)
            faiss.normalize_L2(final_feature)
            print("DEBUG WS: Single feature vector processed and normalized.")

            print("DEBUG WS: Registering user in UserManagement.")
            success, message, new_int_id, new_uuid = self._user_management.register_user(username)
            print(f"DEBUG WS: UserManagement result: success={success}, msg='{message}', id={new_int_id}")
            
            if success:
                print(f"DEBUG WS: Adding vector to FAISS with ID {new_int_id}.")
                self._faiss_index.add_with_ids(final_feature, np.array([new_int_id]).astype('int64'))
                print("DEBUG WS: Writing updated FAISS index to disk.")
                faiss.write_index(self._faiss_index, "facial_faiss_index.bin")
                self._status_queue.put(f"Successfully registered '{username}'.")
                print("DEBUG WS: Local registration complete.")

                response_payload = {
                    "TOPIC": "VMXSys/VAIPLF2CtrlBox/accesscontrol",
                    "PAYLOAD": {
                        "action": "register",
                        "username": username,
                        "face_urls": face_urls, # Still report original URLs
                        "request_id": request_id,
                        "timestamp": int(time.time() * 1000),
                        "faceId": new_uuid,
                        "result": "completed"
                    }
                }
                print("DEBUG WS: Preparing to send 'completed' message to server.")
                await ws.send(json.dumps(response_payload))
                print("DEBUG WS: 'completed' message sent successfully.")
                self._status_queue.put(f"Sent registration confirmation for '{username}'.")
            else:
                print(f"DEBUG WS: ERROR - UserManagement registration failed: {message}")
                self._status_queue.put(f"Failed to register '{username}': {message}")
        except Exception as e:
            print(f"DEBUG WS: CRITICAL ERROR after feature extraction: {e}")
            self._status_queue.put(f"Critical error during final registration steps: {e}")

        # =================================================================================
        # PREPARED CODE: Store 3 separate vectors for one user
        # =================================================================================
        # # Collect all feature vectors first
        # feature_vectors = []
        # print(f"DEBUG WS: Processing {len(face_urls)} face URLs.")
        # for i, url in enumerate(face_urls):
        #     try:
        #         print(f"DEBUG WS: Processing URL {i+1}/{len(face_urls)}: {url}")
        #         response = requests.get(url, timeout=10)
        #         response.raise_for_status()
        #         image_array = np.frombuffer(response.content, np.uint8)
        #         image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
        #         if image is None:
        #             print(f"DEBUG WS: WARNING - Failed to decode image from {url}.")
        #             continue

        #         result, facial_feature = FR.register_user(image, image.shape[0], image.shape[1], image.shape[2])
                
        #         if result == 0 and len(facial_feature) == 128:
        #             print(f"DEBUG WS: Successfully extracted features from URL {i+1}.")
        #             feature_vectors.append(np.array(facial_feature, dtype='float32'))
        #         else:
        #             print(f"DEBUG WS: WARNING - Failed to get valid features from URL {i+1}. Result: {result}")
        #     except Exception as e:
        #         print(f"DEBUG WS: ERROR - Exception while processing image from {url}: {e}")
        #         continue
        
        # print(f"DEBUG WS: Finished processing URLs. Found {len(feature_vectors)} valid feature vectors.")
        # if not feature_vectors:
        #     message = "No valid faces found in the provided images."
        #     print(f"DEBUG WS: {message} Aborting registration.")
        #     self._status_queue.put(message)
        #     return

        # try:
        #     # Register user in management system ONCE to get a base ID
        #     print("DEBUG WS: Registering user in UserManagement.")
        #     success, message, new_int_id, new_uuid = self._user_management.register_user(username)
        #     print(f"DEBUG WS: UserManagement result: success={success}, msg='{message}', id={new_int_id}")

        #     if success:
        #         # Add each feature vector to FAISS with a unique, derived ID
        #         ids_to_add = []
        #         vectors_to_add = []
        #         for i, vector in enumerate(feature_vectors):
        #             # Derive a unique ID, e.g., for user ID 5, vectors will have IDs 50, 51, 52
        #             vector_id = new_int_id * 10 + i 
        #             ids_to_add.append(vector_id)
                    
        #             # Normalize the vector
        #             normalized_vector = vector.reshape(1, 128)
        #             faiss.normalize_L2(normalized_vector)
        #             vectors_to_add.append(normalized_vector)
                    
        #             print(f"DEBUG WS: Preparing vector {i+1} for FAISS with derived ID {vector_id}.")

        #         # Add all vectors and their IDs to FAISS in one go
        #         if vectors_to_add:
        #             self._faiss_index.add_with_ids(np.vstack(vectors_to_add), np.array(ids_to_add).astype('int64'))
                
        #         print(f"DEBUG WS: Writing {len(vectors_to_add)} vectors to updated FAISS index.")
        #         faiss.write_index(self._faiss_index, "facial_faiss_index.bin")
        #         self._status_queue.put(f"Successfully registered '{username}' with {len(vectors_to_add)} face vectors.")
        #         print("DEBUG WS: Local registration complete.")
                
        #         # ... (Send 'completed' message back via websocket as before) ...
        #         response_payload = {
        #             "TOPIC": "VMXSys/VAIPLF2CtrlBox/accesscontrol",
        #             "PAYLOAD": {
        #                 "action": "register",
        #                 "username": username,
        #                 "face_urls": face_urls,
        #                 "request_id": request_id,
        #                 "timestamp": int(time.time() * 1000),
        #                 "faceId": new_uuid,
        #                 "result": "completed"
        #             }
        #         }
        #         await ws.send(json.dumps(response_payload))

        #     else:
        #         print(f"DEBUG WS: ERROR - UserManagement registration failed: {message}")
        #         self._status_queue.put(f"Failed to register '{username}': {message}")
        # except Exception as e:
        #     print(f"DEBUG WS: CRITICAL ERROR after feature extraction: {e}")
        #     self._status_queue.put(f"Critical error during final registration steps: {e}")
