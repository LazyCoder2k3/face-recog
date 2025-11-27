import streamlit as st
import cv2
import numpy as np
import pandas as pd
from camera import ThreadedCamera
from user_management import UserManagement
from access_history import AccessHistory
import time
import asyncio
from datetime import datetime
from typing import Tuple
import faiss
import json
import os
import queue
from websocket_client import WebSocketClient
import threading
import face_recognition_pybind as FR

st.set_page_config(page_title="Face Recognition System", layout="wide")

# ==============================================================================
# SINGLETON RESOURCES (st.cache_resource)
# ==============================================================================

@st.cache_resource
def get_status_queue():
    return queue.Queue()

@st.cache_resource
def get_user_management():
    return UserManagement()

@st.cache_resource
def get_faiss_index():
    dims = 128
    file_path = "facial_faiss_index.bin"
    if os.path.exists(file_path):
        try:
            loaded_index = faiss.read_index(file_path)
            if not isinstance(loaded_index, faiss.IndexIDMap):
                print(f"⚠️ Loaded index is not IndexIDMap. Rebuilding...")
                base_index = faiss.index_factory(dims, "Flat", faiss.METRIC_INNER_PRODUCT)
                new_index = faiss.IndexIDMap(base_index)
                if loaded_index.ntotal > 0:
                    vectors = np.zeros((loaded_index.ntotal, dims), dtype='float32')
                    for i in range(loaded_index.ntotal):
                        vectors[i] = loaded_index.reconstruct(int(i))
                    ids = np.arange(loaded_index.ntotal, dtype='int64')
                    new_index.add_with_ids(vectors, ids)
                faiss.write_index(new_index, file_path)
                return new_index
            return loaded_index
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
    
    # Create new if not exists or error
    base_index = faiss.index_factory(dims, "Flat", faiss.METRIC_INNER_PRODUCT)
    return faiss.IndexIDMap(base_index)

@st.cache_resource
def get_camera(src):
    # Note: src change will trigger new camera creation if not careful, 
    # but st.cache_resource handles args.
    # We want a single camera instance usually, but if src changes we might want a new one.
    # For simplicity in this app, we assume one main camera.
    return ThreadedCamera(src)

@st.cache_resource
def get_ws_client(uri, _status_queue, _user_management, _faiss_index):
    # Underscore prefix arguments to prevent hashing them if they are not hashable
    client = WebSocketClient(uri, _status_queue, _user_management, _faiss_index)
    return client

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

if "last_known_faces" not in st.session_state:
    st.session_state.last_known_faces = [] # List of dicts: {'box': (l,t,r,b), 'name': str}
if "no_face_counter" not in st.session_state:
    st.session_state.no_face_counter = 0
if "cos_sim_thresh" not in st.session_state:
    st.session_state.cos_sim_thresh = 0.6
if "last_log_status" not in st.session_state:
    st.session_state.last_log_status = "No recent activity"
if "last_log_time" not in st.session_state:
    st.session_state.last_log_time = ""
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# Debounce for recognize message
if "last_sent_recognize" not in st.session_state:
    st.session_state.last_sent_recognize = {} # {uuid: timestamp}

# ==============================================================================
# MAIN APP LOGIC
# ==============================================================================
FREQUENCY = 16
st.title("Check-in System (Refactored)")

# 1. Get Resources
status_queue = get_status_queue()
user_management = get_user_management()
faiss_index = get_faiss_index()
access_history = AccessHistory() # AccessHistory handles its own file I/O, lightweight enough to init

# 2. Sidebar / Configuration
with st.sidebar:
    st.header("Configuration")
    
    # Camera Config
    cam_src = st.text_input("Camera Source", value="rtsp://adminOnvif:Vision123@10.60.3.157/Streaming/Channels/102?transportmode=unicast&profile=Profile_1")
    
    # Handle numeric input for USB cameras
    if cam_src.isdigit():
        cam_src = int(cam_src)
    
    # WebSocket Config
    server_ip = st.text_input("Server IP", value="10.60.1.90")
    server_port = st.number_input("Port", value=8005, min_value=1, max_value=65535)
    ws_uri = f"ws://{server_ip}:{server_port}"

    # Controls - Separated into 4 distinct buttons
    st.subheader("Camera Controls")
    col_cam_start, col_cam_stop = st.columns(2)
    with col_cam_start:
        if st.button("▶️ Start Camera", use_container_width=True):
            if not st.session_state.camera_active:
                st.session_state.camera_active = True
                st.success("Camera starting...")
            else:
                st.warning("Camera is already active")

    with col_cam_stop:
        if st.button("⏹️ Stop Camera", use_container_width=True):
            if st.session_state.camera_active:
                st.session_state.camera_active = False
                # Properly stop the camera thread and release resources
                try:
                    cam = get_camera(cam_src)
                    cam.stop()
                    st.success("Camera stopped and disconnected")
                except Exception as e:
                    st.error(f"Error stopping camera: {e}")
            else:
                st.warning("Camera is already stopped")

    st.divider()
    st.subheader("WebSocket Controls")
    col_ws_connect, col_ws_disconnect = st.columns(2)
    with col_ws_connect:
        if st.button("🔌 Connect WS", use_container_width=True):
            if not st.session_state.ws_connected:
                try:
                    client = get_ws_client(ws_uri, status_queue, user_management, faiss_index)
                    client.start()
                    st.session_state.ws_connected = True
                    st.success(f"Connecting to {ws_uri}...")
                except Exception as e:
                    st.error(f"Failed to connect: {e}")
            else:
                st.warning("WebSocket is already connected")

    with col_ws_disconnect:
        if st.button("🔌 Disconnect WS", use_container_width=True):
            if st.session_state.ws_connected:
                try:
                    client = get_ws_client(ws_uri, status_queue, user_management, faiss_index)
                    client.stop()
                    st.session_state.ws_connected = False
                    # Clear cache to allow fresh connection next time
                    st.cache_resource.clear()
                    st.success("WebSocket disconnected")
                except Exception as e:
                    st.error(f"Error disconnecting: {e}")
            else:
                st.warning("WebSocket is already disconnected") 

    st.divider()
    st.subheader("Face Recognition")
    st.session_state.cos_sim_thresh = st.slider("Threshold", 0.0, 1.0, 0.6, 0.01)
    
    if st.button("Reset All Users"):
        user_management.reset_users()
        # Reset FAISS
        faiss_index.reset()
        faiss.write_index(faiss_index, "facial_faiss_index.bin")
        st.success("Reset completed!")

# 3. Main Layout
col_feed, col_info = st.columns([2, 1])

# Helper function to update history
def update_history_display(placeholder):
    access_history.load_history()
    history_data = access_history.history_data
    lines = []
    for username, timestamps in history_data.items():
        if timestamps:
            lines.append({
                "Name": username,
                "Last Checkin": timestamps[-1],
                "Total": len(timestamps)
            })
    lines.sort(key=lambda x: x.get("Last Checkin", ""), reverse=True)
    
    with placeholder.container():
        st.subheader("📊 Access History")
        if lines:
            st.dataframe(pd.DataFrame(lines), use_container_width=True)
        else:
            st.info("No history.")

with col_info:
    st.subheader("System Status")
    
    # WS Status Display
    ws_status_text = st.empty()
    ws_status_text.info(f"WS Status: {'Active' if st.session_state.ws_connected else 'Inactive'}")
    
    # Registration
    st.divider()
    st.subheader("Manual Registration")
    reg_name = st.text_input("Name for Registration")
    if st.button("Register Current Face"):
        if st.session_state.camera_active:
            cam = get_camera(cam_src)
            frame = cam.get_frame()
            if frame is not None and reg_name:
                # Detect and Register
                res, feat = FR.register_user(frame, frame.shape[0], frame.shape[1], frame.shape[2])
                if res == 0 and len(feat) == 128:
                    vec = np.array(feat, dtype='float32').reshape(1, 128)
                    faiss.normalize_L2(vec)
                    
                    success, msg, new_id, new_uuid = user_management.register_user(reg_name)
                    if success:
                        faiss_index.add_with_ids(vec, np.array([new_id]).astype('int64'))
                        faiss.write_index(faiss_index, "facial_faiss_index.bin")
                        st.success(f"Registered {reg_name}")
                    else:
                        st.error(msg)
                else:
                    st.error("Face not detected or invalid feature")
            else:
                st.warning("No frame or name empty")
        else:
            st.warning("Camera not active")

    # History
    st.divider()
    history_placeholder = st.empty()
    update_history_display(history_placeholder)

with col_feed:
    st.subheader("Camera Feed")
    feed_placeholder = st.empty()

# 4. Background Logic Loop (Run inside Streamlit's script execution)
if st.session_state.camera_active:
    cam = get_camera(cam_src)
    cam.start() # Idempotent if already running
    
    # Get WS Client if connected
    ws_client = None
    if st.session_state.ws_connected:
        ws_client = get_ws_client(ws_uri, status_queue, user_management, faiss_index)

    # Main Loop
    tmp = 0
    while True:
        # A. Process WebSocket Queue
        try:
            while not status_queue.empty():
                msg = status_queue.get_nowait()
                
                # Handle Commands
                if msg == "CMD:RESET_USERS":
                    user_management.reset_users()
                    faiss_index.reset()
                    faiss.write_index(faiss_index, "facial_faiss_index.bin")
                    st.toast("Remote: All users reset")
                    update_history_display(history_placeholder)
                
                elif msg == "CMD:RELOAD_DATA":
                    # Data reloaded implicitly by accessing user_management/faiss_index next time
                    st.toast("Remote: Data reloaded")
                
                elif msg.startswith("CMD:DELETE_USER:"):
                    # Format: CMD:DELETE_USER:{faceId}
                    face_id_to_delete = msg.split(":")[2]
                    success, del_msg, del_int_id = user_management.delete_user_by_uuid(face_id_to_delete)
                    if success:
                        try:
                            faiss_index.remove_ids(np.array([del_int_id]).astype('int64'))
                            faiss.write_index(faiss_index, "facial_faiss_index.bin")
                            st.toast(f"Remote: Deleted user {del_int_id}")
                        except Exception as e:
                            print(f"FAISS remove error: {e}")
                    else:
                        print(f"Delete failed: {del_msg}")

                elif msg.startswith("Connect") or msg.startswith("Disconnect"):
                     ws_status_text.text(f"WS Status: {msg}")
                
                else:
                    # Log other messages
                    print(f"WS MSG: {msg}")

        except queue.Empty:
            pass

        # B. Process Camera Frame
        frame = cam.get_frame()
        status = cam.get_status()
        
        if status != "Connected" or frame is None:
            feed_placeholder.warning(f"Camera Status: {status}")
            time.sleep(0.1)
            continue

        # C. Face Detection & Recognition (Every 4 frames)
        display_frame = frame.copy()
        
        if tmp == 0:
            # Detect
            faces_result = FR.detect_face(frame, frame.shape[0], frame.shape[1], frame.shape[2])
            
            new_faces_list = []
            
            for face_res in faces_result:
                left, top, right, bottom, spoof, feat = face_res
                
                if left > 0 and top > 0:
                    # Recognize
                    name_display = "Unknown"
                    user_uuid = None
                    
                    if len(feat) == 128 and faiss_index.ntotal > 0:
                        vec = np.array(feat, dtype='float32').reshape(1, 128)
                        faiss.normalize_L2(vec)
                        
                        # Search Top K (k=3)
                        k = 3
                        dists, idxs = faiss_index.search(vec, k)
                        
                        best_match_name = None
                        best_score = -1.0
                        
                        if len(idxs) > 0 and len(idxs[0]) > 0:
                            for i, int_id in enumerate(idxs[0]):
                                score = dists[0][i]
                                if int_id != -1:
                                    user_info = user_management.get_user_by_id(int_id)
                                    if user_info:
                                        u_name = user_info['name']
                                        # Logic chọn best match
                                        if i == 0 and score >= st.session_state.cos_sim_thresh:
                                            best_match_name = u_name
                                            best_score = score
                                            user_uuid = user_info['uuid']

                        if best_match_name:
                            name_display = f"{best_match_name} ({best_score:.2f})"
                            # Log & Send WS Message
                            if access_history.log_access(best_match_name):
                                update_history_display(history_placeholder)
                            
                            if ws_client and user_uuid:
                                now = time.time()
                                last_time = st.session_state.last_sent_recognize.get(user_uuid, 0)
                                if now - last_time > 5.0: # Send max once every 5s
                                    ws_client.send_recognize_message(deviceId=1, faceId=user_uuid)
                                    st.session_state.last_sent_recognize[user_uuid] = now
                                    print(f"Sent recognize message for {best_match_name}")
                    
                    new_faces_list.append({
                        'box': (left, top, right, bottom),
                        'name': name_display
                    })
            
            st.session_state.last_known_faces = new_faces_list

        # D. Draw Logic (Apply to every frame based on last known state)
        for face_data in st.session_state.last_known_faces:
            l, t, r, b = face_data['box']
            cv2.rectangle(display_frame, (l, t), (r, b), (0, 255, 0), 2)
            
            # Draw Name
            txt = face_data['name']
            cv2.putText(display_frame, txt, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Update counter
        tmp = (tmp + 1) % FREQUENCY
        
        # Display
        feed_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        
        # Loop delay
        time.sleep(0.03)
else:
    feed_placeholder.info("Camera is stopped. Click 'Start/Stop Camera' to begin.")
