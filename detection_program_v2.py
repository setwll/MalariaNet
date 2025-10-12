import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk
import json
from datetime import datetime
import threading
import queue
import time

IMG_SIZE = 128
NUM_CLASSES = 3
CLASS_NAMES = ['falciparum', 'vivax', 'uninfected']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_white_noise(img_bgr, threshold=40, ratio_limit=0.01):   
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask_dark = gray <= threshold
    if mask_dark.mean() < ratio_limit: 
        return img_bgr
    pixels = img_bgr[~mask_dark]
    if pixels.size == 0: 
        return img_bgr
    mean, std = pixels.mean(axis=0), pixels.std(axis=0)
    noise = np.random.normal(mean, std, img_bgr.shape).astype(np.uint8)
    for c in range(3): 
        img_bgr[:,:,c][mask_dark] = noise[:,:,c][mask_dark]
    return img_bgr

def enhance_7channel(img_bgr):
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_bgr = apply_white_noise(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)

    contrast = np.array(ImageEnhance.Contrast(pil).enhance(2.0))
    saturation = np.array(ImageEnhance.Color(pil).enhance(2.0))
    sharpness = np.array(ImageEnhance.Sharpness(pil).enhance(2.0))
    edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 100, 180)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    assert img_rgb.shape == contrast.shape == saturation.shape == sharpness.shape == edges_rgb.shape == (IMG_SIZE, IMG_SIZE, 3), "Channel mismatch!"
    
    combined = np.stack([
        img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2],
        contrast[:, :, 0], saturation[:, :, 1], sharpness[:, :, 2],
        edges_rgb[:, :, 0]
    ], axis=2)

    assert combined.shape[2] == 7, f"[ERROR] Channel total: {combined.shape[2]}"
    return combined.astype(np.float32) / 255.0

# CNN Model
class MalariaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(7,32,3,padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.01), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.01), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (IMG_SIZE//16)**2,128), nn.LeakyReLU(0.01), nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES)
        )
    def forward(self,x): return self.classifier(self.features(x))

class BoundingBox:
    """Class to handle bounding box annotations"""
    def __init__(self, x, y, size=IMG_SIZE, confidence=None, predicted_class=None):
        self.x = x
        self.y = y
        self.size = size
        self.confidence = confidence
        self.predicted_class = predicted_class
        self.id = id(self)
    
    def get_bbox_coords(self):
        """Get bounding box coordinates (x1, y1, x2, y2)"""
        x1 = max(0, self.x - self.size // 2)
        y1 = max(0, self.y - self.size // 2)
        x2 = x1 + self.size
        y2 = y1 + self.size
        return x1, y1, x2, y2

class MalariaMicroscopeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Malaria Detection System - Microscope Interface v1.0")
                
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(1400, screen_w - 100)
        height = min(900, screen_h - 100)
        self.root.geometry(f"{width}x{height}")
        
        # Initialize variables
        self.current_image = None
        self.original_image = None
        self.rotation_angle = 0  
        self.image_scale = 1.0
        self.zoom_factor = 1.0
        self.bounding_boxes = []
        self.model = None
        self.model_loaded = False
        
        # Image folder navigation variables
        self.image_folder_path = None
        self.image_files_list = []
        self.current_image_index = 0
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp')
        
        # Panning and interaction variables
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
        self.is_dragging = False
        self.last_click_time = 0
        self.click_threshold = 300  # milliseconds
        self.drag_threshold = 5  # pixels
        self.start_x = 0
        self.start_y = 0
        
        # Camera variables with error handling
        self.camera = None
        self.camera_active = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.camera_lock = threading.Lock()
        self.camera_reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        
        # Setup GUI
        self.setup_gui()
        self.load_model()
        
        # Start periodic camera frame update
        self.update_camera_display()
        
    def setup_gui(self):
        """Setup the main GUI interface with scrollable controls + rotation features"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollable control panel (left side)
        control_container = ttk.Frame(main_frame)
        control_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Create canvas for scrolling
        control_canvas = tk.Canvas(control_container, width=300, highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=control_canvas.yview)
        scrollable_frame = ttk.Frame(control_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        
        control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrolling components
        control_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas for scrolling
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def unbind_mousewheel(event):
            control_canvas.unbind_all("<MouseWheel>")
        
        control_canvas.bind('<Enter>', bind_mousewheel)
        control_canvas.bind('<Leave>', unbind_mousewheel)
        
        # scrollable_frame for controls section
        control_frame = ttk.LabelFrame(scrollable_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(control_frame, text="Camera Controls", padding=5)
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Camera selection
        camera_select_frame = ttk.Frame(camera_frame)
        camera_select_frame.pack(fill=tk.X, pady=2)
        ttk.Label(camera_select_frame, text="Camera:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(camera_select_frame, textvariable=self.camera_var, 
                                   values=["0", "1", "2", "3"], width=5, state="readonly")
        camera_combo.pack(side=tk.RIGHT)
        
        ttk.Button(camera_frame, text="Start Camera", command=self.start_camera).pack(fill=tk.X, pady=2)
        ttk.Button(camera_frame, text="Stop Camera", command=self.stop_camera).pack(fill=tk.X, pady=2)
        ttk.Button(camera_frame, text="Capture Image", command=self.capture_image).pack(fill=tk.X, pady=2)
        ttk.Button(camera_frame, text="Reconnect Camera", command=self.reconnect_camera).pack(fill=tk.X, pady=2)
        
        # Camera status
        self.camera_status_label = ttk.Label(camera_frame, text="Camera: Disconnected", foreground="red")
        self.camera_status_label.pack(pady=2)
        
        # File controls
        file_frame = ttk.LabelFrame(control_frame, text="Image Input Options", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Browse & Load Image", command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load from Folder", command=self.load_from_folder).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="💾 Save Session", command=self.save_session).pack(fill=tk.X, pady=2)
        
        # Image navigation (for folder mode)
        nav_frame = ttk.Frame(file_frame)
        nav_frame.pack(fill=tk.X, pady=2)
        ttk.Button(nav_frame, text="◀ Prev", command=self.prev_image, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_image, width=8).pack(side=tk.RIGHT, padx=2)
        
        self.image_counter_label = ttk.Label(file_frame, text="No images loaded")
        self.image_counter_label.pack(pady=2)
        
        # Image rotation controls
        rotation_frame = ttk.LabelFrame(control_frame, text="Image Rotation Controls", padding=5)
        rotation_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Quick rotation buttons
        quick_rotate_frame = ttk.Frame(rotation_frame)
        quick_rotate_frame.pack(fill=tk.X, pady=2)
        ttk.Button(quick_rotate_frame, text="↻ 90°", command=lambda: self.rotate_image(90), width=6).pack(side=tk.LEFT, padx=1)
        ttk.Button(quick_rotate_frame, text="↻ 180°", command=lambda: self.rotate_image(180), width=6).pack(side=tk.LEFT, padx=1)
        ttk.Button(quick_rotate_frame, text="↻ 270°", command=lambda: self.rotate_image(270), width=6).pack(side=tk.LEFT, padx=1)
        
        # Custom angle rotation
        custom_angle_frame = ttk.Frame(rotation_frame)
        custom_angle_frame.pack(fill=tk.X, pady=2)
        ttk.Label(custom_angle_frame, text="Custom:").pack(side=tk.LEFT)
        self.angle_var = tk.StringVar(value="0")
        angle_entry = ttk.Entry(custom_angle_frame, textvariable=self.angle_var, width=5)
        angle_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(custom_angle_frame, text="°").pack(side=tk.LEFT)
        ttk.Button(custom_angle_frame, text="Apply", command=self.apply_custom_rotation, width=8).pack(side=tk.RIGHT)
        
        # Rotation info and reset
        self.rotation_info_label = ttk.Label(rotation_frame, text="Current rotation: 0°")
        self.rotation_info_label.pack(pady=2)
        
        ttk.Button(rotation_frame, text="🏠 Reset Rotation", command=self.reset_rotation).pack(fill=tk.X, pady=2)
        
        # Rotation instructions
        rotation_help = ttk.Label(rotation_frame, text="⌨️ Shortcuts: Ctrl+R (90°), Ctrl+Shift+R (270°)", 
                                 font=("Arial", 8), foreground="gray")
        rotation_help.pack(pady=2)
        
        # Zoom controls (updated with rotation info)
        zoom_frame = ttk.LabelFrame(control_frame, text="Navigation Controls", padding=5)
        zoom_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(zoom_frame, text="🔍+ Zoom In", command=self.zoom_in).pack(fill=tk.X, pady=2)
        ttk.Button(zoom_frame, text="🔍- Zoom Out", command=self.zoom_out).pack(fill=tk.X, pady=2)
        ttk.Button(zoom_frame, text="🎯 Reset Zoom", command=self.reset_zoom).pack(fill=tk.X, pady=2)
        ttk.Button(zoom_frame, text="🏠 Fit to Window", command=self.fit_to_window).pack(fill=tk.X, pady=2)
        
        # Pan instructions
        pan_info = ttk.Label(zoom_frame, text="🖱️ Pan: Drag with mouse\n⌨️ Arrow keys to pan\n🖱️ Scroll wheel to zoom", 
                            font=("Arial", 8), foreground="gray")
        pan_info.pack(pady=5)
        
        # Annotation controls
        annotation_frame = ttk.LabelFrame(control_frame, text="Annotation Controls", padding=5)
        annotation_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(annotation_frame, text="🗑️ Clear All Annotations", command=self.clear_annotations).pack(fill=tk.X, pady=2)
        ttk.Button(annotation_frame, text="✂️ Crop & Save Cells", command=self.crop_and_save).pack(fill=tk.X, pady=2)
        ttk.Button(annotation_frame, text="💾 Save Current Annotations", command=self.save_current_annotations).pack(fill=tk.X, pady=2)
        ttk.Button(annotation_frame, text="📋 Load Annotations", command=self.load_annotations).pack(fill=tk.X, pady=2)
        
        # Model controls
        model_frame = ttk.LabelFrame(control_frame, text="Model Controls", padding=5)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(model_frame, text="Load Model", command=self.load_model_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(model_frame, text="Classify Selected Cells", command=self.classify_cells).pack(fill=tk.X, pady=2)
        
        # Status frame
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding=5)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Ready", wraplength=200)
        self.status_label.pack()
        
        self.model_status_label = ttk.Label(status_frame, text="Model: Not loaded", wraplength=200, foreground="red")
        self.model_status_label.pack()
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(control_frame, text="Session Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_label = ttk.Label(stats_frame, text="Annotations: 0\nFalciparum: 0\nVivax: 0\nUninfected: 0", wraplength=200)
        self.stats_label.pack()
        
        # Troubleshooting frame
        troubleshoot_frame = ttk.LabelFrame(control_frame, text="Troubleshooting", padding=5)
        troubleshoot_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(troubleshoot_frame, text="🔍 List Available Cameras", command=self.list_cameras).pack(fill=tk.X, pady=2)
        ttk.Button(troubleshoot_frame, text="🔧 Test Camera", command=self.test_camera).pack(fill=tk.X, pady=2)
        
        # Image display (right side)
        image_frame = ttk.LabelFrame(main_frame, text="Microscope View", padding=10)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas for image display with scrollbars
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="black", cursor="crosshair")
        
        # Scrollbars for canvas
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Pack scrollbars and canvas
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_right_click)  
        self.canvas.bind("<Button-2>", self.on_middle_click)  
        
        # mousewheel binding for canvas zoom
        def bind_canvas_mousewheel(event):
            """Bind mousewheel to canvas when mouse enters"""
            self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
            # For Linux systems
            self.canvas.bind("<Button-4>", lambda e: self.on_mouse_wheel_linux(e, 1))
            self.canvas.bind("<Button-5>", lambda e: self.on_mouse_wheel_linux(e, -1))
        
        def unbind_canvas_mousewheel(event):
            """Unbind mousewheel from canvas when mouse leaves"""
            self.canvas.unbind("<MouseWheel>")
            self.canvas.unbind("<Button-4>")
            self.canvas.unbind("<Button-5>")
        
        # Bind/unbind mousewheel based on mouse position
        self.canvas.bind('<Enter>', bind_canvas_mousewheel)
        self.canvas.bind('<Leave>', unbind_canvas_mousewheel)
        
        # Keyboard bindings for panning AND rotation
        self.canvas.bind("<Key>", self.on_key_press)
        self.canvas.focus_set()
        
        # Global keyboard shortcuts for rotation
        self.root.bind("<Control-r>", lambda e: self.rotate_image(90))  # Ctrl+R
        self.root.bind("<Control-R>", lambda e: self.rotate_image(270))  # Ctrl+Shift+R
        self.root.bind("<Control-0>", lambda e: self.reset_rotation())  # Ctrl+0
    
    # Image rotation methods
    def rotate_image(self, angle):
        """Rotate image by specified angle (degrees)"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        try:
            # Update total rotation angle
            self.rotation_angle = (self.rotation_angle + angle) % 360
            
            # Apply rotation to the original image
            height, width = self.original_image.shape[:2]
            center = (width // 2, height // 2)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
            
            # Calculate new bounding dimensions
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_angle) + (width * cos_angle))
            new_height = int((height * cos_angle) + (width * sin_angle))
            
            # Adjust rotation matrix for new center
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Apply rotation
            rotated_image = cv2.warpAffine(self.original_image, rotation_matrix, (new_width, new_height), 
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            
            # Store original if this is the first rotation
            if not hasattr(self, 'base_original_image'):
                self.base_original_image = self.original_image.copy()
            
            # Update current image
            self.original_image = rotated_image
            
            # Transform existing annotations
            self.transform_annotations_for_rotation(rotation_matrix, (width, height), (new_width, new_height))
            
            # Update display
            self.fit_to_window()  # Auto-fit after rotation for better UX
            self.update_rotation_info()
            self.update_status(f"Rotated image by {angle}° (total: {self.rotation_angle}°)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rotate image: {str(e)}")
    
    def apply_custom_rotation(self):
        """Apply custom rotation angle from entry field"""
        try:
            angle = float(self.angle_var.get())
            if -360 <= angle <= 360:
                # Calculate the increment from current rotation
                target_angle = angle % 360
                increment = (target_angle - self.rotation_angle) % 360
                if increment > 180:
                    increment -= 360
                
                if abs(increment) > 0.1:  # Only rotate if significant difference
                    self.rotate_image(increment)
                else:
                    self.update_status("Already at target rotation")
            else:
                messagebox.showerror("Error", "Angle must be between -360 and 360 degrees")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for rotation angle")
    
    def reset_rotation(self):
        """Reset image rotation to 0 degrees"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        if hasattr(self, 'base_original_image'):
            self.original_image = self.base_original_image.copy()
            self.rotation_angle = 0
            
            # Reset annotations to original positions (clear them for simplicity)
            response = messagebox.askyesnocancel("Reset Annotations", 
                "Resetting rotation will clear all annotations.\n\nDo you want to continue?")
            if response is True:
                self.bounding_boxes = []
                self.display_image()
                self.update_rotation_info()
                self.update_statistics()
                self.update_status("Rotation reset to 0°")
            elif response is False:
                # Keep current rotation but inform user
                self.update_status("Rotation reset cancelled")
        else:
            self.rotation_angle = 0
            self.update_rotation_info()
            self.update_status("No rotation to reset")
    
    def transform_annotations_for_rotation(self, rotation_matrix, old_size, new_size):
        """Transform annotation coordinates for rotation"""
        old_width, old_height = old_size
        new_width, new_height = new_size
        
        for bbox in self.bounding_boxes:
            # Convert annotation center to homogeneous coordinates
            point = np.array([[bbox.x, bbox.y]], dtype=np.float32).reshape(-1, 1, 2)
            
            # Apply rotation transformation
            rotated_point = cv2.transform(point, rotation_matrix)
            
            # Update annotation coordinates
            bbox.x = int(rotated_point[0, 0, 0])
            bbox.y = int(rotated_point[0, 0, 1])
            
            # Ensure coordinates are within new image bounds
            bbox.x = max(bbox.size // 2, min(bbox.x, new_width - bbox.size // 2))
            bbox.y = max(bbox.size // 2, min(bbox.y, new_height - bbox.size // 2))
    
    def update_rotation_info(self):
        """Update rotation information display"""
        self.rotation_info_label.config(text=f"Current rotation: {self.rotation_angle}°")
        self.angle_var.set(str(self.rotation_angle))
    
    # Enhanced camera methods with better error handling
    def list_cameras(self):
        """List all available cameras"""
        available_cameras = []
        for i in range(10):  # Check first 10 camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(f"Camera {i}")
                cap.release()
            except:
                continue
        
        if available_cameras:
            messagebox.showinfo("Available Cameras", "Found cameras:\n" + "\n".join(available_cameras))
        else:
            messagebox.showwarning("No Cameras", "No cameras detected. Please check connections.")
    
    def test_camera(self):
        """Test selected camera"""
        try:
            camera_id = int(self.camera_var.get())
            test_cap = cv2.VideoCapture(camera_id)
            
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret:
                    messagebox.showinfo("Camera Test", f"Camera {camera_id} is working!\nResolution: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    messagebox.showerror("Camera Test", f"Camera {camera_id} opened but cannot read frames")
            else:
                messagebox.showerror("Camera Test", f"Cannot open camera {camera_id}")
            
            test_cap.release()
            
        except Exception as e:
            messagebox.showerror("Camera Test Error", f"Error testing camera: {str(e)}")
    
    def load_image_from_list(self):
        """Load image from current index in image files list"""
        if not self.image_files_list or self.current_image_index >= len(self.image_files_list):
            return
        
        try:
            file_path = self.image_files_list[self.current_image_index]
            image = cv2.imread(file_path)
            
            if image is None:
                messagebox.showerror("Error", f"Cannot read image: {os.path.basename(file_path)}")
                return
            
            self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Reset rotation when loading new image
            self.rotation_angle = 0
            if hasattr(self, 'base_original_image'):
                delattr(self, 'base_original_image')
            
            self.bounding_boxes = []  
            self.display_image()
            self.update_rotation_info()
            
            # Update status and counter
            filename = os.path.basename(file_path)
            folder_name = os.path.basename(self.image_folder_path)
            self.update_status(f"Loaded: {filename}")
            self.image_counter_label.config(text=f"{self.current_image_index + 1}/{len(self.image_files_list)} | {folder_name}/{filename}")
            self.update_statistics()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image from list: {str(e)}")
    
    def prev_image(self):
        """Load previous image in folder"""
        if not self.image_files_list:
            messagebox.showwarning("Warning", "No folder loaded. Use 'Load from Folder' first.")
            return
        
        if self.current_image_index > 0:
            # Save current annotations if any
            if self.bounding_boxes:
                response = messagebox.askyesnocancel("Save Annotations", 
                    "You have unsaved annotations. Do you want to save them before switching images?")
                if response is True:  # Yes - save
                    self.save_current_annotations()
                elif response is None:  # Cancel - don't switch
                    return
                # No - continue without saving
            
            self.current_image_index -= 1
            self.load_image_from_list()
        else:
            messagebox.showinfo("Info", "Already at the first image.")
    
    def next_image(self):
        """Load next image in folder"""
        if not self.image_files_list:
            messagebox.showwarning("Warning", "No folder loaded. Use 'Load from Folder' first.")
            return
        
        if self.current_image_index < len(self.image_files_list) - 1:
            # Save current annotations if any
            if self.bounding_boxes:
                response = messagebox.askyesnocancel("Save Annotations", 
                    "You have unsaved annotations. Do you want to save them before switching images?")
                if response is True:  # Yes - save
                    self.save_current_annotations()
                elif response is None:  # Cancel - don't switch
                    return
                # No - continue without saving
            
            self.current_image_index += 1
            self.load_image_from_list()
        else:
            messagebox.showinfo("Info", "Already at the last image.")
    
    def save_current_annotations(self):
        """Save annotations for current image - FIXED VERSION"""
        if not self.bounding_boxes:
            messagebox.showinfo("Info", "No annotations to save")
            return False
        
        try:
            # Determine current file path and base name
            if self.image_files_list and self.current_image_index < len(self.image_files_list):
                # Folder mode - use current image from list
                current_file_path = self.image_files_list[self.current_image_index]
                base_name = os.path.splitext(current_file_path)[0]
            else:
                # Single image mode - ask user for save location
                base_name = filedialog.asksaveasfilename(
                    title="Save annotations as...",
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
                )
                if not base_name:  # User cancelled
                    return False
                base_name = os.path.splitext(base_name)[0]  # Remove extension
            
            annotation_path = f"{base_name}_annotations.json"
            
            # Prepare annotation data
            annotations_data = {
                "image_file": os.path.basename(self.image_files_list[self.current_image_index]) if self.image_files_list else "single_image",
                "timestamp": datetime.now().isoformat(),
                "image_shape": self.original_image.shape if self.original_image is not None else None,
                "rotation_angle": self.rotation_angle,
                "annotations": []
            }
            
            for bbox in self.bounding_boxes:
                annotation = {
                    "x": bbox.x,
                    "y": bbox.y,
                    "size": bbox.size,
                    "predicted_class": bbox.predicted_class,
                    "confidence": bbox.confidence,
                    "class_name": CLASS_NAMES[bbox.predicted_class] if bbox.predicted_class is not None else None
                }
                annotations_data["annotations"].append(annotation)
            
            # Save to JSON file
            with open(annotation_path, 'w') as f:
                json.dump(annotations_data, f, indent=2)
            
            self.update_status(f"Saved {len(self.bounding_boxes)} annotations to {os.path.basename(annotation_path)}")
            messagebox.showinfo("Success", f"Annotations saved successfully!\n\nFile: {annotation_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to save annotations: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.update_status("Save failed")
            print(f"Save annotations error: {e}")  # For debugging
            return False
    
    def display_image(self):
        """Display image on canvas with zoom and annotations"""
        if self.original_image is None:
            return
        
        # Apply zoom
        height, width = self.original_image.shape[:2]
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)
        
        # Resize image
        resized_image = cv2.resize(self.original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Draw bounding boxes
        display_image = resized_image.copy()
        for bbox in self.bounding_boxes:
            x1, y1, x2, y2 = bbox.get_bbox_coords()
            # Scale coordinates with zoom
            x1_scaled = int(x1 * self.zoom_factor)
            y1_scaled = int(y1 * self.zoom_factor)
            x2_scaled = int(x2 * self.zoom_factor)
            y2_scaled = int(y2 * self.zoom_factor)
            
            # Choose color based on prediction
            color = (255, 0, 0)  # Default red
            if bbox.predicted_class is not None:
                if bbox.predicted_class == 0:  # falciparum
                    color = (255, 0, 0)  # Red
                elif bbox.predicted_class == 1:  # vivax
                    color = (0, 255, 0)  # Green
                else:  # uninfected
                    color = (0, 0, 255)  # Blue
            
            cv2.rectangle(display_image, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
            
            # Add label if classified
            if bbox.predicted_class is not None and bbox.confidence is not None:
                label = f"{CLASS_NAMES[bbox.predicted_class]}: {bbox.confidence:.2f}"
                cv2.putText(display_image, label, (x1_scaled, y1_scaled-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Convert to PIL and display
        pil_image = Image.fromarray(display_image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_click(self, event):
        """Handle canvas click for annotation or start panning"""
        if self.original_image is None:
            return
        
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Store start position and time
        self.start_x = event.x
        self.start_y = event.y
        self.pan_start_x = self.canvas.canvasx(event.x)
        self.pan_start_y = self.canvas.canvasy(event.y)
        self.last_click_time = current_time
        self.is_dragging = False
    
    def on_canvas_drag(self, event):
        """Smooth and accurate canvas panning using absolute scroll fractions"""
        if self.original_image is None:
            return

        drag_distance = ((event.x - self.start_x) ** 2 + (event.y - self.start_y) ** 2) ** 0.5
        if drag_distance > self.drag_threshold:
            self.is_dragging = True
            self.canvas.config(cursor="fleur")

            # Hitung delta mouse
            dx = event.x - self.start_x
            dy = event.y - self.start_y

            # Ambil ukuran canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Hitung scroll fraction relatif ke ukuran viewport
            x_frac = -dx / canvas_width
            y_frac = -dy / canvas_height

            # Ambil posisi scroll sekarang (float 0.0–1.0)
            x0, _ = self.canvas.xview()
            y0, _ = self.canvas.yview()

            # Update scroll position
            self.canvas.xview_moveto(x0 + x_frac)
            self.canvas.yview_moveto(y0 + y_frac)

            # Update posisi awal untuk kelanjutan drag
            self.start_x = event.x
            self.start_y = event.y
    
    def on_canvas_release(self, event):
        """Handle canvas release - either annotation or end panning"""
        if self.original_image is None:
            return
        
        current_time = time.time() * 1000
        
        self.canvas.config(cursor="crosshair")  # Reset cursor
        
        # If this was a short click without much dragging, treat as annotation
        if not self.is_dragging and (current_time - self.last_click_time) < self.click_threshold:
            # Convert canvas coordinates to image coordinates
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            # Account for zoom
            image_x = int(canvas_x / self.zoom_factor)
            image_y = int(canvas_y / self.zoom_factor)
            
            # Check if click is within image bounds
            height, width = self.original_image.shape[:2]
            if 0 <= image_x < width and 0 <= image_y < height:
                # Create new bounding box
                bbox = BoundingBox(image_x, image_y)
                self.bounding_boxes.append(bbox)
                self.display_image()
                self.update_status(f"Added annotation at ({image_x}, {image_y})")
                self.update_statistics()
        
        # Reset dragging state
        self.is_dragging = False
    
    def on_middle_click(self, event):
        """Handle middle click for center panning"""
        if self.original_image is None:
            return
        
        # Center the view on the clicked point
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculate scroll fractions to center on clicked point
        scroll_region = self.canvas.bbox("all")
        if scroll_region:
            total_width = scroll_region[2] - scroll_region[0]
            total_height = scroll_region[3] - scroll_region[1]
            
            if total_width > canvas_width:
                center_x = (canvas_x - canvas_width/2) / (total_width - canvas_width)
                center_x = max(0, min(1, center_x))
                self.canvas.xview_moveto(center_x)
            
            if total_height > canvas_height:
                center_y = (canvas_y - canvas_height/2) / (total_height - canvas_height)
                center_y = max(0, min(1, center_y))
                self.canvas.yview_moveto(center_y)
    
    def on_key_press(self, event):
        """Handle keyboard panning and rotation shortcuts"""
        if self.original_image is None:
            return
        
        pan_step = 50  # pixels to pan
        
        if event.keysym == 'Left':
            self.canvas.xview_scroll(-pan_step, "units")
        elif event.keysym == 'Right':
            self.canvas.xview_scroll(pan_step, "units")
        elif event.keysym == 'Up':
            self.canvas.yview_scroll(-pan_step, "units")
        elif event.keysym == 'Down':
            self.canvas.yview_scroll(pan_step, "units")
        elif event.keysym == 'Prior':  # Page Up
            self.zoom_in()
        elif event.keysym == 'Next':  # Page Down
            self.zoom_out()
        # NEW: Additional rotation shortcuts
        elif event.keysym == 'r' and event.state & 0x4:  # Ctrl+r
            self.rotate_image(90)
        elif event.keysym == 'R' and event.state & 0x4:  # Ctrl+Shift+R
            self.rotate_image(270)
    
    def on_mouse_wheel(self, event):
        """FIXED: Handle mouse wheel for zooming at cursor position"""
        if self.original_image is None:
            return
        
        # Get mouse position on canvas (relative to visible area)
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Store old zoom factor
        old_zoom = self.zoom_factor
        
        # Determine zoom direction and factor
        zoom_in = event.delta > 0 if hasattr(event, 'delta') else False
        zoom_multiplier = 1.1 if zoom_in else 1/1.1
        
        # Apply zoom limits
        new_zoom = self.zoom_factor * zoom_multiplier
        self.zoom_factor = max(0.05, min(10.0, new_zoom))
        
        # Only proceed if zoom actually changed
        if abs(old_zoom - self.zoom_factor) < 0.001:
            return
        
        # Calculate zoom ratio for position adjustment
        zoom_ratio = self.zoom_factor / old_zoom
        
        # Get canvas viewport dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculate mouse position relative to canvas center
        mouse_offset_x = event.x - canvas_width / 2
        mouse_offset_y = event.y - canvas_height / 2
        
        # Update the image display first
        self.display_image()
        
        # Adjust scroll position to zoom at cursor location
        scroll_region = self.canvas.bbox("all")
        if scroll_region:
            total_width = scroll_region[2] - scroll_region[0]
            total_height = scroll_region[3] - scroll_region[1]
            
            # Calculate new scroll position to keep mouse point fixed
            if total_width > canvas_width:
                # Current center position in image coordinates
                current_center_x = canvas_x
                # New center position after zoom
                new_center_x = current_center_x * zoom_ratio
                # Adjust for mouse offset
                target_center_x = new_center_x - mouse_offset_x * (zoom_ratio - 1)
                
                # Convert to scroll fraction
                scroll_x = (target_center_x - canvas_width/2) / (total_width - canvas_width)
                scroll_x = max(0.0, min(1.0, scroll_x))
                self.canvas.xview_moveto(scroll_x)
            
            if total_height > canvas_height:
                # Same calculation for Y axis
                current_center_y = canvas_y
                new_center_y = current_center_y * zoom_ratio
                target_center_y = new_center_y - mouse_offset_y * (zoom_ratio - 1)
                
                scroll_y = (target_center_y - canvas_height/2) / (total_height - canvas_height)
                scroll_y = max(0.0, min(1.0, scroll_y))
                self.canvas.yview_moveto(scroll_y)
        
        # Update status
        self.update_status(f"Zoom: {self.zoom_factor:.1f}x")

# STEP 3: Add this NEW method after on_mouse_wheel() method:

    def on_mouse_wheel_linux(self, event, direction):
        """Handle mouse wheel for Linux systems (Button-4/Button-5)"""
        if self.original_image is None:
            return
        
        # Create a mock event object for compatibility
        class MockEvent:
            def __init__(self, x, y, delta):
                self.x = x
                self.y = y
                self.delta = delta
        
        mock_event = MockEvent(event.x, event.y, 120 if direction > 0 else -120)
        self.on_mouse_wheel(mock_event)
    
    def on_right_click(self, event):
        """Handle right click to remove nearest annotation"""
        if self.original_image is None or not self.bounding_boxes:
            return
        
        # Convert canvas coordinates to image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        image_x = int(canvas_x / self.zoom_factor)
        image_y = int(canvas_y / self.zoom_factor)
        
        # Find nearest bounding box
        min_distance = float('inf')
        nearest_bbox = None
        
        for bbox in self.bounding_boxes:
            distance = ((bbox.x - image_x) ** 2 + (bbox.y - image_y) ** 2) ** 0.5
            if distance < min_distance and distance < IMG_SIZE:  # Within bounding box area
                min_distance = distance
                nearest_bbox = bbox
        
        if nearest_bbox:
            self.bounding_boxes.remove(nearest_bbox)
            self.display_image()
            self.update_status("Removed annotation")
            self.update_statistics()
    
    def zoom_in(self):
        """Zoom in the image"""
        old_zoom = self.zoom_factor
        self.zoom_factor = min(self.zoom_factor * 1.25, 10.0)
        
        if old_zoom != self.zoom_factor:
            # Try to maintain center point during zoom
            self.zoom_at_center()
            self.update_status(f"Zoom: {self.zoom_factor:.1f}x")
    
    def zoom_out(self):
        """Zoom out the image"""
        old_zoom = self.zoom_factor
        self.zoom_factor = max(self.zoom_factor / 1.25, 0.05)
        
        if old_zoom != self.zoom_factor:
            # Try to maintain center point during zoom
            self.zoom_at_center()
            self.update_status(f"Zoom: {self.zoom_factor:.1f}x")
    
    def zoom_at_center(self):
        """IMPROVED: Zoom while maintaining center point"""
        if self.original_image is None:
            return
        
        # Get current viewport center
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get current scroll position (center of viewport in image coordinates)
        x_view = self.canvas.xview()
        y_view = self.canvas.yview()
        
        # Calculate center position in image coordinates before zoom
        if len(x_view) >= 2 and x_view[1] > x_view[0]:
            center_x_fraction = (x_view[0] + x_view[1]) / 2
        else:
            center_x_fraction = 0.5
            
        if len(y_view) >= 2 and y_view[1] > y_view[0]:
            center_y_fraction = (y_view[0] + y_view[1]) / 2
        else:
            center_y_fraction = 0.5
        
        # Update display with new zoom
        self.display_image()
        
        # Restore center position after zoom
        scroll_region = self.canvas.bbox("all")
        if scroll_region:
            total_width = scroll_region[2] - scroll_region[0]
            total_height = scroll_region[3] - scroll_region[1]
            
            # Calculate new scroll position to maintain center
            if total_width > canvas_width:
                view_width = canvas_width / total_width
                new_x_start = center_x_fraction - view_width / 2
                new_x_start = max(0.0, min(1.0 - view_width, new_x_start))
                self.canvas.xview_moveto(new_x_start)
            
            if total_height > canvas_height:
                view_height = canvas_height / total_height
                new_y_start = center_y_fraction - view_height / 2
                new_y_start = max(0.0, min(1.0 - view_height, new_y_start))
                self.canvas.yview_moveto(new_y_start)
    
    def reset_zoom(self):
        """Reset zoom to 1x"""
        self.zoom_factor = 1.0
        self.display_image()
        self.update_status("Zoom reset")
    
    def fit_to_window(self):
        """Fit image to window size"""
        if self.original_image is None:
            return
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Get image dimensions
        img_height, img_width = self.original_image.shape[:2]
        
        # Calculate zoom factor to fit image in window
        zoom_x = canvas_width / img_width
        zoom_y = canvas_height / img_height
        
        # Use the smaller zoom factor to ensure entire image fits
        self.zoom_factor = min(zoom_x, zoom_y) * 0.9  # 0.9 for some padding
        
        self.display_image()
        self.update_status(f"Fit to window: {self.zoom_factor:.1f}x")
    
    def clear_annotations(self):
        """Clear all annotations"""
        self.bounding_boxes = []
        self.display_image()
        self.update_status("All annotations cleared")
        self.update_statistics()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = MalariaNet().to(DEVICE)
            # Try to load default model path - modify this path as needed
            model_path = "malaria_model_fold1.pth"  # Default path
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                self.model.eval()
                self.model_loaded = True
                self.model_status_label.config(text="Model: Loaded", foreground="green")
                self.update_status("Model loaded successfully")
            else:
                self.model_loaded = False
                self.model_status_label.config(text="Model: Not found", foreground="orange")
        except Exception as e:
            self.model_loaded = False
            self.model_status_label.config(text="Model: Error", foreground="red")
            print(f"Model loading error: {e}")
    
    def load_model_dialog(self):
        """Load model from file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("PyTorch models", "*.pth *.pt")]
        )
        
        if file_path:
            try:
                self.model = MalariaNet().to(DEVICE)
                self.model.load_state_dict(torch.load(file_path, map_location=DEVICE))
                self.model.eval()
                self.model_loaded = True
                self.model_status_label.config(text="Model: Loaded", foreground="green")
                self.update_status(f"Model loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.model_loaded = False
    
    def classify_cells(self):
        """Classify all annotated cells using the loaded model"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        if not self.bounding_boxes:
            messagebox.showwarning("Warning", "No cells to classify")
            return
        
        if self.original_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        try:
            self.update_status("Classifying cells...")
            classified_count = 0
            
            for bbox in self.bounding_boxes:
                # Extract cell region
                x1, y1, x2, y2 = bbox.get_bbox_coords()
                height, width = self.original_image.shape[:2]
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, width - IMG_SIZE))
                y1 = max(0, min(y1, height - IMG_SIZE))
                x2 = min(width, x1 + IMG_SIZE)
                y2 = min(height, y1 + IMG_SIZE)
                
                # Extract and preprocess cell
                cell_bgr = cv2.cvtColor(self.original_image[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
                
                if cell_bgr.shape[0] > 0 and cell_bgr.shape[1] > 0:
                    # Apply same preprocessing as training
                    processed_cell = enhance_7channel(cell_bgr)
                    
                    # Convert to tensor
                    tensor = torch.tensor(np.transpose(processed_cell, (2, 0, 1)), dtype=torch.float32)
                    tensor = tensor.unsqueeze(0).to(DEVICE)  # Add batch dimension
                    
                    # Classify
                    with torch.no_grad():
                        output = self.model(tensor)
                        probabilities = torch.softmax(output, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    # Update bounding box with prediction
                    bbox.predicted_class = predicted_class
                    bbox.confidence = confidence
                    classified_count += 1
            
            # Refresh display
            self.display_image()
            self.update_status(f"Classified {classified_count} cells")
            self.update_statistics()
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
    
    def load_annotations(self):
        """Load annotations from JSON file"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        file_path = filedialog.askopenfilename(
            title="Load annotations",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    annotations_data = json.load(f)
                
                # Clear existing annotations
                self.bounding_boxes = []
                
                # Load rotation info if available
                if "rotation_angle" in annotations_data:
                    saved_rotation = annotations_data["rotation_angle"]
                    if saved_rotation != self.rotation_angle:
                        response = messagebox.askyesnocancel("Rotation Mismatch", 
                            f"Saved annotations were made at {saved_rotation}° rotation.\n"
                            f"Current image is at {self.rotation_angle}° rotation.\n\n"
                            f"Apply saved rotation to match annotations?\n\n"
                            f"Yes: Rotate image to {saved_rotation}°\n"
                            f"No: Load annotations as-is (may be misplaced)\n"
                            f"Cancel: Don't load annotations")
                        
                        if response is True:  # Apply rotation
                            rotation_diff = (saved_rotation - self.rotation_angle) % 360
                            if rotation_diff > 180:
                                rotation_diff -= 360
                            if abs(rotation_diff) > 0.1:
                                self.rotate_image(rotation_diff)
                        elif response is None:  # Cancel
                            return
                        # If No: continue with current rotation
                
                # Load annotations
                for ann in annotations_data.get("annotations", []):
                    bbox = BoundingBox(
                        x=ann["x"],
                        y=ann["y"],
                        size=ann.get("size", IMG_SIZE)
                    )
                    bbox.predicted_class = ann.get("predicted_class")
                    bbox.confidence = ann.get("confidence")
                    self.bounding_boxes.append(bbox)
                
                self.display_image()
                self.update_statistics()
                self.update_status(f"Loaded {len(self.bounding_boxes)} annotations")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load annotations: {str(e)}")
    
    def crop_and_save(self):
        """Crop and save all annotated cells with enhanced options"""
        if not self.bounding_boxes:
            messagebox.showwarning("Warning", "No cells to save")
            return
        
        if self.original_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        # Ask for save directory
        save_dir = filedialog.askdirectory(title="Select directory to save cropped cells")
        if not save_dir:
            return
        
        # Ask for naming options
        naming_choice = messagebox.askyesnocancel(
            "Naming Options", 
            "Include classification in filename?\n\nYes: Include class name and confidence\nNo: Simple numbering\nCancel: Abort saving"
        )
        
        if naming_choice is None:  # Cancel
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_count = 0
            
            # Get current image name for context
            if self.image_files_list and self.current_image_index < len(self.image_files_list):
                current_image_name = os.path.splitext(os.path.basename(self.image_files_list[self.current_image_index]))[0]
            else:
                current_image_name = "image"
            
            for i, bbox in enumerate(self.bounding_boxes):
                # Extract cell region
                x1, y1, x2, y2 = bbox.get_bbox_coords()
                height, width = self.original_image.shape[:2]
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, width - IMG_SIZE))
                y1 = max(0, min(y1, height - IMG_SIZE))
                x2 = min(width, x1 + IMG_SIZE)
                y2 = min(height, y1 + IMG_SIZE)
                
                # Extract cell
                cell_region = self.original_image[y1:y2, x1:x2]
                
                if cell_region.shape[0] > 0 and cell_region.shape[1] > 0:
                    # Generate filename based on user choice
                    if naming_choice and bbox.predicted_class is not None:  # Include classification
                        class_name = CLASS_NAMES[bbox.predicted_class]
                        confidence = bbox.confidence
                        filename = f"{current_image_name}_cell_{i:03d}_{class_name}_{confidence:.3f}_{timestamp}.png"
                    else:  # Simple numbering
                        filename = f"{current_image_name}_cell_{i:03d}_{timestamp}.png"
                    
                    filepath = os.path.join(save_dir, filename)
                    
                    # Save cell image
                    cell_pil = Image.fromarray(cell_region)
                    cell_pil.save(filepath)
                    saved_count += 1
            
            # Save session metadata
            metadata = {
                "source_image": current_image_name,
                "timestamp": timestamp,
                "total_cells": len(self.bounding_boxes),
                "saved_cells": saved_count,
                "image_dimensions": self.original_image.shape,
                "rotation_angle": self.rotation_angle,  # NEW: Include rotation info
                "classifications": {}
            }
            
            for bbox in self.bounding_boxes:
                if bbox.predicted_class is not None:
                    class_name = CLASS_NAMES[bbox.predicted_class]
                    if class_name not in metadata["classifications"]:
                        metadata["classifications"][class_name] = 0
                    metadata["classifications"][class_name] += 1
            
            metadata_path = os.path.join(save_dir, f"{current_image_name}_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            messagebox.showinfo("Success", f"Saved {saved_count} cells to {save_dir}\nMetadata saved as JSON file")
            self.update_status(f"Saved {saved_count} cells")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save cells: {str(e)}")
    
    def save_session(self):
        """Save current session (annotations + image)"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save session",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if save_path:
            try:
                session_data = {
                    "timestamp": datetime.now().isoformat(),
                    "image_shape": self.original_image.shape,
                    "annotations": []
                }
                
                for bbox in self.bounding_boxes:
                    annotation = {
                        "x": bbox.x,
                        "y": bbox.y,
                        "size": bbox.size,
                        "predicted_class": bbox.predicted_class,
                        "confidence": bbox.confidence
                    }
                    session_data["annotations"].append(annotation)
                
                with open(save_path, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                self.update_status(f"Session saved: {os.path.basename(save_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save session: {str(e)}")
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def update_statistics(self):
        """Update statistics display"""
        total_annotations = len(self.bounding_boxes)
        
        # Count classifications
        falciparum_count = sum(1 for bbox in self.bounding_boxes if bbox.predicted_class == 0)
        vivax_count = sum(1 for bbox in self.bounding_boxes if bbox.predicted_class == 1)
        uninfected_count = sum(1 for bbox in self.bounding_boxes if bbox.predicted_class == 2)
        
        stats_text = f"Annotations: {total_annotations}\nFalciparum: {falciparum_count}\nVivax: {vivax_count}\nUninfected: {uninfected_count}"
        self.stats_label.config(text=stats_text)
    def start_camera(self):        
        if self.camera_active:
            self.update_status("Camera already active")
            return
        
        try:
            camera_id = int(self.camera_var.get())
            
            with self.camera_lock:
                self.camera = cv2.VideoCapture(camera_id)
                
                if not self.camera.isOpened():
                    raise Exception(f"Cannot open camera {camera_id}")
                
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
                
                # Try to set resolution - don't fail if it doesn't work
                try:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                except:
                    pass  # Use default resolution if custom setting fails
                
                # Test if camera can actually read frames
                ret, test_frame = self.camera.read()
                if not ret or test_frame is None:
                    self.camera.release()
                    raise Exception("Camera opened but cannot read frames")
                
                self.camera_active = True
                self.camera_reconnect_attempts = 0
                
                # Start capture thread
                self.capture_thread = threading.Thread(target=self.camera_capture_loop, daemon=True)
                self.capture_thread.start()
                
                self.camera_status_label.config(text=f"Camera {camera_id}: Connected", foreground="green")
                self.update_status(f"Camera {camera_id} started successfully")
                
        except Exception as e:
            self.camera_status_label.config(text="Camera: Error", foreground="red")
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}\n\nTry:\n1. Check camera connection\n2. Close other camera apps\n3. Try different camera ID\n4. Use 'List Available Cameras'")
    
    def stop_camera(self):
        self.camera_active = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)  # Wait max 2 seconds
        
        with self.camera_lock:
            if self.camera:
                try:
                    self.camera.release()
                except:
                    pass  # Ignore errors when releasing
                self.camera = None
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        self.camera_status_label.config(text="Camera: Disconnected", foreground="red")
        self.update_status("Camera stopped")
    
    def reconnect_camera(self):
        self.stop_camera()
        time.sleep(1)  # Give camera time to release
        self.start_camera()
    
    def camera_capture_loop(self):
        consecutive_failures = 0
        max_failures = 10
        
        while self.camera_active:
            try:
                with self.camera_lock:
                    if self.camera is None or not self.camera.isOpened():
                        break
                    
                    ret, frame = self.camera.read()
                
                if ret and frame is not None:
                    consecutive_failures = 0
                    
                    # Clear old frames and add new one
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            break
                    
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass  # Skip this frame if queue is full
                    
                else:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        self.update_status("Camera connection lost")
                        break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures > max_failures:
                    self.update_status(f"Camera error: {str(e)}")
                    break
                time.sleep(0.1)
        
        # Clean up on exit
        self.camera_active = False
    
    def update_camera_display(self):
        """FIXED: Only update display when camera is active"""
        try:
            # Only update if camera is actively running AND we don't have a static image
            if self.camera_active and not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                
                # Convert to RGB and update display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.original_image = rgb_frame
                self.display_image()
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Display update error: {e}")
        
        # Schedule next update (this will continue running but won't do anything when camera is stopped)
        self.root.after(50, self.update_camera_display)
    
    def capture_image(self):
        """FIXED: Capture current camera frame as static image and stop live feed"""
        if not self.camera_active or self.camera is None:
            messagebox.showwarning("Warning", "Camera is not active")
            return
        
        try:
            # Get latest frame from queue
            frame = None
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            
            if frame is not None:
                # Convert to RGB and store as static image
                self.original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # IMPORTANT: Stop the camera to make it truly static
                self.stop_camera()
                
                # Clear annotations for new capture
                self.bounding_boxes = []
                
                # Reset rotation when capturing new image
                self.rotation_angle = 0
                if hasattr(self, 'base_original_image'):
                    delattr(self, 'base_original_image')
                self.update_rotation_info()
                
                # Display the static image
                self.display_image()
                self.update_status("Image captured and camera stopped - now static image")
                self.update_statistics()
                
                # Clear folder navigation since this is now a captured image
                self.image_folder_path = None
                self.image_files_list = []
                self.current_image_index = 0
                self.image_counter_label.config(text="Captured image (static)")
                
            else:
                messagebox.showwarning("Warning", "No frame available")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture image: {str(e)}")
    
    def load_image(self):
        """Load single image from file"""
        file_path = filedialog.askopenfilename(
            title="Select microscope image",
            filetypes=[
                ("All supported", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("TIFF files", "*.tiff *.tif"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                from PIL import Image
                import numpy as np
                
                # Ganti OpenCV dengan PIL
                pil_image = Image.open(file_path)
                                               
                # Convert ke RGB dan numpy array
                pil_image = pil_image.convert("RGB")
                image = np.array(pil_image)
                
                if image is None or image.size == 0:
                    messagebox.showerror("Error", "Cannot read image file. Please check file format.")
                    return
                
                self.original_image = image  # Sudah dalam format RGB
                self.bounding_boxes = []  # Clear previous annotations
                self.display_image()
                
                # Update status and counter
                filename = os.path.basename(file_path)
                self.update_status(f"Loaded: {filename}")
                self.image_counter_label.config(text=f"Single image: {filename}")
                self.update_statistics()
                
                # Clear folder navigation if was in folder mode
                self.image_folder_path = None
                self.image_files_list = []
                self.current_image_index = 0
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def load_from_folder(self):
        """Load all images from a selected folder"""
        folder_path = filedialog.askdirectory(title="Select folder containing microscope images")
        
        if folder_path:
            try:
                # Find all supported image files in folder
                image_files = []
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(self.supported_formats):
                        image_files.append(os.path.join(folder_path, filename))
                
                if not image_files:
                    messagebox.showwarning("Warning", f"No supported image files found in selected folder.\nSupported formats: {', '.join(self.supported_formats)}")
                    return
                
                # Sort files for consistent ordering
                image_files.sort()
                
                # Store folder information
                self.image_folder_path = folder_path
                self.image_files_list = image_files
                self.current_image_index = 0
                
                # Load first image
                self.load_image_from_list()
                
                messagebox.showinfo("Success", f"Loaded folder with {len(image_files)} images.\nUse Prev/Next buttons to navigate.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load folder: {str(e)}")
def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = MalariaMicroscopeGUI(root)
    
    # Handle window closing
    def on_closing():
        if app.camera_active:
            app.stop_camera()
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_closing()


if __name__ == "__main__":
    print("=== Malaria Detection System - FIXED VERSION ===")
    print("Microscope Interface v1.1")
    print("Biomedical Engineering Lab")
    print("-" * 50)
    print("FIXES INCLUDED:")
    print("✅ Scrollable control panel for better UI")
    print("✅ Enhanced USB webcam support")
    print("✅ Camera connection stability improvements")
    print("✅ Better error handling for external cameras")
    print("✅ Camera troubleshooting tools")
    print("✅ Separate mousewheel handling for controls vs canvas")
    print("✅ Fixed method indentation and structure issues")
    print("-" * 50)
    print("WEBCAM TROUBLESHOOTING:")
    print("1. Try different camera IDs (0, 1, 2, 3)")
    print("2. Use 'List Available Cameras' to detect cameras")
    print("3. Use 'Test Camera' to verify camera functionality")
    print("4. Close other apps using the camera before connecting")
    print("5. Try 'Reconnect Camera' if connection fails")
    print("-" * 50)
    print("Instructions:")
    print("1. IMAGE INPUT OPTIONS:")
    print("   • Start camera for live capture OR")
    print("   • Browse & Load single image OR") 
    print("   • Load from Folder (navigate with Prev/Next)")
    print("2. NAVIGATION CONTROLS:")
    print("   • Mouse drag: Pan around image")
    print("   • Mouse wheel over image: Zoom in/out at cursor") 
    print("   • Mouse wheel over controls: Scroll control panel")
    print("   • Arrow keys: Pan with keyboard")
    print("   • Middle click: Center view on point")
    print("   • Page Up/Down: Zoom in/out")
    print("   • Home key: Reset zoom")
    print("3. ANNOTATION:")
    print("   • Left click (quick): Add annotation")
    print("   • Right click: Remove nearest annotation")
    print("   • Drag: Pan (won't add annotation)")
    print("4. CAMERA CONTROLS:")
    print("   • Select camera ID from dropdown")
    print("   • Start/Stop camera as needed")
    print("   • Capture to convert live feed to static image")
    print("   • Use troubleshooting tools if issues occur")
    print("5. Classify: Load model and classify annotated cells")
    print("6. Save: Crop cells or save annotations as JSON")
    print("7. Batch Mode: Use folder loading for multiple images")
    print("-" * 50)
    
    # Check dependencies
    missing_deps = []
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        from PIL import Image, ImageTk, ImageEnhance
    except ImportError:
        missing_deps.append("Pillow")
    
    if missing_deps:
        print(f"ERROR: Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        sys.exit(1)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    print("Starting application...")
    main()