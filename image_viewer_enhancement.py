#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Viewer Enhancement Module
Menambahkan fitur rotate, zoom/pan, dan fit to window untuk model_val.py

Fitur yang ditambahkan:
1. Zoom in/out dengan mouse wheel atau tombol
2. Pan/drag image dengan mouse
3. Rotate image (90 derajat)
4. Fit to window
5. Reset view
6. Zoom to actual size
"""

import tkinter as tk
from tkinter import ttk
import math
import numpy as np
from PIL import Image, ImageTk
import cv2

class EnhancedImageCanvas:
    """Enhanced canvas dengan fitur zoom, pan, rotate, dan fit to window"""
    
    def __init__(self, parent, canvas):
        self.parent = parent
        self.canvas = canvas
        
        # Image properties
        self.original_image = None
        self.display_image = None
        self.current_image = None
        self.photo = None
        
        # Transform properties
        self.zoom_factor = 1.0
        self.rotation_angle = 0  # dalam derajat
        self.pan_x = 0
        self.pan_y = 0
        
        # Mouse interaction
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.is_dragging = False
        
        # Canvas image item
        self.canvas_image_id = None
        
        # Setup event bindings
        self.setup_bindings()
        
    def setup_bindings(self):
        """Setup mouse and keyboard bindings"""
        # Mouse wheel untuk zoom
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux
        
        # Mouse drag untuk pan
        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # Double click untuk fit to window
        self.canvas.bind("<Double-Button-1>", self.fit_to_window)
        
        # Keyboard shortcuts (perlu focus pada canvas)
        self.canvas.bind("<Key>", self.on_key_press)
        self.canvas.focus_set()
        
    def on_mousewheel(self, event):
        """Handle mouse wheel zoom"""
        if self.original_image is None:
            return
            
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:
            zoom_delta = 1.1
        else:
            zoom_delta = 0.9
            
        # Calculate new zoom factor
        new_zoom = self.zoom_factor * zoom_delta
        
        # Limit zoom range
        if 0.1 <= new_zoom <= 10.0:
            # Get mouse position relative to canvas
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            # Zoom towards mouse position
            self.zoom_to_point(new_zoom, canvas_x, canvas_y)
    
    def zoom_to_point(self, new_zoom, canvas_x, canvas_y):
        """Zoom image toward specific point"""
        if self.original_image is None:
            return
            
        # Calculate zoom ratio
        zoom_ratio = new_zoom / self.zoom_factor
        
        # Update pan to zoom toward mouse position
        self.pan_x = canvas_x - zoom_ratio * (canvas_x - self.pan_x)
        self.pan_y = canvas_y - zoom_ratio * (canvas_y - self.pan_y)
        
        self.zoom_factor = new_zoom
        self.update_display()
    
    def on_mouse_press(self, event):
        """Handle mouse press for dragging"""
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        self.is_dragging = False
        self.canvas.focus_set()  # Set focus untuk keyboard shortcuts
    
    def on_mouse_drag(self, event):
        """Handle mouse drag for panning"""
        if self.original_image is None:
            return
            
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        
        # Only pan if moved significantly (avoid accidental panning on clicks)
        if abs(dx) > 2 or abs(dy) > 2 or self.is_dragging:
            self.is_dragging = True
            self.pan_x += dx
            self.pan_y += dy
            self.update_display()
        
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
    
    def on_mouse_release(self, event):
        """Handle mouse release"""
        self.is_dragging = False
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if self.original_image is None:
            return
            
        key = event.keysym.lower()
        
        if key == 'r':  # Rotate
            self.rotate_image()
        elif key == 'f':  # Fit to window
            self.fit_to_window()
        elif key == 'h':  # Reset view (Home)
            self.reset_view()
        elif key == 'space':  # Actual size (100% zoom)
            self.actual_size()
        elif key == 'plus' or key == 'equal':  # Zoom in
            self.zoom_in()
        elif key == 'minus':  # Zoom out
            self.zoom_out()
    
    def set_image(self, image_array):
        """Set new image to display"""
        if image_array is None:
            return
            
        # Convert numpy array to PIL Image
        if isinstance(image_array, np.ndarray):
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            self.original_image = Image.fromarray(image_array)
        else:
            self.original_image = image_array
            
        # Reset transform
        self.reset_view()
    
    def update_display(self):
        """Update the displayed image with current transforms"""
        if self.original_image is None:
            return
            
        # Apply rotation
        if self.rotation_angle != 0:
            rotated = self.original_image.rotate(self.rotation_angle, expand=True)
        else:
            rotated = self.original_image
            
        # Apply zoom
        if self.zoom_factor != 1.0:
            new_size = (
                int(rotated.width * self.zoom_factor),
                int(rotated.height * self.zoom_factor)
            )
            if new_size[0] > 0 and new_size[1] > 0:
                zoomed = rotated.resize(new_size, Image.Resampling.LANCZOS)
            else:
                zoomed = rotated
        else:
            zoomed = rotated
            
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(zoomed)
        
        # Update canvas
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(
                self.pan_x, self.pan_y, anchor=tk.NW, image=self.photo
            )
        else:
            self.canvas.coords(self.canvas_image_id, self.pan_x, self.pan_y)
            self.canvas.itemconfig(self.canvas_image_id, image=self.photo)
            
        # Update scroll region
        self.update_scroll_region()
    
    def update_scroll_region(self):
        """Update canvas scroll region"""
        if self.photo is None:
            return
            
        # Calculate image bounds with pan offset
        x1 = self.pan_x
        y1 = self.pan_y  
        x2 = self.pan_x + self.photo.width()
        y2 = self.pan_y + self.photo.height()
        
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Extend scroll region to include visible area
        scroll_x1 = min(x1, 0)
        scroll_y1 = min(y1, 0)
        scroll_x2 = max(x2, canvas_width)
        scroll_y2 = max(y2, canvas_height)
        
        self.canvas.configure(scrollregion=(scroll_x1, scroll_y1, scroll_x2, scroll_y2))
    
    def fit_to_window(self, event=None):
        """Fit image to canvas window"""
        if self.original_image is None:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        # Calculate required zoom to fit image
        img_width = self.original_image.width
        img_height = self.original_image.height
        
        # Adjust for rotation
        if self.rotation_angle % 180 != 0:
            img_width, img_height = img_height, img_width
            
        zoom_x = canvas_width / img_width
        zoom_y = canvas_height / img_height
        
        # Use smaller zoom to fit completely
        fit_zoom = min(zoom_x, zoom_y) * 0.95  # 5% margin
        
        # Center image
        self.zoom_factor = fit_zoom
        self.pan_x = (canvas_width - img_width * fit_zoom) / 2
        self.pan_y = (canvas_height - img_height * fit_zoom) / 2
        
        self.update_display()
    
    def actual_size(self):
        """Show image at actual size (100% zoom)"""
        if self.original_image is None:
            return
            
        self.zoom_factor = 1.0
        
        # Center image
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        img_width = self.original_image.width
        img_height = self.original_image.height
        
        # Adjust for rotation
        if self.rotation_angle % 180 != 0:
            img_width, img_height = img_height, img_width
            
        self.pan_x = (canvas_width - img_width) / 2
        self.pan_y = (canvas_height - img_height) / 2
        
        self.update_display()
    
    def zoom_in(self):
        """Zoom in by fixed amount"""
        if self.original_image is None:
            return
            
        new_zoom = self.zoom_factor * 1.2
        if new_zoom <= 10.0:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.zoom_to_point(new_zoom, canvas_width/2, canvas_height/2)
    
    def zoom_out(self):
        """Zoom out by fixed amount"""
        if self.original_image is None:
            return
            
        new_zoom = self.zoom_factor / 1.2
        if new_zoom >= 0.1:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.zoom_to_point(new_zoom, canvas_width/2, canvas_height/2)
    
    def rotate_image(self):
        """Rotate image by 90 degrees"""
        if self.original_image is None:
            return
            
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.update_display()
    
    def reset_view(self):
        """Reset all transforms"""
        if self.original_image is None:
            return
            
        self.zoom_factor = 1.0
        self.rotation_angle = 0
        self.pan_x = 0
        self.pan_y = 0
        
        self.update_display()
        
        # Auto fit to window after reset
        self.canvas.after(100, self.fit_to_window)
    
    def clear(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        self.canvas_image_id = None
        self.original_image = None
        self.photo = None


def create_image_control_panel(parent, enhanced_canvas):
    """Create control panel with image manipulation buttons"""
    control_frame = ttk.LabelFrame(parent, text="Image Controls", padding=5)
    
    # Zoom controls
    zoom_frame = ttk.Frame(control_frame)
    zoom_frame.pack(fill=tk.X, pady=2)
    
    ttk.Button(zoom_frame, text="Zoom In (+)", 
               command=enhanced_canvas.zoom_in).pack(side=tk.LEFT, padx=2)
    ttk.Button(zoom_frame, text="Zoom Out (-)", 
               command=enhanced_canvas.zoom_out).pack(side=tk.LEFT, padx=2)
    
    # View controls
    view_frame = ttk.Frame(control_frame)
    view_frame.pack(fill=tk.X, pady=2)
    
    ttk.Button(view_frame, text="Fit Window (F)", 
               command=enhanced_canvas.fit_to_window).pack(side=tk.LEFT, padx=2)
    ttk.Button(view_frame, text="Actual Size (Space)", 
               command=enhanced_canvas.actual_size).pack(side=tk.LEFT, padx=2)
    
    # Transform controls  
    transform_frame = ttk.Frame(control_frame)
    transform_frame.pack(fill=tk.X, pady=2)
    
    ttk.Button(transform_frame, text="Rotate (R)", 
               command=enhanced_canvas.rotate_image).pack(side=tk.LEFT, padx=2)
    ttk.Button(transform_frame, text="Reset (H)", 
               command=enhanced_canvas.reset_view).pack(side=tk.LEFT, padx=2)
    
    # Zoom info
    info_frame = ttk.Frame(control_frame)
    info_frame.pack(fill=tk.X, pady=2)
    
    zoom_label = ttk.Label(info_frame, text="Zoom: 100%")
    zoom_label.pack(side=tk.LEFT)
    
    # Update zoom label periodically
    def update_zoom_label():
        if enhanced_canvas.original_image is not None:
            zoom_pct = enhanced_canvas.zoom_factor * 100
            rotation = enhanced_canvas.rotation_angle
            zoom_label.config(text=f"Zoom: {zoom_pct:.0f}% | Rotation: {rotation}°")
        control_frame.after(500, update_zoom_label)
    
    update_zoom_label()
    
    # Help text
    help_text = """
Mouse Controls:
• Wheel: Zoom in/out
• Drag: Pan image  
• Double-click: Fit to window

Keyboard Shortcuts:
• R: Rotate 90°
• F: Fit to window
• H: Reset view
• Space: Actual size
• +/-: Zoom in/out
"""
    
    help_label = ttk.Label(control_frame, text=help_text, 
                          font=('TkDefaultFont', 8), justify=tk.LEFT)
    help_label.pack(pady=5)
    
    return control_frame


# Example usage and integration instructions
def example_integration():
    """
    Contoh cara mengintegrasikan dengan model_val.py:
    
    1. Import module ini di bagian atas model_val.py:
       from image_viewer_enhancement import EnhancedImageCanvas, create_image_control_panel
    
    2. Di dalam MalariaValidationGUI.__init__(), setelah setup_gui():
       self.enhanced_canvases = {}
       
    3. Di dalam setup_gui(), setelah membuat canvas untuk setiap tab:
       
       # Untuk ground truth canvas
       self.enhanced_gt = EnhancedImageCanvas(gt_frame, self.gt_canvas)
       self.enhanced_canvases['gt'] = self.enhanced_gt
       
       # Untuk predictions canvas  
       self.enhanced_pred = EnhancedImageCanvas(pred_frame, self.pred_canvas)
       self.enhanced_canvases['pred'] = self.enhanced_pred
       
       # Untuk comparison canvas
       self.enhanced_comp = EnhancedImageCanvas(comp_frame, self.comp_canvas)
       self.enhanced_canvases['comp'] = self.enhanced_comp
    
    4. Tambahkan control panel di control_frame (bagian kiri):
       
       # Di dalam setup_gui(), setelah action_frame:
       image_controls = create_image_control_panel(control_frame, self.enhanced_gt)
       image_controls.pack(fill=tk.X, pady=(0, 10))
    
    5. Update method display_on_canvas():
       
       def display_on_canvas(self, canvas, image):
           if canvas == self.gt_canvas:
               self.enhanced_gt.set_image(image)
           elif canvas == self.pred_canvas:
               self.enhanced_pred.set_image(image)
           elif canvas == self.comp_canvas:
               self.enhanced_comp.set_image(image)
    
    6. Update method display_images():
       
       def display_images(self):
           if self.current_image is None:
               return
           
           # Display ground truth
           if self.ground_truth_annotations:
               gt_image = self.draw_annotations(self.current_image.copy(), 
                                              self.ground_truth_annotations, "ground_truth")
               self.enhanced_gt.set_image(gt_image)
           
           # Display predictions
           if self.predicted_annotations:
               pred_image = self.draw_annotations(self.current_image.copy(), 
                                                self.predicted_annotations, "predictions")
               self.enhanced_pred.set_image(pred_image)
    
    7. Update create_comparison_image():
       
       def show_comparison(self):
           # ... existing code ...
           comp_image = self.create_comparison_image()
           self.enhanced_comp.set_image(comp_image)
           self.notebook.select(2)
    """
    pass


if __name__ == "__main__":
    # Test standalone
    root = tk.Tk()
    root.title("Enhanced Image Canvas Test")
    root.geometry("800x600")
    
    # Create test canvas
    canvas = tk.Canvas(root, bg="white")
    canvas.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
    
    # Create enhanced canvas
    enhanced = EnhancedImageCanvas(root, canvas)
    
    # Create control panel
    controls = create_image_control_panel(root, enhanced)
    controls.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
    
    # Load test image if available
    try:
        test_image = Image.new('RGB', (400, 300), color='red')
        # Draw some test pattern
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([50, 50, 150, 150], fill='blue')
        draw.rectangle([200, 100, 300, 200], fill='green')
        enhanced.set_image(test_image)
    except:
        pass
    
    root.mainloop()