#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Malaria Classification Validation System
Author: Biomedical Engineering Lab
Purpose: Validate and compare automated malaria classification results with ground truth annotations

Features:
1. Load ground truth annotations from text files (bbox check.py format)
2. Load automated predictions from detection program v2.py
3. Visual comparison of annotations
4. Calculate accuracy metrics (precision, recall, F1-score)
5. Generate detailed validation reports
"""

import os
import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import pandas as pd
from datetime import datetime
from collections import defaultdict
import seaborn as sns
from image_viewer_enhancement import EnhancedImageCanvas, create_image_control_panel
from smart_batch_loader import SmartBatchLoader

class ImageMemoryOptimizer:
    """Class to handle memory-efficient image processing"""
    
    @staticmethod
    def get_optimal_size(image_shape, max_dimension=1500, max_memory_mb=50):
        """Calculate optimal image size to prevent memory issues"""
        height, width = image_shape[:2]
        
        # Calculate memory requirement in MB for comparison image (2x width)
        memory_mb = (height * width * 2 * 3) / (1024 * 1024)
        
        if memory_mb <= max_memory_mb and max(height, width * 2) <= max_dimension:
            return height, width
        
        # Calculate scale factor based on memory constraint
        scale_memory = np.sqrt(max_memory_mb / memory_mb)
        
        # Calculate scale factor based on dimension constraint  
        scale_dimension = max_dimension / max(height, width * 2)
        
        # Use the more restrictive scale factor
        scale = min(scale_memory, scale_dimension, 1.0)
        
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        return new_height, new_width
    
    @staticmethod
    def resize_image_safe(image, target_size):
        """Safely resize image with memory check"""
        target_height, target_width = target_size
        
        if image.shape[:2] == (target_height, target_width):
            return image
        
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def scale_annotations(annotations, original_size, new_size):
        """Scale annotation coordinates to match resized image"""
        orig_h, orig_w = original_size
        new_h, new_w = new_size
        
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        scaled_annotations = []
        for ann in annotations:
            scaled_ann = ann.copy()
            x1, y1, x2, y2 = ann['bbox']
            
            scaled_ann['bbox'] = (
                x1 * scale_x,
                y1 * scale_y, 
                x2 * scale_x,
                y2 * scale_y
            )
            scaled_annotations.append(scaled_ann)
        
        return scaled_annotations

class ValidationResults:
    """Class to store and calculate validation metrics - FIXED VERSION"""
    def __init__(self):
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.true_negatives = defaultdict(int)
        self.total_predictions = 0
        self.total_ground_truth = 0
        self.matched_pairs = []
        self.unmatched_predictions = []
        self.unmatched_ground_truth = []
        self.class_names = ['falciparum', 'vivax', 'uninfected']
        
    def add_result(self, predicted_class, actual_class, matched=True):
        """Add a single classification result - FIXED VERSION"""
        classes = ['falciparum', 'vivax', 'uninfected']
        
        print(f"DEBUG: Adding result - Pred: {predicted_class} ({classes[predicted_class]}), Actual: {actual_class} ({classes[actual_class]}), Matched: {matched}")
        
        if matched:
            if predicted_class == actual_class:
                # Correct prediction
                self.true_positives[classes[predicted_class]] += 1
                print(f"  -> True positive for {classes[predicted_class]}")
                
                # Update true negatives for other classes
                for i, cls in enumerate(classes):
                    if i != predicted_class:
                        self.true_negatives[cls] += 1
                        print(f"  -> True negative for {cls}")
            else:
                # Incorrect prediction
                self.false_positives[classes[predicted_class]] += 1
                self.false_negatives[classes[actual_class]] += 1
                print(f"  -> False positive for {classes[predicted_class]}")
                print(f"  -> False negative for {classes[actual_class]}")
        
        # Update totals
        self.total_predictions += 1
        if matched:
            self.total_ground_truth += 1
    
    def add_unmatched_prediction(self, predicted_class):
        """Add unmatched prediction (false positive)"""
        self.false_positives[self.class_names[predicted_class]] += 1
        # True negatives for other classes
        for i, cls in enumerate(self.class_names):
            if i != predicted_class:
                self.true_negatives[cls] += 1
    
    def add_unmatched_ground_truth(self, actual_class):
        """Add unmatched ground truth (false negative)"""
        self.false_negatives[self.class_names[actual_class]] += 1
        # True negatives for other classes
        for i, cls in enumerate(self.class_names):
            if i != actual_class:
                self.true_negatives[cls] += 1
    
    def calculate_metrics(self):
        """Calculate precision, recall, F1-score for each class - FIXED VERSION"""
        metrics = {}
        
        for cls in self.class_names:
            tp = self.true_positives[cls]
            fp = self.false_positives[cls]
            fn = self.false_negatives[cls]
            tn = self.true_negatives[cls]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            }
        
        # Calculate overall accuracy
        total_tp = sum(self.true_positives.values())
        total_samples = total_tp + sum(self.false_positives.values()) + sum(self.false_negatives.values())
        overall_accuracy = total_tp / total_samples if total_samples > 0 else 0
        
        metrics['overall'] = {
            'accuracy': overall_accuracy,
            'total_samples': total_samples,
            'total_correct': total_tp
        }
        
        return metrics

class AnnotationComparison:
    """Class to handle annotation comparison and matching"""
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        intersection_x_min = max(x1_min, x2_min)
        intersection_y_min = max(y1_min, y2_min)
        intersection_x_max = min(x1_max, x2_max)
        intersection_y_max = min(y1_max, y2_max)
        
        if intersection_x_max <= intersection_x_min or intersection_y_max <= intersection_y_min:
            return 0.0
        
        intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def match_annotations(self, ground_truth_boxes, predicted_boxes):
        """Match ground truth and predicted annotations based on IoU"""
        matches = []
        unmatched_gt = list(range(len(ground_truth_boxes)))
        unmatched_pred = list(range(len(predicted_boxes)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(ground_truth_boxes), len(predicted_boxes)))
        for i, gt_box in enumerate(ground_truth_boxes):
            for j, pred_box in enumerate(predicted_boxes):
                iou_matrix[i, j] = self.calculate_iou(gt_box['bbox'], pred_box['bbox'])
        
        # Find best matches using greedy approach
        while len(unmatched_gt) > 0 and len(unmatched_pred) > 0:
            # Find maximum IoU in remaining boxes
            max_iou = 0
            best_match = None
            
            for i in unmatched_gt:
                for j in unmatched_pred:
                    if iou_matrix[i, j] > max_iou and iou_matrix[i, j] >= self.iou_threshold:
                        max_iou = iou_matrix[i, j]
                        best_match = (i, j)
            
            if best_match is None:
                break
            
            i, j = best_match
            matches.append({
                'gt_idx': i,
                'pred_idx': j,
                'iou': max_iou,
                'gt_class': ground_truth_boxes[i]['class'],
                'pred_class': predicted_boxes[j]['class'],
                'gt_bbox': ground_truth_boxes[i]['bbox'],
                'pred_bbox': predicted_boxes[j]['bbox']
            })
            
            unmatched_gt.remove(i)
            unmatched_pred.remove(j)
        
        return matches, unmatched_gt, unmatched_pred

class EnhancedFolderBatchLoader:
    """Enhanced batch loader untuk handle struktur folder yang kompleks"""
    
    def __init__(self, validation_gui):
        self.validation_gui = validation_gui
        self.image_extensions = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp']
    
    def show_enhanced_batch_dialog(self):
        """Dialog untuk enhanced batch processing dengan folder mapping - FIXED LAYOUT"""
        dialog = tk.Toplevel(self.validation_gui.root)
        dialog.title("Enhanced Batch Processing")
        dialog.geometry("1400x800")  # Reduced height to fit screen better
        dialog.transient(self.validation_gui.root)
        dialog.grab_set()
        
        # Create main scrollable frame
        main_canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_frame = ttk.Frame(scrollable_frame, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Enhanced Batch Processing", 
                font=('Arial', 14, 'bold')).pack(pady=5)
        
        # Folder selection section
        folder_frame = ttk.LabelFrame(main_frame, text="Folder Configuration", padding=10)
        folder_frame.pack(fill=tk.X, pady=5)
        
        # Image + Predictions folder
        ttk.Label(folder_frame, text="Images + Predictions Folder:").pack(anchor=tk.W)
        img_pred_frame = ttk.Frame(folder_frame)
        img_pred_frame.pack(fill=tk.X, pady=2)
        
        self.img_pred_folder_var = tk.StringVar()
        ttk.Entry(img_pred_frame, textvariable=self.img_pred_folder_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(img_pred_frame, text="Browse", 
                command=lambda: self.browse_folder(self.img_pred_folder_var, "Select Images+Predictions Folder")).pack(side=tk.RIGHT, padx=(5,0))
        
        # Ground Truth folder
        ttk.Label(folder_frame, text="Ground Truth Folder:").pack(anchor=tk.W, pady=(10,0))
        gt_folder_frame = ttk.Frame(folder_frame)
        gt_folder_frame.pack(fill=tk.X, pady=2)
        
        self.gt_folder_var = tk.StringVar()
        ttk.Entry(gt_folder_frame, textvariable=self.gt_folder_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(gt_folder_frame, text="Browse", 
                command=lambda: self.browse_folder(self.gt_folder_var, "Select Ground Truth Folder")).pack(side=tk.RIGHT, padx=(5,0))
        
        # File pattern configuration - COMPACT VERSION
        pattern_frame = ttk.LabelFrame(main_frame, text="File Patterns", padding=5)
        pattern_frame.pack(fill=tk.X, pady=5)
        
        # Image extensions - single line
        ext_frame = ttk.Frame(pattern_frame)
        ext_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ext_frame, text="Image extensions:", width=20).pack(side=tk.LEFT)
        self.img_ext_var = tk.StringVar(value="jpg,jpeg,png,tif,tiff,bmp")
        ttk.Entry(ext_frame, textvariable=self.img_ext_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Prediction pattern - compact
        pred_frame = ttk.Frame(pattern_frame)
        pred_frame.pack(fill=tk.X, pady=2)
        ttk.Label(pred_frame, text="Prediction pattern:", width=20).pack(side=tk.LEFT)
        
        self.pred_pattern_var = tk.StringVar(value="custom_suffix")
        ttk.Radiobutton(pred_frame, text="Same name", variable=self.pred_pattern_var, 
                    value="same_name").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(pred_frame, text="Custom suffix:", variable=self.pred_pattern_var, 
                    value="custom_suffix").pack(side=tk.LEFT, padx=5)
        
        self.pred_suffix_var = tk.StringVar(value="_annotations.json")
        ttk.Entry(pred_frame, textvariable=self.pred_suffix_var, width=20).pack(side=tk.LEFT, padx=5)
        
        # GT pattern - compact
        gt_frame = ttk.Frame(pattern_frame)
        gt_frame.pack(fill=tk.X, pady=2)
        ttk.Label(gt_frame, text="GT pattern:", width=20).pack(side=tk.LEFT)
        
        self.gt_pattern_var = tk.StringVar(value="same_name")
        ttk.Radiobutton(gt_frame, text="Same name", variable=self.gt_pattern_var, 
                    value="same_name").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(gt_frame, text="Custom suffix:", variable=self.gt_pattern_var, 
                    value="custom_suffix").pack(side=tk.LEFT, padx=5)
        
        self.gt_suffix_var = tk.StringVar(value="_gt.txt")
        ttk.Entry(gt_frame, textvariable=self.gt_suffix_var, width=20).pack(side=tk.LEFT, padx=5)
        
        # Scan and preview section - REDUCED HEIGHT
        preview_frame = ttk.LabelFrame(main_frame, text="File Matching Preview", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Control buttons for preview
        preview_control_frame = ttk.Frame(preview_frame)
        preview_control_frame.pack(fill=tk.X, pady=(0,5))
        
        ttk.Button(preview_control_frame, text="Scan Folders", 
                command=self.scan_and_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(preview_control_frame, text="Auto-detect Patterns", 
                command=self.auto_detect_patterns).pack(side=tk.LEFT, padx=5)
        
        # Results tree - REDUCED HEIGHT
        columns = ('Image', 'Predictions', 'Ground_Truth', 'Status')
        self.preview_tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=8)  # Reduced from 12 to 8
        
        # Configure columns
        for i, col in enumerate(columns):
            self.preview_tree.heading(col, text=col)
            if col == 'Image':
                self.preview_tree.column(col, width=200, minwidth=150)
            else:
                self.preview_tree.column(col, width=150, minwidth=100)
        
        self.preview_tree.column('#0', width=0, stretch=False)
        
        # Add scrollbars
        tree_frame = ttk.Frame(preview_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        tree_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.preview_tree.yview)
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        
        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Statistics display
        self.stats_label = ttk.Label(preview_frame, text="No scan performed yet")
        self.stats_label.pack(pady=2)
        
        # Processing options - COMPACT
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding=5)
        options_frame.pack(fill=tk.X, pady=5)
        
        # Single row for all options
        options_row = ttk.Frame(options_frame)
        options_row.pack(fill=tk.X)
        
        self.process_subdirs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_row, text="Process subdirectories", 
                    variable=self.process_subdirs_var).pack(side=tk.LEFT)
        
        self.ignore_missing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_row, text="Skip missing files", 
                    variable=self.ignore_missing_var).pack(side=tk.LEFT, padx=(20,0))
        
        self.save_individual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_row, text="Save individual reports", 
                    variable=self.save_individual_var).pack(side=tk.LEFT, padx=(20,0))
        
        # Default species - COMPACT
        species_frame = ttk.LabelFrame(main_frame, text="Default Parasite Species", padding=5)
        species_frame.pack(fill=tk.X, pady=5)
        
        species_row = ttk.Frame(species_frame)
        species_row.pack(fill=tk.X)
        
        self.default_species_var = tk.StringVar(value="falciparum")
        ttk.Radiobutton(species_row, text="Falciparum", variable=self.default_species_var, 
                    value="falciparum").pack(side=tk.LEFT)
        ttk.Radiobutton(species_row, text="Vivax", variable=self.default_species_var, 
                    value="vivax").pack(side=tk.LEFT, padx=(20,0))
        ttk.Radiobutton(species_row, text="WBC", variable=self.default_species_var, 
                    value="WBC").pack(side=tk.LEFT, padx=(20,0))
        
        # Action buttons - ALWAYS VISIBLE AT BOTTOM
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10, side=tk.BOTTOM)  # Pack at bottom
        
        ttk.Button(button_frame, text="Start Processing", 
                command=lambda: self.start_enhanced_processing(dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export File List", 
                command=self.export_file_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Pack the canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configure tags for tree
        self.preview_tree.tag_configure('complete', background='lightgreen')
        self.preview_tree.tag_configure('missing_gt', background='lightyellow')
        self.preview_tree.tag_configure('missing_pred', background='lightblue')
        self.preview_tree.tag_configure('missing_both', background='lightcoral')
        
        # Bind mousewheel to canvas for scrolling
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        return dialog
    
    def browse_folder(self, var, title):
        """Browse for folder and update variable"""
        folder = filedialog.askdirectory(title=title)
        if folder:
            var.set(folder)
    
    def browse_mapping_file(self):
        """Browse for mapping file"""
        file_path = filedialog.askopenfilename(
            title="Select mapping file",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")]
        )
        if file_path:
            self.mapping_file_var.set(file_path)
    
    def update_pattern_visibility(self):
        """Update visibility of pattern configuration frames"""
        # Prediction pattern
        if self.pred_pattern_var.get() == "custom_suffix":
            self.pred_suffix_frame.pack(fill=tk.X, pady=2)
        else:
            self.pred_suffix_frame.pack_forget()
        
        # Ground truth pattern
        if self.gt_pattern_var.get() == "custom_suffix":
            self.gt_suffix_frame.pack(fill=tk.X, pady=2)
            self.mapping_frame.pack_forget()
        elif self.gt_pattern_var.get() == "mapping_file":
            self.gt_suffix_frame.pack_forget()
            self.mapping_frame.pack(fill=tk.X, pady=2)
        else:
            self.gt_suffix_frame.pack_forget()
            self.mapping_frame.pack_forget()
    
    def scan_and_preview(self):
        """Scan folders and show matching preview"""
        if not self.img_pred_folder_var.get() or not self.gt_folder_var.get():
            messagebox.showwarning("Warning", "Please select both folders first")
            return
        
        try:
            matches = self.find_file_matches()
            self.update_preview_tree(matches)
            self.update_statistics(matches)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error scanning folders: {str(e)}")
    
    def auto_detect_patterns(self):
        """Auto-detect file naming patterns"""
        if not self.img_pred_folder_var.get() or not self.gt_folder_var.get():
            messagebox.showwarning("Warning", "Please select both folders first")
            return
        
        try:
            patterns = self.detect_naming_patterns()
            
            if patterns:
                # Show detected patterns dialog
                self.show_pattern_suggestions(patterns)
            else:
                messagebox.showinfo("Auto-detect", "No clear patterns detected. Please configure manually.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting patterns: {str(e)}")
    
    def detect_naming_patterns(self):
        """Detect common naming patterns between folders"""
        img_pred_folder = self.img_pred_folder_var.get()
        gt_folder = self.gt_folder_var.get()
        
        # Get sample files
        image_files = []
        json_files = []
        txt_files = []
        
        # Scan image+pred folder
        for root, dirs, files in os.walk(img_pred_folder):
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(f'.{ext}') for ext in self.image_extensions):
                    image_files.append(file)
                elif file_lower.endswith('.json'):
                    json_files.append(file)
        
        # Scan GT folder
        for root, dirs, files in os.walk(gt_folder):
            for file in files:
                if file.lower().endswith('.txt'):
                    txt_files.append(file)
        
        # Analyze patterns
        patterns = {
            'pred_same_name': 0,
            'pred_suffix': [],
            'gt_same_name': 0,
            'gt_suffix': [],
            'potential_mappings': []
        }
        
        # Check prediction patterns
        for img in image_files[:20]:  # Sample first 20
            img_base = os.path.splitext(img)[0]
            
            # Check for same name JSON
            if f"{img_base}.json" in json_files:
                patterns['pred_same_name'] += 1
            
            # Check for suffixed JSON
            for json_file in json_files:
                json_base = os.path.splitext(json_file)[0]
                if img_base in json_base and img_base != json_base:
                    suffix = json_base.replace(img_base, '') + '.json'
                    if suffix not in patterns['pred_suffix']:
                        patterns['pred_suffix'].append(suffix)
        
        # Check GT patterns
        for img in image_files[:20]:
            img_base = os.path.splitext(img)[0]
            
            # Check for same name TXT
            if f"{img_base}.txt" in txt_files:
                patterns['gt_same_name'] += 1
            
            # Check for suffixed TXT
            for txt_file in txt_files:
                txt_base = os.path.splitext(txt_file)[0]
                if img_base in txt_base and img_base != txt_base:
                    suffix = txt_base.replace(img_base, '') + '.txt'
                    if suffix not in patterns['gt_suffix']:
                        patterns['gt_suffix'].append(suffix)
        
        return patterns
    
    def show_pattern_suggestions(self, patterns):
        """Show detected pattern suggestions"""
        suggestion_dialog = tk.Toplevel()
        suggestion_dialog.title("Auto-detected Patterns")
        suggestion_dialog.geometry("500x400")
        suggestion_dialog.transient(self.validation_gui.root)
        suggestion_dialog.grab_set()
        
        main_frame = ttk.Frame(suggestion_dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Detected Patterns:", font=('Arial', 12, 'bold')).pack(pady=5)
        
        text_widget = tk.Text(main_frame, height=15, width=60)
        text_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=text_scroll.set)
        
        # Add detected patterns
        text_widget.insert(tk.END, "PREDICTION PATTERNS:\n")
        if patterns['pred_same_name'] > 0:
            text_widget.insert(tk.END, f"• Same name matches: {patterns['pred_same_name']} files\n")
        if patterns['pred_suffix']:
            text_widget.insert(tk.END, f"• Common suffixes: {', '.join(patterns['pred_suffix'])}\n")
        
        text_widget.insert(tk.END, "\nGROUND TRUTH PATTERNS:\n")
        if patterns['gt_same_name'] > 0:
            text_widget.insert(tk.END, f"• Same name matches: {patterns['gt_same_name']} files\n")
        if patterns['gt_suffix']:
            text_widget.insert(tk.END, f"• Common suffixes: {', '.join(patterns['gt_suffix'])}\n")
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Apply suggestions button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def apply_suggestions():
            # Apply most common patterns
            if patterns['pred_same_name'] > 0:
                self.pred_pattern_var.set("same_name")
            elif patterns['pred_suffix']:
                self.pred_pattern_var.set("custom_suffix")
                self.pred_suffix_var.set(patterns['pred_suffix'][0])
            
            if patterns['gt_same_name'] > 0:
                self.gt_pattern_var.set("same_name")
            elif patterns['gt_suffix']:
                self.gt_pattern_var.set("custom_suffix")
                self.gt_suffix_var.set(patterns['gt_suffix'][0])
            
            self.update_pattern_visibility()
            suggestion_dialog.destroy()
        
        ttk.Button(button_frame, text="Apply Suggestions", command=apply_suggestions).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=suggestion_dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def find_file_matches(self):
        """Find matching files based on current configuration"""
        img_pred_folder = self.img_pred_folder_var.get()
        gt_folder = self.gt_folder_var.get()
        
        if not img_pred_folder or not gt_folder:
            return []
        
        matches = []
        
        # Get all image files
        image_files = []
        if self.process_subdirs_var.get():
            for root, dirs, files in os.walk(img_pred_folder):
                for file in files:
                    if any(file.lower().endswith(f'.{ext}') for ext in self.img_ext_var.get().split(',')):
                        image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(img_pred_folder):
                if any(file.lower().endswith(f'.{ext}') for ext in self.img_ext_var.get().split(',')):
                    image_files.append(os.path.join(img_pred_folder, file))
        
        # Load name mapping if using mapping file
        name_mapping = {}
        if self.gt_pattern_var.get() == "mapping_file" and self.mapping_file_var.get():
            try:
                import pandas as pd
                df = pd.read_csv(self.mapping_file_var.get())
                if 'image_name' in df.columns and 'gt_name' in df.columns:
                    name_mapping = dict(zip(df['image_name'], df['gt_name']))
            except Exception as e:
                print(f"Error loading mapping file: {e}")
        
        # Find matches for each image
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            img_base = os.path.splitext(img_name)[0]
            
            match_info = {
                'image_path': img_path,
                'image_name': img_name,
                'pred_path': None,
                'pred_name': None,
                'gt_path': None,
                'gt_name': None,
                'status': 'Missing both'
            }
            
            # Find prediction file
            pred_name = self.get_prediction_filename(img_name)
            pred_path = os.path.join(os.path.dirname(img_path), pred_name)
            if os.path.exists(pred_path):
                match_info['pred_path'] = pred_path
                match_info['pred_name'] = pred_name
            
            # Find ground truth file
            gt_name = self.get_ground_truth_filename(img_name, name_mapping)
            gt_path = self.find_ground_truth_file(gt_name, gt_folder)
            if gt_path:
                match_info['gt_path'] = gt_path
                match_info['gt_name'] = os.path.basename(gt_path)
            
            # Update status
            has_pred = match_info['pred_path'] is not None
            has_gt = match_info['gt_path'] is not None
            
            if has_pred and has_gt:
                match_info['status'] = 'Complete'
            elif has_pred:
                match_info['status'] = 'Missing GT'
            elif has_gt:
                match_info['status'] = 'Missing Pred'
            else:
                match_info['status'] = 'Missing both'
            
            matches.append(match_info)
        
        return matches
    
    def get_prediction_filename(self, image_name):
        """Get expected prediction filename based on pattern"""
        img_base = os.path.splitext(image_name)[0]
        
        if self.pred_pattern_var.get() == "same_name":
            return f"{img_base}.json"
        else:  # custom_suffix
            suffix = self.pred_suffix_var.get()
            return f"{img_base}{suffix}"
    
    def get_ground_truth_filename(self, image_name, name_mapping=None):
        """Get expected ground truth filename based on pattern"""
        img_base = os.path.splitext(image_name)[0]
        
        if self.gt_pattern_var.get() == "same_name":
            return f"{img_base}.txt"
        elif self.gt_pattern_var.get() == "custom_suffix":
            suffix = self.gt_suffix_var.get()
            return f"{img_base}{suffix}"
        else:  # mapping_file
            if name_mapping and image_name in name_mapping:
                return name_mapping[image_name]
            else:
                return f"{img_base}.txt"  # fallback
    
    def find_ground_truth_file(self, gt_name, gt_folder):
        """Find ground truth file in GT folder (including subdirectories)"""
        if self.process_subdirs_var.get():
            for root, dirs, files in os.walk(gt_folder):
                if gt_name in files:
                    return os.path.join(root, gt_name)
        else:
            gt_path = os.path.join(gt_folder, gt_name)
            if os.path.exists(gt_path):
                return gt_path
        
        return None
    
    def update_preview_tree(self, matches):
        """Update preview tree with matches - FIXED VERSION"""
        # Clear existing items
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        
        # Add matches
        for match in matches:
            pred_display = match['pred_name'] if match['pred_name'] else 'MISSING'
            gt_display = match['gt_name'] if match['gt_name'] else 'MISSING'
            
            # Set color based on status
            if match['status'] == 'Complete':
                tag = 'complete'
                status_display = '✓ Complete'
            elif match['status'] == 'Missing GT':
                tag = 'missing_gt'
                status_display = '⚠ Missing GT'
            elif match['status'] == 'Missing Pred':
                tag = 'missing_pred'
                status_display = '⚠ Missing Pred'
            else:
                tag = 'missing_both'
                status_display = '✗ Missing both'
            
            # Insert item dengan tags yang benar
            item = self.preview_tree.insert('', 'end', 
                values=(match['image_name'], pred_display, gt_display, status_display),
                tags=(tag,))  # Gunakan tags parameter, bukan set method
        
        # Configure tags dengan warna yang benar
        self.preview_tree.tag_configure('complete', background='lightgreen')
        self.preview_tree.tag_configure('missing_gt', background='lightyellow') 
        self.preview_tree.tag_configure('missing_pred', background='lightblue')
        self.preview_tree.tag_configure('missing_both', background='lightcoral')
    
    def update_statistics(self, matches):
        """Update statistics display"""
        total = len(matches)
        complete = len([m for m in matches if m['status'] == 'Complete'])
        missing_gt = len([m for m in matches if m['status'] == 'Missing GT'])
        missing_pred = len([m for m in matches if m['status'] == 'Missing Pred'])
        missing_both = len([m for m in matches if m['status'] == 'Missing both'])
        
        stats_text = f"Total: {total} | Complete: {complete} | Missing GT: {missing_gt} | Missing Pred: {missing_pred} | Missing both: {missing_both}"
        self.stats_label.config(text=stats_text)
    
    def export_file_list(self):
        """Export file matching list to CSV"""
        matches = self.find_file_matches()
        if not matches:
            messagebox.showwarning("Warning", "No matches found. Please scan folders first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export file list",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                import pandas as pd
                
                data = []
                for match in matches:
                    data.append({
                        'image_path': match['image_path'],
                        'image_name': match['image_name'],
                        'pred_path': match['pred_path'] or '',
                        'pred_name': match['pred_name'] or '',
                        'gt_path': match['gt_path'] or '',
                        'gt_name': match['gt_name'] or '',
                        'status': match['status']
                    })
                
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
                
                messagebox.showinfo("Success", f"File list exported to {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export file list: {str(e)}")
    
    def start_enhanced_processing(self, dialog):
        """Start enhanced batch processing"""
        matches = self.find_file_matches()
        if not matches:
            messagebox.showwarning("Warning", "No files found. Please scan folders first.")
            return
        
        # Filter based on options
        if self.ignore_missing_var.get():
            complete_matches = [m for m in matches if m['status'] == 'Complete']
        else:
            complete_matches = [m for m in matches if m['pred_path'] and m['gt_path']]
        
        if not complete_matches:
            messagebox.showwarning("Warning", "No complete matches found for processing.")
            return
        
        # Confirm processing
        result = messagebox.askyesno("Confirm Processing", 
            f"Found {len(complete_matches)} files ready for processing.\n\n"
            f"Complete matches: {len([m for m in matches if m['status'] == 'Complete'])}\n"
            f"Missing GT: {len([m for m in matches if m['status'] == 'Missing GT'])}\n"
            f"Missing Pred: {len([m for m in matches if m['status'] == 'Missing Pred'])}\n\n"
            f"Proceed with batch processing?")
        
        if not result:
            return
        
        # Close dialog and start processing
        dialog.destroy()
        
        # Start processing with progress dialog
        self.process_enhanced_batch(complete_matches)
    
    def process_enhanced_batch(self, matches):
        """Process the enhanced batch with progress tracking"""
        total_matches = len(matches)
        
        # Create progress dialog
        progress_dialog = tk.Toplevel(self.validation_gui.root)
        progress_dialog.title("Processing Enhanced Batch")
        progress_dialog.geometry("600x300")
        progress_dialog.transient(self.validation_gui.root)
        progress_dialog.grab_set()
        
        progress_frame = ttk.Frame(progress_dialog, padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress info
        ttk.Label(progress_frame, text=f"Processing {total_matches} files...", 
                 font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Current file label
        self.current_file_label = ttk.Label(progress_frame, text="Initializing...")
        self.current_file_label.pack(pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', maximum=total_matches)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_progress_label = ttk.Label(progress_frame, text="")
        self.status_progress_label.pack(pady=5)
        
        # Detailed progress text
        self.progress_text = tk.Text(progress_frame, height=10, width=70)
        progress_scroll = ttk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=self.progress_text.yview)
        self.progress_text.configure(yscrollcommand=progress_scroll.set)
        
        self.progress_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        progress_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize batch results
        batch_results = {
            'processed_files': [],
            'failed_files': [],
            'missing_gt_files': [],
            'missing_pred_files': [],
            'overall_metrics': self.validation_gui.validation_results.__class__(),
            'individual_results': {},
            'folder_info': {
                'img_pred_folder': self.img_pred_folder_var.get(),
                'gt_folder': self.gt_folder_var.get(),
                'total_scanned': len(matches),
                'processing_started': datetime.now()
            }
        }
        
        # Process each match
        for i, match in enumerate(matches):
            try:
                # Update progress
                self.progress_bar['value'] = i
                self.current_file_label.config(text=f"Processing: {match['image_name']}")
                self.status_progress_label.config(text=f"{i+1}/{total_matches}")
                progress_dialog.update()
                
                # Process single file
                result = self.process_single_enhanced_match(match, batch_results)
                
                # Update progress text
                if result['success']:
                    accuracy = result['metrics']['overall']['accuracy'] if result['metrics'] else 0
                    self.progress_text.insert(tk.END, f"[OK] {match['image_name']} - Accuracy: {accuracy:.1%}\n")
                    batch_results['processed_files'].append(result)
                else:
                    self.progress_text.insert(tk.END, f"[FAIL] {match['image_name']} - {result['error']}\n")
                    batch_results['failed_files'].append({
                        'file': match['image_path'],
                        'error': result['error']
                    })
                
                self.progress_text.see(tk.END)
                progress_dialog.update()
                
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.progress_text.insert(tk.END, f"[ERROR] {match['image_name']} - {error_msg}\n")
                batch_results['failed_files'].append({
                    'file': match['image_path'],
                    'error': error_msg
                })
                self.progress_text.see(tk.END)
                progress_dialog.update()
        
        # Finalize
        self.progress_bar['value'] = total_matches
        self.current_file_label.config(text="Generating reports...")
        progress_dialog.update()
        
        # Generate comprehensive report
        self.generate_enhanced_batch_report(batch_results)
        
        # Close progress dialog and show results
        progress_dialog.destroy()
        self.show_enhanced_completion_dialog(batch_results)
    
    def process_single_enhanced_match(self, match, batch_results):
        """Process a single enhanced match - FIXED VERSION"""
        try:
            from PIL import Image
            import numpy as np
            
            result = {
                'image_path': match['image_path'],
                'gt_path': match['gt_path'],
                'pred_path': match['pred_path'],
                'success': False,
                'metrics': None,
                'error': None
            }
            
            # Load image (not needed for validation, but kept for consistency)
            try:
                pil_image = Image.open(match['image_path'])
                pil_image = pil_image.convert("RGB")
                image = np.array(pil_image)
                
                if image is None:
                    result['error'] = "Cannot read image file"
                    return result
            except Exception as e:
                result['error'] = f"Image loading error: {str(e)}"
                return result
            
            # Load ground truth annotations
            gt_annotations = self.load_ground_truth_enhanced(match['gt_path'])
            if not gt_annotations:
                result['error'] = "No valid ground truth annotations"
                return result
            
            # Load prediction annotations
            pred_annotations = self.load_predictions_enhanced(match['pred_path'])
            if not pred_annotations:
                result['error'] = "No valid prediction annotations"
                return result
            
            # IMPORTANT: Use the same comparison instance and IoU threshold as single validation
            # Create a fresh ValidationResults for this file
            file_validation = ValidationResults()
            
            # Use the same IoU threshold from the main GUI
            comparison = AnnotationComparison(iou_threshold=self.validation_gui.comparison.iou_threshold)
            
            # Match annotations using the same method as single validation
            matches_found, unmatched_gt, unmatched_pred = comparison.match_annotations(
                gt_annotations, pred_annotations
            )
            
            # Process matches EXACTLY like single validation
            for match_result in matches_found:
                # Add to file-specific results
                file_validation.add_result(match_result['pred_class'], match_result['gt_class'], matched=True)
                # Add to batch overall results
                batch_results['overall_metrics'].add_result(match_result['pred_class'], match_result['gt_class'], matched=True)
            
            # Store detailed match information for debugging
            file_validation.matched_pairs = matches_found
            file_validation.unmatched_predictions = [pred_annotations[i] for i in unmatched_pred]
            file_validation.unmatched_ground_truth = [gt_annotations[i] for i in unmatched_gt]
            
            # Calculate metrics
            file_metrics = file_validation.calculate_metrics()
            
            result.update({
                'success': True,
                'metrics': file_metrics,
                'matches': matches_found,
                'unmatched_gt': len(unmatched_gt),
                'unmatched_pred': len(unmatched_pred),
                'gt_count': len(gt_annotations),
                'pred_count': len(pred_annotations)
            })
            
            # Save individual report if requested
            if self.save_individual_var.get():
                self.save_enhanced_individual_report(result, match)
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
    
    def load_ground_truth_enhanced(self, file_path):
        """Load ground truth annotations for enhanced processing - FIXED FOR ACTUAL FORMAT"""
        try:
            annotations = []
            default_species = self.default_species_var.get()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Skip first line if it's metadata (image dimensions)
                start_line = 1 if len(lines) > 0 and lines[0].count(',') == 2 else 0
                
                for line_num, line in enumerate(lines[start_line:], start=start_line+1):
                    try:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        parts = line.split(',')
                        print(f"Line {line_num}: {len(parts)} parts - {parts}")  # Debug output
                        
                        # Check for different possible formats
                        if len(parts) >= 7:  # Format: ID, Label, Comment, Type, ?, X, Y
                            _, label = parts[0], parts[1]
                            x1, y1 = map(float, [parts[5], parts[6]])
                            
                            # Convert center point to bounding box
                            box_size = 128
                            half = box_size / 2
                            x1_box = x1 - half
                            y1_box = y1 - half
                            x2_box = x1_box + box_size
                            y2_box = y1_box + box_size
                            
                            # Map label to class
                            if label in ['Parasite']:
                                if default_species == 'falciparum':
                                    class_idx = 0
                                    class_name = 'falciparum'
                                elif default_species == 'vivax':
                                    class_idx = 1
                                    class_name = 'vivax'
                                else:  # default_species == 'WBC'
                                    class_idx = 2
                                    class_name = 'uninfected'
                            elif label in ['Parasitized']:
                                class_idx = 1
                                class_name = 'vivax'
                            elif label in ['White_Blood_Cell', 'WBC', 'Uninfected']:
                                class_idx = 2
                                class_name = 'uninfected'
                            else:
                                print(f"Warning: Unknown label '{label}' at line {line_num}, mapping to uninfected")
                                class_idx = 2
                                class_name = 'uninfected'
                            
                            annotations.append({
                                'bbox': (x1_box, y1_box, x2_box, y2_box),
                                'class': class_idx,
                                'class_name': class_name,
                                'label': label,
                                'line_number': line_num
                            })
                            print(f"Added annotation: {label} -> {class_name} at ({x1}, {y1})")
                            
                        elif len(parts) >= 9:  # Original format assumption
                            _, label, *_, x1, y1, x2, y2 = parts
                            x1, y1 = map(float, [x1, y1])
                            
                            # Convert center point to bounding box
                            box_size = 128
                            half = box_size / 2
                            x1_box = x1 - half
                            y1_box = y1 - half
                            x2_box = x1_box + box_size
                            y2_box = y1_box + box_size
                            
                            # Map label to class (same logic as above)
                            if label in ['Parasite']:
                                if default_species == 'falciparum':
                                    class_idx = 0
                                    class_name = 'falciparum'
                                elif default_species == 'vivax':
                                    class_idx = 1
                                    class_name = 'vivax'
                                else:
                                    class_idx = 2
                                    class_name = 'uninfected'
                            elif label in ['Parasitized']:
                                class_idx = 1
                                class_name = 'vivax'
                            elif label in ['White_Blood_Cell', 'WBC', 'Uninfected']:
                                class_idx = 2
                                class_name = 'uninfected'
                            else:
                                print(f"Warning: Unknown label '{label}' at line {line_num}, mapping to uninfected")
                                class_idx = 2
                                class_name = 'uninfected'
                            
                            annotations.append({
                                'bbox': (x1_box, y1_box, x2_box, y2_box),
                                'class': class_idx,
                                'class_name': class_name,
                                'label': label,
                                'line_number': line_num
                            })
                            print(f"Added annotation: {label} -> {class_name} at ({x1}, {y1})")
                        else:
                            print(f"Skipping line {line_num}: insufficient parts ({len(parts)})")
                            continue
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line {line_num} in {file_path}: {e}")
                        print(f"Line content: {line}")
                        continue
            
            print(f"Loaded {len(annotations)} ground truth annotations from {file_path}")
            
            # Debug: Print label distribution
            if annotations:
                label_counts = {}
                for ann in annotations:
                    original_label = ann['label']
                    mapped_class = ann['class_name']
                    key = f"{original_label} -> {mapped_class}"
                    label_counts[key] = label_counts.get(key, 0) + 1
                
                print("Label mapping distribution:")
                for mapping, count in label_counts.items():
                    print(f"  {mapping}: {count}")
            else:
                print("No annotations loaded - check file format and parsing logic")
            
            return annotations
            
        except Exception as e:
            print(f"Error loading ground truth file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def load_predictions_enhanced(self, file_path):
        """Load prediction annotations for enhanced processing"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            annotations = []
            
            if 'annotations' in data:
                for ann in data['annotations']:
                    try:
                        x, y = ann['x'], ann['y']
                        size = ann.get('size', 128)
                        predicted_class = ann.get('predicted_class')
                        confidence = ann.get('confidence', 0)
                        
                        if predicted_class is not None:
                            # Convert center point to bounding box
                            half = size / 2
                            x1, y1 = x - half, y - half
                            x2, y2 = x1 + size, y1 + size
                            
                            class_names = ['falciparum', 'vivax', 'uninfected']
                            annotations.append({
                                'bbox': (x1, y1, x2, y2),
                                'class': predicted_class,
                                'class_name': class_names[predicted_class],
                                'confidence': confidence
                            })
                    except (KeyError, TypeError, IndexError) as e:
                        print(f"Error parsing annotation in {file_path}: {e}")
                        continue
            
            return annotations
            
        except Exception as e:
            print(f"Error loading predictions file {file_path}: {e}")
            return []
    
    def save_enhanced_individual_report(self, result, match):
        """Save individual report for enhanced processing"""
        try:
            base_name = os.path.splitext(match['image_name'])[0]
            report_path = os.path.join(os.path.dirname(match['image_path']), f"{base_name}_enhanced_validation.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"ENHANCED VALIDATION REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # File paths
                f.write("FILE INFORMATION:\n")
                f.write(f"Image: {match['image_path']}\n")
                f.write(f"Ground Truth: {match['gt_path']}\n")
                f.write(f"Predictions: {match['pred_path']}\n\n")
                
                # Processing results
                f.write("PROCESSING RESULTS:\n")
                f.write(f"GT Annotations: {result['gt_count']}\n")
                f.write(f"Predictions: {result['pred_count']}\n")
                f.write(f"Successful Matches: {len(result['matches'])}\n")
                f.write(f"Unmatched GT: {result['unmatched_gt']}\n")
                f.write(f"Unmatched Pred: {result['unmatched_pred']}\n\n")
                
                # Metrics
                if result['metrics']:
                    overall = result['metrics']['overall']
                    f.write("PERFORMANCE METRICS:\n")
                    f.write(f"Overall Accuracy: {overall['accuracy']:.1%}\n\n")
                    
                    class_names = ['falciparum', 'vivax', 'uninfected']
                    for class_name in class_names:
                        if class_name in result['metrics']:
                            m = result['metrics'][class_name]
                            f.write(f"{class_name.upper()}:\n")
                            f.write(f"  Precision: {m['precision']:.1%}\n")
                            f.write(f"  Recall: {m['recall']:.1%}\n")
                            f.write(f"  F1-Score: {m['f1_score']:.1%}\n\n")
                
                # Detailed matches
                if result['matches']:
                    f.write("DETAILED MATCHES:\n")
                    for i, match_detail in enumerate(result['matches']):
                        gt_class = ['falciparum', 'vivax', 'uninfected'][match_detail['gt_class']]
                        pred_class = ['falciparum', 'vivax', 'uninfected'][match_detail['pred_class']]
                        iou = match_detail['iou']
                        correct = "CORRECT" if match_detail['gt_class'] == match_detail['pred_class'] else "INCORRECT"
                        f.write(f"{i+1:2d}. {correct} - GT: {gt_class} | Pred: {pred_class} | IoU: {iou:.3f}\n")
                
        except Exception as e:
            print(f"Error saving enhanced individual report: {e}")
    
    def generate_enhanced_batch_report(self, batch_results):
        """Generate comprehensive enhanced batch report"""
        try:
            # Create report directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = f"enhanced_batch_validation_{timestamp}"
            os.makedirs(report_dir, exist_ok=True)
            
            # Store report directory for access
            self.last_enhanced_report_dir = report_dir
            
            # Calculate overall metrics
            overall_metrics = batch_results['overall_metrics'].calculate_metrics()
            
            # Generate main summary
            self.create_enhanced_summary_report(report_dir, batch_results, overall_metrics)
            
            # Generate CSV exports
            self.create_enhanced_csv_exports(report_dir, batch_results)
            
            # Create visualizations
            if overall_metrics:
                self.create_enhanced_visualizations(report_dir, overall_metrics, batch_results)
            
            # Create folder structure analysis
            self.create_folder_analysis_report(report_dir, batch_results)
            
        except Exception as e:
            print(f"Error generating enhanced batch report: {e}")
            raise
    
    def create_enhanced_summary_report(self, report_dir, batch_results, overall_metrics):
        """Create enhanced summary report"""
        summary_path = os.path.join(report_dir, "enhanced_batch_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED BATCH VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Folder information
            folder_info = batch_results['folder_info']
            f.write("FOLDER CONFIGURATION:\n")
            f.write(f"Images + Predictions: {folder_info['img_pred_folder']}\n")
            f.write(f"Ground Truth: {folder_info['gt_folder']}\n")
            f.write(f"Total Files Scanned: {folder_info['total_scanned']}\n")
            f.write(f"Processing Started: {folder_info['processing_started'].strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Processing statistics
            total_processed = len(batch_results['processed_files'])
            total_failed = len(batch_results['failed_files'])
            success_rate = total_processed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0
            
            f.write("PROCESSING STATISTICS:\n")
            f.write(f"Successfully Processed: {total_processed}\n")
            f.write(f"Failed to Process: {total_failed}\n")
            f.write(f"Success Rate: {success_rate:.1%}\n\n")
            
            # Overall performance
            if overall_metrics:
                overall = overall_metrics['overall']
                f.write("OVERALL PERFORMANCE:\n")
                f.write(f"Overall Accuracy: {overall['accuracy']:.1%}\n")
                f.write(f"Total Samples: {overall['total_samples']}\n")
                f.write(f"Correct Predictions: {overall['total_correct']}\n\n")
                
                # Per-class metrics
                class_names = ['falciparum', 'vivax', 'uninfected']
                f.write("PER-CLASS PERFORMANCE:\n")
                for class_name in class_names:
                    if class_name in overall_metrics:
                        m = overall_metrics[class_name]
                        f.write(f"\n{class_name.upper()}:\n")
                        f.write(f"  Precision: {m['precision']:.1%}\n")
                        f.write(f"  Recall: {m['recall']:.1%}\n")
                        f.write(f"  F1-Score: {m['f1_score']:.1%}\n")
                        f.write(f"  True Positives: {m['true_positives']}\n")
                        f.write(f"  False Positives: {m['false_positives']}\n")
                        f.write(f"  False Negatives: {m['false_negatives']}\n")
            
            # File-level analysis
            if batch_results['processed_files']:
                accuracies = [r['metrics']['overall']['accuracy'] for r in batch_results['processed_files'] 
                            if r['metrics']]
                
                if accuracies:
                    f.write(f"\nFILE-LEVEL ACCURACY ANALYSIS:\n")
                    f.write(f"Mean Accuracy: {np.mean(accuracies):.1%}\n")
                    f.write(f"Standard Deviation: {np.std(accuracies):.1%}\n")
                    f.write(f"Minimum Accuracy: {np.min(accuracies):.1%}\n")
                    f.write(f"Maximum Accuracy: {np.max(accuracies):.1%}\n")
                    f.write(f"Median Accuracy: {np.median(accuracies):.1%}\n")
            
            # Error analysis
            if batch_results['failed_files']:
                f.write(f"\nERROR ANALYSIS:\n")
                error_types = {}
                for failed in batch_results['failed_files']:
                    error = failed['error']
                    error_types[error] = error_types.get(error, 0) + 1
                
                for error, count in error_types.items():
                    f.write(f"  {error}: {count} files\n")
    
    def create_enhanced_csv_exports(self, report_dir, batch_results):
        """Create CSV exports for enhanced batch results"""
        try:
            import pandas as pd
            
            # File summary
            file_data = []
            for result in batch_results['processed_files']:
                basename = os.path.basename(result['image_path'])
                metrics = result['metrics']
                
                row = {
                    'filename': basename,
                    'image_path': result['image_path'],
                    'gt_path': result['gt_path'],
                    'pred_path': result['pred_path'],
                    'gt_count': result['gt_count'],
                    'pred_count': result['pred_count'],
                    'matches': len(result['matches']),
                    'unmatched_gt': result['unmatched_gt'],
                    'unmatched_pred': result['unmatched_pred'],
                    'overall_accuracy': metrics['overall']['accuracy'] if metrics else 0
                }
                
                # Add per-class metrics
                class_names = ['falciparum', 'vivax', 'uninfected']
                for class_name in class_names:
                    if metrics and class_name in metrics:
                        m = metrics[class_name]
                        row[f'{class_name}_precision'] = m['precision']
                        row[f'{class_name}_recall'] = m['recall']
                        row[f'{class_name}_f1'] = m['f1_score']
                        row[f'{class_name}_tp'] = m['true_positives']
                        row[f'{class_name}_fp'] = m['false_positives']
                        row[f'{class_name}_fn'] = m['false_negatives']
                    else:
                        row[f'{class_name}_precision'] = 0
                        row[f'{class_name}_recall'] = 0
                        row[f'{class_name}_f1'] = 0
                        row[f'{class_name}_tp'] = 0
                        row[f'{class_name}_fp'] = 0
                        row[f'{class_name}_fn'] = 0
                
                file_data.append(row)
            
            if file_data:
                df_files = pd.DataFrame(file_data)
                df_files.to_csv(os.path.join(report_dir, "enhanced_file_summary.csv"), index=False)
            
            # Detailed match data
            match_data = []
            for result in batch_results['processed_files']:
                basename = os.path.basename(result['image_path'])
                for match in result['matches']:
                    class_names = ['falciparum', 'vivax', 'uninfected']
                    match_data.append({
                        'filename': basename,
                        'image_path': result['image_path'],
                        'gt_class': class_names[match['gt_class']],
                        'pred_class': class_names[match['pred_class']],
                        'iou': match['iou'],
                        'correct': match['gt_class'] == match['pred_class'],
                        'gt_bbox': str(match['gt_bbox']),
                        'pred_bbox': str(match['pred_bbox'])
                    })
            
            if match_data:
                df_matches = pd.DataFrame(match_data)
                df_matches.to_csv(os.path.join(report_dir, "enhanced_detailed_matches.csv"), index=False)
            
            # Failed files
            if batch_results['failed_files']:
                failed_data = []
                for failed in batch_results['failed_files']:
                    failed_data.append({
                        'filename': os.path.basename(failed['file']),
                        'filepath': failed['file'],
                        'error': failed['error']
                    })
                
                df_failed = pd.DataFrame(failed_data)
                df_failed.to_csv(os.path.join(report_dir, "enhanced_failed_files.csv"), index=False)
                
        except Exception as e:
            print(f"Error creating enhanced CSV exports: {e}")
    
    def create_enhanced_visualizations(self, report_dir, overall_metrics, batch_results):
        """Create enhanced visualizations - FIXED VERSION"""
        try:
            # Use the same visualization methods as single validation
            self.validation_gui.create_confusion_matrix(report_dir, overall_metrics)
            self.validation_gui.create_performance_plots(report_dir, overall_metrics)
            
            # Create CORRECTED folder comparison plot
            self.create_folder_comparison_plot_fixed(report_dir, batch_results)
            
        except Exception as e:
            print(f"Error creating enhanced visualizations: {e}")

    def create_folder_comparison_plot(self, report_dir, batch_results):
        """Create folder-specific comparison plots - FIXED VERSION"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # FIXED: Accuracy distribution
            accuracies = []
            for r in batch_results['processed_files']:
                if r['metrics'] and 'overall' in r['metrics']:
                    accuracy = r['metrics']['overall']['accuracy']
                    if accuracy is not None:
                        accuracies.append(accuracy)
            
            if accuracies:
                ax1.hist(accuracies, bins=min(20, len(accuracies)), edgecolor='black', alpha=0.7, color='skyblue')
                mean_acc = np.mean(accuracies)
                ax1.axvline(mean_acc, color='red', linestyle='--', 
                        label=f'Mean: {mean_acc:.1%}')
                ax1.set_xlabel('Accuracy')
                ax1.set_ylabel('Number of Files')
                ax1.set_title('File Accuracy Distribution')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No accuracy data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('File Accuracy Distribution - No Data')
            
            # Processing success rate
            total_processed = len(batch_results['processed_files'])
            total_failed = len(batch_results['failed_files'])
            
            if total_processed > 0 or total_failed > 0:
                labels = ['Successful', 'Failed']
                sizes = [total_processed, total_failed]
                colors = ['lightgreen', 'lightcoral']
                
                # Filter out zero values
                non_zero_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
                if non_zero_data:
                    labels, sizes, colors = zip(*non_zero_data)
                    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                else:
                    ax2.text(0.5, 0.5, 'No processing data', ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'No processing data', ha='center', va='center', transform=ax2.transAxes)
            
            ax2.set_title('Processing Success Rate')
            
            # FIXED: Annotation counts comparison
            gt_counts = [r['gt_count'] for r in batch_results['processed_files'] if r['success']]
            pred_counts = [r['pred_count'] for r in batch_results['processed_files'] if r['success']]
            
            if gt_counts and pred_counts and len(gt_counts) == len(pred_counts):
                ax3.scatter(gt_counts, pred_counts, alpha=0.6, s=50)
                max_count = max(max(gt_counts) if gt_counts else 0, max(pred_counts) if pred_counts else 0)
                if max_count > 0:
                    ax3.plot([0, max_count], [0, max_count], 'r--', alpha=0.5, label='Perfect Match')
                    ax3.legend()
                ax3.set_xlabel('Ground Truth Count')
                ax3.set_ylabel('Prediction Count')
                ax3.set_title('GT vs Prediction Annotation Counts')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No annotation count data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('GT vs Prediction Counts - No Data')
            
            # FIXED: Match efficiency calculation
            match_ratios = []
            for r in batch_results['processed_files']:
                if r['success'] and r['gt_count'] > 0 and r['pred_count'] > 0:
                    # Calculate match efficiency as percentage of annotations that were matched
                    matches = len(r['matches']) if 'matches' in r else 0
                    total_possible_matches = min(r['gt_count'], r['pred_count'])  # Maximum possible matches
                    
                    if total_possible_matches > 0:
                        match_efficiency = matches / total_possible_matches
                        match_ratios.append(match_efficiency)
            
            if match_ratios:
                ax4.hist(match_ratios, bins=min(15, len(match_ratios)), edgecolor='black', alpha=0.7, color='lightsteelblue')
                mean_ratio = np.mean(match_ratios)
                ax4.axvline(mean_ratio, color='red', linestyle='--', 
                        label=f'Mean: {mean_ratio:.1%}')
                ax4.set_xlabel('Match Efficiency Ratio')
                ax4.set_ylabel('Number of Files')
                ax4.set_title('Annotation Matching Efficiency')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_xlim(0, 1)
            else:
                ax4.text(0.5, 0.5, 'No match efficiency data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Matching Efficiency - No Data')
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "enhanced_analysis_plots.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Debug: Print summary statistics
            print(f"\nDEBUG: Enhanced analysis plot data:")
            print(f"  Processed files: {len(batch_results['processed_files'])}")
            print(f"  Failed files: {len(batch_results['failed_files'])}")
            print(f"  Files with accuracy data: {len(accuracies)}")
            print(f"  Files with annotation counts: GT={len(gt_counts)}, Pred={len(pred_counts)}")
            print(f"  Files with match efficiency: {len(match_ratios)}")
            if accuracies:
                print(f"  Accuracy range: {min(accuracies):.1%} - {max(accuracies):.1%}")
                print(f"  Mean accuracy: {np.mean(accuracies):.1%}")
            
        except Exception as e:
            print(f"Error creating folder comparison plot: {e}")
            import traceback
            traceback.print_exc()
    
    def create_folder_analysis_report(self, report_dir, batch_results):
        """Create folder structure analysis report"""
        try:
            analysis_path = os.path.join(report_dir, "folder_structure_analysis.txt")
            
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write("FOLDER STRUCTURE ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Folder paths
                folder_info = batch_results['folder_info']
                f.write("FOLDER CONFIGURATION:\n")
                f.write(f"Images + Predictions Folder: {folder_info['img_pred_folder']}\n")
                f.write(f"Ground Truth Folder: {folder_info['gt_folder']}\n\n")
                
                # File distribution analysis
                f.write("FILE DISTRIBUTION ANALYSIS:\n")
                
                # Count files by subfolder
                img_subfolders = {}
                gt_subfolders = {}
                
                for result in batch_results['processed_files']:
                    # Image subfolder
                    img_rel_path = os.path.relpath(result['image_path'], folder_info['img_pred_folder'])
                    img_subfolder = os.path.dirname(img_rel_path) if os.path.dirname(img_rel_path) else "root"
                    img_subfolders[img_subfolder] = img_subfolders.get(img_subfolder, 0) + 1
                    
                    # GT subfolder
                    gt_rel_path = os.path.relpath(result['gt_path'], folder_info['gt_folder'])
                    gt_subfolder = os.path.dirname(gt_rel_path) if os.path.dirname(gt_rel_path) else "root"
                    gt_subfolders[gt_subfolder] = gt_subfolders.get(gt_subfolder, 0) + 1
                
                f.write("Images + Predictions Distribution:\n")
                for subfolder, count in sorted(img_subfolders.items()):
                    f.write(f"  {subfolder}: {count} files\n")
                
                f.write("\nGround Truth Distribution:\n")
                for subfolder, count in sorted(gt_subfolders.items()):
                    f.write(f"  {subfolder}: {count} files\n")
                
                # Performance by subfolder
                f.write(f"\nPERFORMANCE BY SUBFOLDER:\n")
                subfolder_performance = {}
                
                for result in batch_results['processed_files']:
                    img_rel_path = os.path.relpath(result['image_path'], folder_info['img_pred_folder'])
                    img_subfolder = os.path.dirname(img_rel_path) if os.path.dirname(img_rel_path) else "root"
                    
                    if img_subfolder not in subfolder_performance:
                        subfolder_performance[img_subfolder] = {'accuracies': [], 'count': 0}
                    
                    if result['metrics']:
                        accuracy = result['metrics']['overall']['accuracy']
                        subfolder_performance[img_subfolder]['accuracies'].append(accuracy)
                    subfolder_performance[img_subfolder]['count'] += 1
                
                for subfolder, perf_data in sorted(subfolder_performance.items()):
                    f.write(f"\n{subfolder}:\n")
                    f.write(f"  Files processed: {perf_data['count']}\n")
                    if perf_data['accuracies']:
                        mean_acc = np.mean(perf_data['accuracies'])
                        f.write(f"  Mean accuracy: {mean_acc:.1%}\n")
                        f.write(f"  Min accuracy: {np.min(perf_data['accuracies']):.1%}\n")
                        f.write(f"  Max accuracy: {np.max(perf_data['accuracies']):.1%}\n")
        
        except Exception as e:
            print(f"Error creating folder analysis report: {e}")
    
    def show_enhanced_completion_dialog(self, batch_results):
        """Show enhanced completion dialog - FIXED VERSION"""
        completion_dialog = tk.Toplevel(self.validation_gui.root)
        completion_dialog.title("Enhanced Batch Processing Complete")
        completion_dialog.geometry("700x600")
        completion_dialog.transient(self.validation_gui.root)
        completion_dialog.grab_set()
        
        main_frame = ttk.Frame(completion_dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Enhanced Batch Processing Complete", 
                font=('Arial', 14, 'bold')).pack(pady=5)
        
        # Summary text
        summary_text = tk.Text(main_frame, height=25, width=80)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=summary_text.yview)
        summary_text.configure(yscrollcommand=scrollbar.set)
        
        # Generate CORRECTED summary content
        total_processed = len(batch_results['processed_files'])
        total_failed = len(batch_results['failed_files'])
        folder_info = batch_results['folder_info']
        
        summary_text.insert(tk.END, "ENHANCED BATCH PROCESSING RESULTS\n")
        summary_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Folder information
        summary_text.insert(tk.END, "FOLDER CONFIGURATION:\n")
        summary_text.insert(tk.END, f"Images + Predictions: {os.path.basename(folder_info['img_pred_folder'])}\n")
        summary_text.insert(tk.END, f"Ground Truth: {os.path.basename(folder_info['gt_folder'])}\n")
        summary_text.insert(tk.END, f"Total Files Scanned: {folder_info['total_scanned']}\n\n")
        
        # Processing results
        success_rate = total_processed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0
        summary_text.insert(tk.END, "PROCESSING RESULTS:\n")
        summary_text.insert(tk.END, f"Successfully Processed: {total_processed} files\n")
        summary_text.insert(tk.END, f"Failed to Process: {total_failed} files\n")
        summary_text.insert(tk.END, f"Processing Success Rate: {success_rate:.1%}\n\n")
        
        # CORRECTED Performance statistics
        if batch_results['processed_files']:
            accuracies = []
            for r in batch_results['processed_files']:
                if r['metrics'] and 'overall' in r['metrics']:
                    acc = r['metrics']['overall']['accuracy']
                    if acc is not None:
                        accuracies.append(acc)
            
            if accuracies:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                min_acc = np.min(accuracies)
                max_acc = np.max(accuracies)
                
                summary_text.insert(tk.END, "PERFORMANCE STATISTICS:\n")
                summary_text.insert(tk.END, f"Mean Accuracy: {mean_acc:.1%}\n")
                summary_text.insert(tk.END, f"Standard Deviation: {std_acc:.1%}\n")
                summary_text.insert(tk.END, f"Best Performance: {max_acc:.1%}\n")
                summary_text.insert(tk.END, f"Worst Performance: {min_acc:.1%}\n\n")
                
                # CORRECTED Top and bottom performers
                file_performance = []
                for r in batch_results['processed_files']:
                    if r['metrics'] and 'overall' in r['metrics']:
                        acc = r['metrics']['overall']['accuracy']
                        if acc is not None:
                            basename = os.path.basename(r['image_path'])
                            file_performance.append((basename, acc))
                
                if file_performance:
                    file_performance.sort(key=lambda x: x[1], reverse=True)
                    
                    summary_text.insert(tk.END, "TOP 5 BEST PERFORMING FILES:\n")
                    for i, (filename, acc) in enumerate(file_performance[:5]):
                        summary_text.insert(tk.END, f"{i+1}. {filename} - {acc:.1%}\n")
                    
                    summary_text.insert(tk.END, "\nTOP 5 WORST PERFORMING FILES:\n")
                    for i, (filename, acc) in enumerate(file_performance[-5:]):
                        summary_text.insert(tk.END, f"{i+1}. {filename} - {acc:.1%}\n")
                    
                    # Files needing attention (accuracy < 70%)
                    low_performers = [f for f in file_performance if f[1] < 0.7]
                    if low_performers:
                        summary_text.insert(tk.END, f"\nFILES NEEDING ATTENTION (< 70% accuracy): {len(low_performers)}\n")
                        for filename, acc in low_performers[:10]:  # Show first 10
                            summary_text.insert(tk.END, f"  • {filename} - {acc:.1%}\n")
                        if len(low_performers) > 10:
                            summary_text.insert(tk.END, f"  ... and {len(low_performers)-10} more\n")
            else:
                summary_text.insert(tk.END, "PERFORMANCE STATISTICS:\n")
                summary_text.insert(tk.END, "No valid accuracy data found.\n\n")
        
        # Calculate and display CORRECTED overall metrics from batch results
        try:
            overall_metrics = batch_results['overall_metrics'].calculate_metrics()
            if overall_metrics and 'overall' in overall_metrics:
                summary_text.insert(tk.END, "OVERALL BATCH METRICS:\n")
                overall = overall_metrics['overall']
                summary_text.insert(tk.END, f"Batch Overall Accuracy: {overall['accuracy']:.1%}\n")
                summary_text.insert(tk.END, f"Total Matched Annotations: {overall['total_samples']}\n")
                summary_text.insert(tk.END, f"Correct Matches: {overall['total_correct']}\n\n")
                
                # Per-class metrics
                class_names = ['falciparum', 'vivax', 'uninfected']
                summary_text.insert(tk.END, "PER-CLASS METRICS:\n")
                for class_name in class_names:
                    if class_name in overall_metrics:
                        m = overall_metrics[class_name]
                        summary_text.insert(tk.END, f"{class_name.upper()}:\n")
                        summary_text.insert(tk.END, f"  Precision: {m['precision']:.1%}\n")
                        summary_text.insert(tk.END, f"  Recall: {m['recall']:.1%}\n")
                        summary_text.insert(tk.END, f"  F1-Score: {m['f1_score']:.1%}\n\n")
        except Exception as e:
            summary_text.insert(tk.END, f"Error calculating overall metrics: {e}\n\n")
        
        # Error analysis (unchanged)
        if batch_results['failed_files']:
            summary_text.insert(tk.END, f"FAILED FILES ANALYSIS:\n")
            error_types = {}
            for failed in batch_results['failed_files']:
                error = failed['error']
                error_types[error] = error_types.get(error, 0) + 1
            
            for error, count in error_types.items():
                summary_text.insert(tk.END, f"  {error}: {count} files\n")
            
            # Show first few failed files
            summary_text.insert(tk.END, f"\nFailed Files Details:\n")
            for failed in batch_results['failed_files'][:5]:
                basename = os.path.basename(failed['file'])
                summary_text.insert(tk.END, f"  • {basename}: {failed['error']}\n")
            if len(batch_results['failed_files']) > 5:
                summary_text.insert(tk.END, f"  ... and {len(batch_results['failed_files'])-5} more\n")
        
        # Report location (unchanged)
        summary_text.insert(tk.END, f"\nREPORT LOCATION:\n")
        summary_text.insert(tk.END, f"{self.last_enhanced_report_dir}\n\n")
        
        # Recommendations (improved)
        summary_text.insert(tk.END, f"RECOMMENDATIONS:\n")
        if total_failed > 0:
            summary_text.insert(tk.END, f"• Review failed files and fix file path/format issues\n")
        
        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            if mean_acc < 0.8:
                summary_text.insert(tk.END, f"• Mean accuracy below 80% - consider model improvements\n")
            if std_acc > 0.2:
                summary_text.insert(tk.END, f"• High accuracy variance detected - investigate inconsistent files\n")
        
        # Pack the text widget
        summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons (unchanged)
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def open_report_folder():
            import subprocess, platform
            if platform.system() == "Windows":
                subprocess.Popen(['explorer', self.last_enhanced_report_dir])
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(['open', self.last_enhanced_report_dir])
            else:  # Linux
                subprocess.Popen(['xdg-open', self.last_enhanced_report_dir])
        
        ttk.Button(button_frame, text="Open Report Folder", command=open_report_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=completion_dialog.destroy).pack(side=tk.RIGHT, padx=5)


class MalariaValidationGUI:
    """Main GUI class for malaria classification validation"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Malaria Classification Validation System")
        self.root.geometry("1600x900")
        
        # Fixed Class mapping
        self.class_names = ['falciparum', 'vivax', 'uninfected']
        self.class_colors = {
            'falciparum': 'red',
            'vivax': 'green', 
            'uninfected': 'blue',
            'Parasite': 'red',      # For falciparum ground truth files
            'Parasitized': 'green'    # For vivax ground truth files  
        }
        
        # Data storage
        self.current_image = None
        self.ground_truth_annotations = []
        self.predicted_annotations = []
        self.validation_results = ValidationResults()
        self.comparison = AnnotationComparison()
        
        # UI setup
        self.setup_gui()
        self.smart_loader = SmartBatchLoader(self)
        self.enhanced_canvases = {}
        
    def setup_gui(self):
        """Setup the main GUI interface"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel (left side)
        control_frame = ttk.LabelFrame(main_frame, text="Validation Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # File loading section
        file_frame = ttk.LabelFrame(control_frame, text="Load Data", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image_fixed).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Ground Truth (TXT)", command=self.load_ground_truth_fixed).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Predictions (JSON)", command=self.load_predictions_fixed).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Batch Loader", command=self.load_enhanced_batch_folder).pack(fill=tk.X, pady=2)
        
        # Validation settings
        settings_frame = ttk.LabelFrame(control_frame, text="Validation Settings", padding=5)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(settings_frame, text="IoU Threshold:").pack(anchor=tk.W)
        self.iou_threshold_var = tk.DoubleVar(value=0.5)
        iou_scale = ttk.Scale(settings_frame, from_=0.1, to=0.9, variable=self.iou_threshold_var, orient=tk.HORIZONTAL)
        iou_scale.pack(fill=tk.X)
        self.iou_label = ttk.Label(settings_frame, text="0.5")
        self.iou_label.pack()
        
        # Bind scale to update label
        iou_scale.bind("<Motion>", self.update_iou_label)
        
        # Validation actions
        action_frame = ttk.LabelFrame(control_frame, text="Validation Actions", padding=5)
        action_frame.pack(fill=tk.X, pady=(0, 10))
          
        ttk.Button(action_frame, text="Run Validation", command=self.run_validation).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Show Comparison", command=self.show_comparison).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Generate Report", command=self.generate_report).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=2)
        
        # Results display
        results_frame = ttk.LabelFrame(control_frame, text="Validation Results", padding=5)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.results_text = tk.Text(results_frame, height=15, width=40)
        scrollbar_results = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar_results.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_results.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Ready to load data")
        self.status_label.pack(pady=5)
        
        # Image display (right side)
        image_frame = ttk.LabelFrame(main_frame, text="Image Comparison", padding=10)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create notebook for tabbed display
        self.notebook = ttk.Notebook(image_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
             
        # Ground truth tab
        gt_frame = ttk.Frame(self.notebook)
        self.notebook.add(gt_frame, text="Ground Truth")
        self.gt_canvas = tk.Canvas(gt_frame, bg="white")
        gt_scroll_x = ttk.Scrollbar(gt_frame, orient=tk.HORIZONTAL, command=self.gt_canvas.xview)
        gt_scroll_y = ttk.Scrollbar(gt_frame, orient=tk.VERTICAL, command=self.gt_canvas.yview)
        self.gt_canvas.configure(xscrollcommand=gt_scroll_x.set, yscrollcommand=gt_scroll_y.set)
        
        # Predictions tab
        pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(pred_frame, text="Predictions")
        self.pred_canvas = tk.Canvas(pred_frame, bg="white")
        pred_scroll_x = ttk.Scrollbar(pred_frame, orient=tk.HORIZONTAL, command=self.pred_canvas.xview)
        pred_scroll_y = ttk.Scrollbar(pred_frame, orient=tk.VERTICAL, command=self.pred_canvas.yview)
        self.pred_canvas.configure(xscrollcommand=pred_scroll_x.set, yscrollcommand=pred_scroll_y.set)
        
        # Comparison tab
        comp_frame = ttk.Frame(self.notebook)
        self.notebook.add(comp_frame, text="Comparison")
        self.comp_canvas = tk.Canvas(comp_frame, bg="white")
        comp_scroll_x = ttk.Scrollbar(comp_frame, orient=tk.HORIZONTAL, command=self.comp_canvas.xview)
        comp_scroll_y = ttk.Scrollbar(comp_frame, orient=tk.VERTICAL, command=self.comp_canvas.yview)
        self.comp_canvas.configure(xscrollcommand=comp_scroll_x.set, yscrollcommand=comp_scroll_y.set)
        
        # Pack canvases and scrollbars
        for canvas, scroll_x, scroll_y in [(self.gt_canvas, gt_scroll_x, gt_scroll_y),
                                          (self.pred_canvas, pred_scroll_x, pred_scroll_y),
                                          (self.comp_canvas, comp_scroll_x, comp_scroll_y)]:
            scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.enhanced_gt = EnhancedImageCanvas(gt_frame, self.gt_canvas)
        self.enhanced_pred = EnhancedImageCanvas(pred_frame, self.pred_canvas)
        self.enhanced_comp = EnhancedImageCanvas(comp_frame, self.comp_canvas)
        self.enhanced_canvases = {
            'gt': self.enhanced_gt,
            'pred': self.enhanced_pred, 
            'comp': self.enhanced_comp
        }

        image_controls = create_image_control_panel(control_frame, self.enhanced_gt)
        image_controls.pack(fill=tk.X, pady=(0, 10))
    
    def update_iou_label(self, event=None):
        """Update IoU threshold label"""
        self.iou_label.config(text=f"{self.iou_threshold_var.get():.2f}")
        self.comparison.iou_threshold = self.iou_threshold_var.get()

    def scale_bbox_for_display(self, bbox, from_original=True):
        """Scale bounding box coordinates untuk display"""
        if not hasattr(self, 'image_scale_factor') or self.image_scale_factor == 1.0:
            return bbox
        
        x1, y1, x2, y2 = bbox
        
        if from_original:
            # Scale dari original ke display size
            scale = self.image_scale_factor
        else:
            # Scale dari display ke original size
            scale = 1.0 / self.image_scale_factor
        
        return (x1 * scale, y1 * scale, x2 * scale, y2 * scale)
    
    def load_image_fixed(self):
        """Fixed image loading dengan proper annotation handling"""
        file_path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")]
        )
        
        if file_path:
            try:
                # Load dengan PIL
                pil_image = Image.open(file_path)
                pil_image = pil_image.convert("RGB")
                original_image = np.array(pil_image)
                
                if original_image is None:
                    raise ValueError("Cannot read image file")
                
                # Store original size untuk reference
                self.original_image_size = original_image.shape[:2]  # (height, width)
                
                # Check if image needs resizing
                height, width = original_image.shape[:2]
                max_dimension = 1500
                
                if max(height, width) > max_dimension:
                    # Calculate scale factor
                    scale = max_dimension / max(height, width)
                    new_height = int(height * scale)
                    new_width = int(width * scale)
                    
                    self.current_image = cv2.resize(original_image, (new_width, new_height), 
                                                interpolation=cv2.INTER_AREA)
                    self.current_image_size = (new_height, new_width)
                    self.image_scale_factor = scale
                    self.image_was_resized = True
                    
                    status_msg = f"Loaded and resized: {os.path.basename(file_path)} ({width}x{height} → {new_width}x{new_height}, scale: {scale:.2f})"
                else:
                    self.current_image = original_image
                    self.current_image_size = self.original_image_size
                    self.image_scale_factor = 1.0
                    self.image_was_resized = False
                    status_msg = f"Loaded image: {os.path.basename(file_path)} ({width}x{height})"
                
                self.update_status(status_msg)
                self.display_images_fixed()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def load_ground_truth_fixed(self):
        """Fixed: Handle different file formats for different cell types"""
        file_path = filedialog.askopenfilename(
            title="Select ground truth file",
            filetypes=[("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                print(f"=== DEBUG: Loading file {file_path} ===")
                
                # Count parasites in file first
                parasite_count = 0
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 2 and parts[1] in ['Parasite', 'Parasitized']:
                            parasite_count += 1
                
                species_choice = "falciparum"  # Default value
                
                # Only show dialog if there are parasites
                if parasite_count > 0:
                    # Dialog untuk memilih species
                    species_dialog = tk.Toplevel(self.root)
                    species_dialog.title("Select Parasite Species")
                    species_dialog.geometry("300x300")
                    species_dialog.transient(self.root)
                    species_dialog.grab_set()
                    
                    tk.Label(species_dialog, text=f"Found {parasite_count} parasite annotations").pack(pady=10)
                    tk.Label(species_dialog, text="Select the parasite species:").pack(pady=5)
                    
                    species_var = tk.StringVar(value="falciparum")
                    
                    frame = tk.Frame(species_dialog)
                    frame.pack(pady=10)
                    
                    tk.Radiobutton(frame, text="Falciparum", variable=species_var, 
                                value="falciparum").pack(anchor=tk.W)
                    tk.Radiobutton(frame, text="Vivax", variable=species_var, 
                                value="vivax").pack(anchor=tk.W)
                    
                    dialog_result = {"cancelled": False}
                    
                    def on_ok():
                        nonlocal species_choice
                        species_choice = species_var.get()
                        species_dialog.destroy()
                    
                    def on_cancel():
                        dialog_result["cancelled"] = True
                        species_dialog.destroy()
                    
                    button_frame = tk.Frame(species_dialog)
                    button_frame.pack(pady=10)
                    tk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
                    tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
                    
                    # Wait for dialog to close
                    species_dialog.wait_window()
                    
                    if dialog_result["cancelled"]:
                        return  # User cancelled
                
                # Load annotations with selected species
                self.ground_truth_annotations = []
                processed_count = 0
                
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split(',')
                        print(f"DEBUG: Processing line {line_num}: {line.strip()}")
                        
                        # Skip header line or lines with insufficient data
                        if len(parts) < 7:
                            print(f"DEBUG: Skipping line {line_num} - not enough parts ({len(parts)} < 7)")
                            continue
                        
                        # Handle different formats
                        if len(parts) >= 9:
                            # Format for parasites: _, label, *_, x1, y1, x2, y2
                            _, label, *middle_parts, x1, y1, x2, y2 = parts
                            try:
                                x1, y1 = map(float, [x1, y1])
                            except ValueError as e:
                                print(f"DEBUG: Error converting coordinates: {e}")
                                continue
                        elif len(parts) == 7:
                            # Format for uninfected: id, label, comment, type, count, x1, y1
                            _, label, _, _, _, x1, y1 = parts
                            try:
                                x1, y1 = map(float, [x1, y1])
                            except ValueError as e:
                                print(f"DEBUG: Error converting coordinates: {e}")
                                continue
                        else:
                            print(f"DEBUG: Unknown format for line {line_num}")
                            continue
                        
                        print(f"DEBUG: Label: '{label}', x1: {x1}, y1: {y1}")
                        
                        # Convert center point to bounding box
                        box_size = 128
                        half = box_size / 2
                        x1_box = x1 - half
                        y1_box = y1 - half
                        x2_box = x1_box + box_size
                        y2_box = y1_box + box_size
                        
                        # Map label to class
                        if label == 'Parasite':
                            if species_choice == 'falciparum':
                                class_idx = 0
                                class_name = 'falciparum'
                            else:  # vivax
                                class_idx = 1
                                class_name = 'vivax'
                        elif label == 'Parasitized':
                            class_idx = 1  # vivax
                            class_name = 'vivax'
                        elif label == 'White_Blood_Cell':
                            class_idx = 2  # uninfected
                            class_name = 'uninfected'
                        else:
                            # For any other labels, treat as uninfected
                            class_idx = 2  # uninfected
                            class_name = 'uninfected'
                        
                        print(f"DEBUG: Mapped to class_idx: {class_idx}, class_name: {class_name}")
                        
                        annotation = {
                            'bbox': (x1_box, y1_box, x2_box, y2_box),
                            'class': class_idx,
                            'class_name': class_name,
                            'label': label
                        }
                        
                        self.ground_truth_annotations.append(annotation)
                        processed_count += 1
                        print(f"DEBUG: Added annotation {processed_count}")
                
                print(f"DEBUG: Final annotations count: {len(self.ground_truth_annotations)}")
                
                self.update_status(f"Loaded {len(self.ground_truth_annotations)} ground truth annotations")
                self.display_images_fixed()
                
            except Exception as e:
                print(f"DEBUG: Exception occurred: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", f"Failed to load ground truth: {str(e)}")
    
    def load_predictions_fixed(self):
        """Fixed predictions loading dengan proper coordinate handling"""
        file_path = filedialog.askopenfilename(
            title="Select predictions file",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                self.predicted_annotations = []
                
                if 'annotations' in data:
                    for ann in data['annotations']:
                        x, y = ann['x'], ann['y']
                        size = ann.get('size', 128)
                        predicted_class = ann.get('predicted_class')
                        confidence = ann.get('confidence', 0)
                        
                        if predicted_class is not None:
                            # Convert center point to bounding box - ORIGINAL coordinates
                            half = size / 2
                            x1, y1 = x - half, y - half
                            x2, y2 = x1 + size, y1 + size
                            
                            self.predicted_annotations.append({
                                'bbox': (x1, y1, x2, y2),  # Original coordinates
                                'class': predicted_class,
                                'class_name': self.class_names[predicted_class],
                                'confidence': confidence
                            })
                
                self.update_status(f"Loaded {len(self.predicted_annotations)} predictions")
                self.display_images_fixed()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load predictions: {str(e)}")

    def load_enhanced_batch_folder(self):
        """Load enhanced batch folder - method untuk ditambah ke MalariaValidationGUI"""
        if not hasattr(self, 'enhanced_loader'):
            self.enhanced_loader = EnhancedFolderBatchLoader(self)
        
        dialog = self.enhanced_loader.show_enhanced_batch_dialog()
          
    def display_images_fixed(self):
        """Fixed image display dengan proper annotation handling"""
        if self.current_image is None:
            return
        
        try:
            # Display ground truth
            if self.ground_truth_annotations:
                gt_image = self.draw_annotations_fixed(self.current_image, 
                                                    self.ground_truth_annotations, "ground_truth")
                self.enhanced_gt.set_image(gt_image)
            else:
                self.enhanced_gt.set_image(self.current_image)
            
            # Display predictions
            if self.predicted_annotations:
                pred_image = self.draw_annotations_fixed(self.current_image,
                                                    self.predicted_annotations, "predictions")
                self.enhanced_pred.set_image(pred_image)
            else:
                self.enhanced_pred.set_image(self.current_image)
                
        except Exception as e:
            print(f"Error displaying images: {e}")
            messagebox.showwarning("Display Warning", f"Error displaying annotations: {str(e)}")
    
    def draw_annotations_fixed(self, image, annotations, annotation_type):
        """Fixed annotation drawing dengan proper scaling"""
        display_image = image.copy()
        
        for ann in annotations:
            # Scale bbox untuk display
            bbox = self.scale_bbox_for_display(ann['bbox'], from_original=True)
            x1, y1, x2, y2 = bbox
            
            # Pastikan coordinates dalam bounds
            height, width = display_image.shape[:2]
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            # Skip jika bbox terlalu kecil setelah scaling
            if abs(x2-x1) < 2 or abs(y2-y1) < 2:
                continue
                
            class_name = ann['class_name']
            
            # Get color for the class
            if annotation_type == "ground_truth" and 'label' in ann:
                color_name = self.class_colors.get(ann['label'], 'gray')
            else:
                color_name = self.class_colors.get(class_name, 'gray')
            
            # Convert color name to RGB
            color_map = {
                'red': (255, 0, 0), 
                'lime': (0, 255, 0), 
                'green': (0, 255, 0),
                'blue': (0, 0, 255), 
                'gray': (128, 128, 128)
            }
            color = color_map.get(color_name, (128, 128, 128))
            
            # Draw rectangle dengan line thickness yang disesuaikan
            thickness = max(1, int(2 * self.image_scale_factor))
            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw label dengan font size yang disesuaikan
            label_text = class_name
            if 'confidence' in ann:
                label_text += f" ({ann['confidence']:.2f})"
            
            font_scale = max(0.3, 0.5 * self.image_scale_factor)
            cv2.putText(display_image, label_text, (int(x1), int(y1-5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return display_image
    
    def display_on_canvas(self, canvas, image):
           if canvas == self.gt_canvas:
               self.enhanced_gt.set_image(image)
           elif canvas == self.pred_canvas:
               self.enhanced_pred.set_image(image)
           elif canvas == self.comp_canvas:
               self.enhanced_comp.set_image(image)
    
    def run_validation(self):
        """Run validation with better debugging - FIXED VERSION"""
        if not self.ground_truth_annotations or not self.predicted_annotations:
            messagebox.showwarning("Warning", "Please load both ground truth and predictions first")
            return
        
        try:
            print(f"\nDEBUG: Starting validation...")
            print(f"Ground truth annotations: {len(self.ground_truth_annotations)}")
            print(f"Predicted annotations: {len(self.predicted_annotations)}")
            print(f"IoU threshold: {self.comparison.iou_threshold}")
            
            # Reset validation results
            self.validation_results = ValidationResults()
            
            # Match annotations
            matches, unmatched_gt, unmatched_pred = self.comparison.match_annotations(
                self.ground_truth_annotations, self.predicted_annotations
            )
            
            print(f"Matching results:")
            print(f"  Matches found: {len(matches)}")
            print(f"  Unmatched GT: {len(unmatched_gt)}")
            print(f"  Unmatched predictions: {len(unmatched_pred)}")
            
            # Process matches
            for i, match in enumerate(matches):
                gt_class = match['gt_class']
                pred_class = match['pred_class']
                
                print(f"Match {i+1}: GT class {gt_class} -> Pred class {pred_class} (IoU: {match['iou']:.3f})")
                
                self.validation_results.add_result(pred_class, gt_class, matched=True)
            
            # Store results for detailed analysis
            self.validation_results.matched_pairs = matches
            self.validation_results.unmatched_predictions = [self.predicted_annotations[i] for i in unmatched_pred]
            self.validation_results.unmatched_ground_truth = [self.ground_truth_annotations[i] for i in unmatched_gt]
            
            # Calculate metrics
            metrics = self.validation_results.calculate_metrics()
            print(f"\nCalculated metrics: {metrics}")
            
            # Display results
            self.display_validation_results(metrics, matches, unmatched_gt, unmatched_pred)
            self.update_status("Validation completed")
            
            # Debug validation results
            self.debug_validation_results()
            
        except Exception as e:
            print(f"Validation error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Validation failed: {str(e)}")
       
    def display_validation_results(self, metrics, matches, unmatched_gt, unmatched_pred):
        """Display validation results in the text widget"""
        self.results_text.delete(1.0, tk.END)
        
        # Overall results
        overall = metrics['overall']
        self.results_text.insert(tk.END, "=== VALIDATION RESULTS ===\n\n")
        self.results_text.insert(tk.END, f"Overall Accuracy: {overall['accuracy']:.1%}\n")
        self.results_text.insert(tk.END, f"Total Samples: {overall['total_samples']}\n")
        self.results_text.insert(tk.END, f"Correct Predictions: {overall['total_correct']}\n\n")
        
        # Per-class metrics
        self.results_text.insert(tk.END, "=== PER-CLASS METRICS ===\n")
        for class_name in self.class_names:
            if class_name in metrics:
                m = metrics[class_name]
                self.results_text.insert(tk.END, f"\n{class_name.upper()}:\n")
                self.results_text.insert(tk.END, f"  Precision: {m['precision']:.1%}\n")
                self.results_text.insert(tk.END, f"  Recall: {m['recall']:.1%}\n")
                self.results_text.insert(tk.END, f"  F1-Score: {m['f1_score']:.1%}\n")
                self.results_text.insert(tk.END, f"  True Positives: {m['true_positives']}\n")
                self.results_text.insert(tk.END, f"  False Positives: {m['false_positives']}\n")
                self.results_text.insert(tk.END, f"  False Negatives: {m['false_negatives']}\n")
        
        # Matching statistics
        self.results_text.insert(tk.END, f"\n=== MATCHING STATISTICS ===\n")
        self.results_text.insert(tk.END, f"Matched Pairs: {len(matches)}\n")
        self.results_text.insert(tk.END, f"Unmatched Ground Truth: {len(unmatched_gt)}\n")
        self.results_text.insert(tk.END, f"Unmatched Predictions: {len(unmatched_pred)}\n")
        self.results_text.insert(tk.END, f"IoU Threshold: {self.comparison.iou_threshold:.2f}\n")
        
        # Detailed matches
        if matches:
            self.results_text.insert(tk.END, f"\n=== DETAILED MATCHES ===\n")
            for i, match in enumerate(matches):
                gt_class = self.class_names[match['gt_class']]
                pred_class = self.class_names[match['pred_class']]
                iou = match['iou']
                correct = "✓" if match['gt_class'] == match['pred_class'] else "✗"
                self.results_text.insert(tk.END, f"{i+1:2d}. {correct} GT:{gt_class} → Pred:{pred_class} (IoU:{iou:.3f})\n")
    
    def show_comparison(self):
        """Show side-by-side comparison with matches highlighted"""
        if not hasattr(self.validation_results, 'matched_pairs'):
            messagebox.showwarning("Warning", "Please run validation first")
            return
        
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        # Create comparison image
        comp_image = self.create_comparison_image_fixed()
        self.display_on_canvas(self.comp_canvas, comp_image)
        
        # Switch to comparison tab
        self.notebook.select(2)
    
    def create_comparison_image_fixed(self):
        """Fixed comparison image creation"""
        if self.current_image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        height, width = self.current_image.shape[:2]
        
        # Create side-by-side comparison
        try:
            comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
            comparison[:, :width] = self.current_image
            comparison[:, width:] = self.current_image
            
            # Draw ground truth on left side
            for ann in self.ground_truth_annotations:
                bbox = self.scale_bbox_for_display(ann['bbox'], from_original=True)
                x1, y1, x2, y2 = bbox
                
                # Clamp coordinates
                x1 = max(0, min(int(x1), width-1))
                y1 = max(0, min(int(y1), height-1))
                x2 = max(0, min(int(x2), width-1))
                y2 = max(0, min(int(y2), height-1))
                
                if x2 > x1 and y2 > y1:  # Valid bbox
                    color = (255, 0, 0)  # Green for GT
                    thickness = max(1, int(2 * self.image_scale_factor))
                    cv2.rectangle(comparison, (x1, y1), (x2, y2), color, thickness)
                    
                    font_scale = max(0.3, 0.5 * self.image_scale_factor)
                    cv2.putText(comparison, f"GT: {ann['class_name']}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            # Draw predictions on right side
            for ann in self.predicted_annotations:
                bbox = self.scale_bbox_for_display(ann['bbox'], from_original=True)
                x1, y1, x2, y2 = bbox
                x1_right, x2_right = x1 + width, x2 + width
                
                # Clamp coordinates
                x1_right = max(width, min(int(x1_right), width*2-1))
                y1 = max(0, min(int(y1), height-1))
                x2_right = max(width, min(int(x2_right), width*2-1))
                y2 = max(0, min(int(y2), height-1))
                
                if x2_right > x1_right and y2 > y1:  # Valid bbox
                    color = (0, 100, 155)  # Blue for predictions
                    thickness = max(1, int(2 * self.image_scale_factor))
                    cv2.rectangle(comparison, (x1_right, y1), (x2_right, y2), color, thickness)
                    
                    label = f"Pred: {ann['class_name']}"
                    if 'confidence' in ann:
                        label += f" ({ann['confidence']:.2f})"
                    
                    font_scale = max(0.3, 0.5 * self.image_scale_factor)
                    cv2.putText(comparison, label, (x1_right, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            # Add section labels
            font_scale = max(0.5, 1.0 * self.image_scale_factor)
            thickness = max(1, int(2 * self.image_scale_factor))
            cv2.putText(comparison, "GROUND TRUTH", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            cv2.putText(comparison, "PREDICTIONS", (width + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            return comparison
            
        except MemoryError:
            # Fallback untuk memory issues
            return self.create_small_comparison_fallback()
    
    def create_small_comparison_fallback(self):
        """Fallback comparison untuk memory issues"""
        # Create smaller version
        scale = 0.5
        small_height = int(self.current_image.shape[0] * scale)
        small_width = int(self.current_image.shape[1] * scale)
        
        small_image = cv2.resize(self.current_image, (small_width, small_height))
        comparison = np.zeros((small_height, small_width * 2, 3), dtype=np.uint8)
        comparison[:, :small_width] = small_image
        comparison[:, small_width:] = small_image
        
        # Draw simplified annotations
        for ann in self.ground_truth_annotations[:20]:  # Limit annotations
            bbox = self.scale_bbox_for_display(ann['bbox'], from_original=True)
            x1, y1, x2, y2 = [coord * scale for coord in bbox]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            if 0 <= x1 < small_width and 0 <= y1 < small_height and x2 > x1 and y2 > y1:
                cv2.rectangle(comparison, (x1, y1), (min(x2, small_width), min(y2, small_height)), 
                            (0, 255, 0), 1)
        
        cv2.putText(comparison, "GROUND TRUTH", (5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(comparison, "PREDICTIONS", (small_width + 5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return comparison

    def generate_report(self):
        """Generate detailed validation report with better error handling - FIXED VERSION"""
        if not hasattr(self.validation_results, 'matched_pairs'):
            messagebox.showwarning("Warning", "Please run validation first")
            return
        
        try:
            # Debug before generating report
            self.debug_validation_results()
            
            # Create report directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = f"validation_report_{timestamp}"
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate metrics
            metrics = self.validation_results.calculate_metrics()
            print(f"Generated metrics for report: {metrics}")
            
            # Create confusion matrix with better error handling
            self.create_confusion_matrix(report_dir, metrics)
            
            # Create performance plots
            self.create_performance_plots(report_dir, metrics)
            
            # Save comparison image
            if self.current_image is not None:
                comp_image = self.create_comparison_image_fixed()
                cv2.imwrite(os.path.join(report_dir, "comparison.png"), 
                        cv2.cvtColor(comp_image, cv2.COLOR_RGB2BGR))
            
            # Generate text report
            self.create_text_report(report_dir, metrics)
            
            messagebox.showinfo("Success", f"Report generated in folder: {report_dir}")
            self.update_status(f"Report saved to {report_dir}")
            
        except Exception as e:
            print(f"Report generation error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def create_confusion_matrix(self, report_dir, metrics):
        """Create and save confusion matrix visualization - FIXED VERSION"""
        try:                       
            # Create confusion matrix data
            classes = self.class_names
            matrix = np.zeros((len(classes), len(classes)), dtype=int)
            
            print(f"\nDEBUG: Creating confusion matrix...")
            print(f"Classes: {classes}")
            
            # Check if we have matched pairs data
            if hasattr(self.validation_results, 'matched_pairs') and self.validation_results.matched_pairs:
                print(f"Found {len(self.validation_results.matched_pairs)} matched pairs")
                
                # Fill matrix from matched pairs
                for i, match in enumerate(self.validation_results.matched_pairs):
                    gt_idx = match['gt_class']
                    pred_idx = match['pred_class']
                    
                    print(f"Match {i+1}: GT class {gt_idx} ({classes[gt_idx]}) -> Pred class {pred_idx} ({classes[pred_idx]})")
                    
                    # Validate indices
                    if 0 <= gt_idx < len(classes) and 0 <= pred_idx < len(classes):
                        matrix[gt_idx][pred_idx] += 1
                    else:
                        print(f"WARNING: Invalid class indices - GT: {gt_idx}, Pred: {pred_idx}")
                
            else:
                # Alternative: Build matrix from metrics data
                print("No matched_pairs found, trying to build from metrics...")
                
                if metrics and isinstance(metrics, dict):
                    for class_name in classes:
                        if class_name in metrics:
                            m = metrics[class_name]
                            class_idx = classes.index(class_name)
                            
                            # Add true positives to diagonal
                            tp = m.get('true_positives', 0)
                            matrix[class_idx][class_idx] += tp
                            print(f"{class_name}: Added {tp} true positives")
                            
                            # For false positives and negatives, we need to estimate distribution
                            # This is not perfect but better than all zeros
                            fp = m.get('false_positives', 0)
                            fn = m.get('false_negatives', 0)
                            
                            if fp > 0:
                                # Distribute false positives among other classes
                                other_classes = len(classes) - 1
                                if other_classes > 0:
                                    fp_per_class = fp // other_classes
                                    fp_remainder = fp % other_classes
                                    
                                    for other_idx in range(len(classes)):
                                        if other_idx != class_idx:
                                            matrix[other_idx][class_idx] += fp_per_class
                                            if fp_remainder > 0:
                                                matrix[other_idx][class_idx] += 1
                                                fp_remainder -= 1
                            
                            if fn > 0:
                                # Distribute false negatives among other classes
                                other_classes = len(classes) - 1
                                if other_classes > 0:
                                    fn_per_class = fn // other_classes
                                    fn_remainder = fn % other_classes
                                    
                                    for other_idx in range(len(classes)):
                                        if other_idx != class_idx:
                                            matrix[class_idx][other_idx] += fn_per_class
                                            if fn_remainder > 0:
                                                matrix[class_idx][other_idx] += 1
                                                fn_remainder -= 1
            
            print(f"\nFinal confusion matrix:")
            print(matrix)
            print(f"Matrix sum: {np.sum(matrix)}")
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            
            # Check if matrix has any data
            if np.sum(matrix) == 0:
                # Create a placeholder matrix with at least some data for visualization
                print("WARNING: Matrix is all zeros, creating placeholder...")
                
                # Add minimal data for visualization purposes
                for i in range(len(classes)):
                    matrix[i][i] = 1  # Add at least 1 to diagonal
                
                plt.text(0.5, 0.95, 'Warning: No matching data found - showing placeholder', 
                        transform=plt.gca().transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            # Create heatmap
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=classes, yticklabels=classes,
                        cbar_kws={'label': 'Number of Samples'},
                        annot_kws={"size": 12})

            
            plt.title('Confusion Matrix', fontsize=12)
            plt.xlabel('Predicted Class', fontsize=12)
            plt.ylabel('Actual Class', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            
            # Add accuracy information if available
            if metrics and 'overall' in metrics:
                accuracy = metrics['overall'].get('accuracy', 0)
                plt.text(0.02, 0.98, f'Overall Accuracy: {accuracy:.1%}', 
                        transform=plt.gca().transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Confusion matrix saved successfully")
            
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a simple fallback matrix
            try:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f'Error creating confusion matrix:\n{str(e)}', 
                        ha='center', va='center', transform=plt.gca().transAxes,
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
                plt.title('Confusion Matrix - Error')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(report_dir, "confusion_matrix_error.png"), dpi=300, bbox_inches='tight')
                plt.close()
            except:
                pass

    def debug_validation_results(self):
        """Debug function to check validation results structure"""
        print("\nDEBUG: Validation Results Analysis")
        print("=" * 50)
        
        # Check validation_results structure
        if hasattr(self, 'validation_results'):
            print(f"validation_results type: {type(self.validation_results)}")
            
            # Check for matched_pairs
            if hasattr(self.validation_results, 'matched_pairs'):
                pairs = self.validation_results.matched_pairs
                print(f"matched_pairs length: {len(pairs) if pairs else 0}")
                
                if pairs and len(pairs) > 0:
                    print("First few matches:")
                    for i, match in enumerate(pairs[:3]):
                        print(f"  Match {i+1}: {match}")
                else:
                    print("No matched pairs found!")
            
            # Check metrics data
            try:
                metrics = self.validation_results.calculate_metrics()
                print(f"Calculated metrics keys: {list(metrics.keys()) if metrics else 'None'}")
                
                if metrics and 'overall' in metrics:
                    overall = metrics['overall']
                    print(f"Overall accuracy: {overall.get('accuracy', 'N/A')}")
                    print(f"Total samples: {overall.get('total_samples', 'N/A')}")
                    print(f"Total correct: {overall.get('total_correct', 'N/A')}")
                
                # Check per-class metrics
                for class_name in self.class_names:
                    if metrics and class_name in metrics:
                        m = metrics[class_name]
                        print(f"{class_name}: TP={m.get('true_positives', 0)}, FP={m.get('false_positives', 0)}, FN={m.get('false_negatives', 0)}")
                
            except Exception as e:
                print(f"Error calculating metrics: {e}")
        
        else:
            print("No validation_results found!")
        
        print("=" * 50)
    
    def create_performance_plots(self, report_dir, metrics):
        """Create performance visualization plots"""
        classes = self.class_names
        
        # Extract metrics for plotting
        precision_scores = [metrics[cls]['precision'] for cls in classes if cls in metrics]
        recall_scores = [metrics[cls]['recall'] for cls in classes if cls in metrics]
        f1_scores = [metrics[cls]['f1_score'] for cls in classes if cls in metrics]
        
        # Create bar plot
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(10, 8))
        plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        plt.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Classification Performance Metrics')
        plt.xticks(x, classes)
        plt.legend()
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
            plt.text(i - width, p + 0.01, f'{p:.2f}', ha='center', va='bottom')
            plt.text(i, r + 0.01, f'{r:.2f}', ha='center', va='bottom')
            plt.text(i + width, f + 0.01, f'{f:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "performance_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create IoU distribution plot
        if self.validation_results.matched_pairs:
            iou_values = [match['iou'] for match in self.validation_results.matched_pairs]
            
            plt.figure(figsize=(8, 6))
            plt.hist(iou_values, bins=20, edgecolor='black', alpha=0.7)
            plt.axvline(self.comparison.iou_threshold, color='red', linestyle='--', 
                       label=f'Threshold: {self.comparison.iou_threshold:.2f}')
            plt.xlabel('IoU Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of IoU Scores for Matched Pairs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "iou_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_text_report(self, report_dir, metrics):
        """Create detailed text report"""
        report_path = os.path.join(report_dir, "validation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("MALARIA CLASSIFICATION VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall metrics
            overall = metrics['overall']
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"Overall Accuracy: {overall['accuracy']:.1%}\n")
            f.write(f"Total Samples: {overall['total_samples']}\n")
            f.write(f"Correct Predictions: {overall['total_correct']}\n\n")
            
            # Per-class metrics
            f.write("PER-CLASS PERFORMANCE:\n")
            for class_name in self.class_names:
                if class_name in metrics:
                    m = metrics[class_name]
                    f.write(f"\n{class_name.upper()}:\n")
                    f.write(f"  Precision: {m['precision']:.1%}\n")
                    f.write(f"  Recall: {m['recall']:.1%}\n")
                    f.write(f"  F1-Score: {m['f1_score']:.1%}\n")
                    f.write(f"  True Positives: {m['true_positives']}\n")
                    f.write(f"  False Positives: {m['false_positives']}\n")
                    f.write(f"  False Negatives: {m['false_negatives']}\n")
            
            # Matching statistics
            f.write(f"\nMATCHING STATISTICS:\n")
            f.write(f"IoU Threshold: {self.comparison.iou_threshold:.2f}\n")
            f.write(f"Matched Pairs: {len(self.validation_results.matched_pairs)}\n")
            f.write(f"Unmatched Ground Truth: {len(self.validation_results.unmatched_ground_truth)}\n")
            f.write(f"Unmatched Predictions: {len(self.validation_results.unmatched_predictions)}\n\n")
            
            # Detailed match analysis
            if self.validation_results.matched_pairs:
                f.write("DETAILED MATCH ANALYSIS:\n")
                correct_matches = 0
                total_matches = len(self.validation_results.matched_pairs)
                
                for i, match in enumerate(self.validation_results.matched_pairs):
                    gt_class = self.class_names[match['gt_class']]
                    pred_class = self.class_names[match['pred_class']]
                    iou = match['iou']
                    is_correct = match['gt_class'] == match['pred_class']
                    
                    if is_correct:
                        correct_matches += 1
                    
                    status = "CORRECT" if is_correct else "INCORRECT"
                    f.write(f"{i+1:3d}. {status} - GT: {gt_class} | Pred: {pred_class} | IoU: {iou:.3f}\n")
                
                f.write(f"\nMatch Accuracy: {correct_matches}/{total_matches} ({correct_matches/total_matches:.1%})\n")
            
            # Recommendations
            f.write(f"\nRECOMMENDATIONS:\n")
            if overall['accuracy'] < 0.8:
                f.write("- Overall accuracy is below 80%. Consider:\n")
                f.write("  • Improving training data quality\n")
                f.write("  • Increasing training data quantity\n")
                f.write("  • Adjusting model hyperparameters\n")
            
            if len(self.validation_results.unmatched_predictions) > 0:
                f.write(f"- {len(self.validation_results.unmatched_predictions)} false positive predictions detected\n")
                f.write("  • Consider adjusting confidence threshold\n")
                f.write("  • Review prediction quality\n")
            
            if len(self.validation_results.unmatched_ground_truth) > 0:
                f.write(f"- {len(self.validation_results.unmatched_ground_truth)} missed ground truth annotations\n")
                f.write("  • Consider lowering IoU threshold for matching\n")
                f.write("  • Review model sensitivity\n")
    
    def export_results(self):
        """Export results to CSV format"""
        if not hasattr(self.validation_results, 'matched_pairs'):
            messagebox.showwarning("Warning", "Please run validation first")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export validation results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                # Prepare data for export
                export_data = []
                
                # Add matched pairs
                for i, match in enumerate(self.validation_results.matched_pairs):
                    export_data.append({
                        'Type': 'Match',
                        'Index': i + 1,
                        'Ground_Truth_Class': self.class_names[match['gt_class']],
                        'Predicted_Class': self.class_names[match['pred_class']],
                        'IoU': match['iou'],
                        'Correct': match['gt_class'] == match['pred_class'],
                        'GT_Bbox': str(match['gt_bbox']),
                        'Pred_Bbox': str(match['pred_bbox'])
                    })
                
                # Add unmatched ground truth
                for i, gt in enumerate(self.validation_results.unmatched_ground_truth):
                    export_data.append({
                        'Type': 'Unmatched_GT',
                        'Index': i + 1,
                        'Ground_Truth_Class': gt['class_name'],
                        'Predicted_Class': '',
                        'IoU': 0,
                        'Correct': False,
                        'GT_Bbox': str(gt['bbox']),
                        'Pred_Bbox': ''
                    })
                
                # Add unmatched predictions
                for i, pred in enumerate(self.validation_results.unmatched_predictions):
                    export_data.append({
                        'Type': 'Unmatched_Pred',
                        'Index': i + 1,
                        'Ground_Truth_Class': '',
                        'Predicted_Class': pred['class_name'],
                        'IoU': 0,
                        'Correct': False,
                        'GT_Bbox': '',
                        'Pred_Bbox': str(pred['bbox'])
                    })
                
                # Create DataFrame and save
                df = pd.DataFrame(export_data)
                df.to_csv(file_path, index=False)
                
                messagebox.showinfo("Success", f"Results exported to {os.path.basename(file_path)}")
                self.update_status(f"Results exported")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

def main():
    """Main function to run the validation application"""
    root = tk.Tk()
    app = MalariaValidationGUI(root)
    
    # Handle window closing
    def on_closing():
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    print("=== Malaria Classification Validation System ===")
    print("Biomedical Engineering Lab")
    print("-" * 50)
    print("FEATURES:")
    print("✓ Load ground truth annotations from TXT files")
    print("✓ Load predictions from JSON files") 
    print("✓ Visual comparison of annotations")
    print("✓ IoU-based matching algorithm")
    print("✓ Comprehensive accuracy metrics")
    print("✓ Confusion matrix generation")
    print("✓ Performance visualization")
    print("✓ Detailed validation reports")
    print("✓ CSV export functionality")
    print("-" * 50)
    print("INSTRUCTIONS:")
    print("1. Load an image file")
    print("2. Load ground truth annotations (TXT format from bbox check.py)")
    print("3. Load prediction annotations (JSON format from detection program v2.py)")
    print("4. Adjust IoU threshold if needed (default: 0.5)")
    print("5. Run validation to compare annotations")
    print("6. View results in different tabs:")
    print("   • Ground Truth: Shows original annotations")
    print("   • Predictions: Shows model predictions")  
    print("   • Comparison: Side-by-side with matches highlighted")
    print("7. Generate detailed report with visualizations")
    print("8. Export results to CSV for further analysis")
    print("-" * 50)
    print("METRICS CALCULATED:")
    print("• Overall Accuracy")
    print("• Per-class Precision, Recall, F1-Score")
    print("• True/False Positives and Negatives")
    print("• IoU distribution for matches")
    print("• Confusion matrix")
    print("-" * 50)
    
    # Check dependencies
    missing_deps = []
    required_modules = ['cv2', 'matplotlib', 'seaborn', 'pandas', 'numpy', 'PIL']
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(module)
    
    if missing_deps:
        print(f"ERROR: Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install opencv-python matplotlib seaborn pandas numpy Pillow")
        sys.exit(1)
    
    print("All dependencies found. Starting application...")
    main()