#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Batch Loader untuk Model Validation
Solusi untuk load folder tanpa suffix konsisten
"""

import os
import json
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox

class SmartBatchLoader:
    """Class untuk handle batch loading tanpa suffix konsisten"""
    
    def __init__(self, validation_gui):
        self.validation_gui = validation_gui
        self.image_extensions = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp']
    
    def show_smart_batch_dialog(self, folder_path):
        """Dialog untuk smart batch processing"""
        dialog = tk.Toplevel(self.validation_gui.root)
        dialog.title("Smart Batch Processing")
        dialog.geometry("700x800")
        dialog.transient(self.validation_gui.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Folder info
        ttk.Label(main_frame, text=f"Analyzing folder: {os.path.basename(folder_path)}", 
                 font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Scan files and show structure
        file_structure = self.analyze_folder_structure(folder_path)
        
        # Display file analysis
        analysis_frame = ttk.LabelFrame(main_frame, text="File Analysis", padding=5)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        analysis_text = tk.Text(analysis_frame, height=8, width=80)
        analysis_scroll = ttk.Scrollbar(analysis_frame, orient=tk.VERTICAL, command=analysis_text.yview)
        analysis_text.configure(yscrollcommand=analysis_scroll.set)
        
        # Show analysis results
        analysis_content = self.generate_analysis_report(file_structure)
        analysis_text.insert(tk.END, analysis_content)
        analysis_text.config(state=tk.DISABLED)
        
        analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        analysis_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Matching options
        matching_frame = ttk.LabelFrame(main_frame, text="File Matching Strategy", padding=5)
        matching_frame.pack(fill=tk.X, pady=5)
        
        self.matching_strategy = tk.StringVar(value="exact_name")
        
        ttk.Radiobutton(matching_frame, text="Exact name matching (image.jpg ↔ image.txt ↔ image.json)", 
                       variable=self.matching_strategy, value="exact_name").pack(anchor=tk.W)
        ttk.Radiobutton(matching_frame, text="Flexible matching (find closest names)", 
                       variable=self.matching_strategy, value="flexible").pack(anchor=tk.W)
        ttk.Radiobutton(matching_frame, text="Manual pattern specification", 
                       variable=self.matching_strategy, value="manual").pack(anchor=tk.W)
        
        # Manual pattern frame (initially hidden)
        self.manual_frame = ttk.Frame(matching_frame)
        
        ttk.Label(self.manual_frame, text="Ground Truth pattern (e.g., *_gt.txt, *_annotations.txt):").pack(anchor=tk.W)
        self.gt_pattern_var = tk.StringVar(value="*.txt")
        ttk.Entry(self.manual_frame, textvariable=self.gt_pattern_var, width=50).pack(fill=tk.X, pady=2)
        
        ttk.Label(self.manual_frame, text="Predictions pattern (e.g., *_pred.json, *_results.json):").pack(anchor=tk.W)
        self.pred_pattern_var = tk.StringVar(value="*.json")
        ttk.Entry(self.manual_frame, textvariable=self.pred_pattern_var, width=50).pack(fill=tk.X, pady=2)
        
        # Bind radio button to show/hide manual frame
        def on_strategy_change():
            if self.matching_strategy.get() == "manual":
                self.manual_frame.pack(fill=tk.X, pady=5)
            else:
                self.manual_frame.pack_forget()
        
        for widget in matching_frame.winfo_children():
            if isinstance(widget, ttk.Radiobutton):
                widget.configure(command=on_strategy_change)
        
        # Preview matching results
        preview_frame = ttk.LabelFrame(main_frame, text="Matching Preview", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create treeview for preview
        columns = ('Image', 'Ground Truth', 'Predictions', 'Status')
        self.preview_tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=150)
        
        preview_scroll_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=preview_scroll_y.set)
        
        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Update Preview", 
                  command=lambda: self.update_matching_preview(folder_path, file_structure)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Start Batch Processing", 
                  command=lambda: self.start_smart_batch_processing(folder_path, dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Initial preview
        self.update_matching_preview(folder_path, file_structure)
    
    def analyze_folder_structure(self, folder_path):
        """Analyze folder structure untuk identify files"""
        structure = {
            'images': [],
            'txt_files': [],
            'json_files': [],
            'other_files': [],
            'subdirs': []
        }
        
        for root, dirs, files in os.walk(folder_path):
            structure['subdirs'].extend([os.path.join(root, d) for d in dirs])
            
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                
                if any(file_lower.endswith(f'.{ext}') for ext in self.image_extensions):
                    structure['images'].append(file_path)
                elif file_lower.endswith('.txt'):
                    structure['txt_files'].append(file_path)
                elif file_lower.endswith('.json'):
                    structure['json_files'].append(file_path)
                else:
                    structure['other_files'].append(file_path)
        
        return structure
    
    def generate_analysis_report(self, structure):
        """Generate text report dari folder analysis"""
        report = "FOLDER STRUCTURE ANALYSIS:\n"
        report += "=" * 40 + "\n\n"
        
        report += f"Image files found: {len(structure['images'])}\n"
        report += f"Text files found: {len(structure['txt_files'])}\n"
        report += f"JSON files found: {len(structure['json_files'])}\n"
        report += f"Other files: {len(structure['other_files'])}\n"
        report += f"Subdirectories: {len(structure['subdirs'])}\n\n"
        
        # Show sample filenames to identify patterns
        report += "SAMPLE FILENAMES:\n"
        report += "-" * 20 + "\n"
        
        if structure['images']:
            report += "Images (first 5):\n"
            for img in structure['images'][:5]:
                report += f"  • {os.path.basename(img)}\n"
            report += "\n"
        
        if structure['txt_files']:
            report += "Text files (first 5):\n"
            for txt in structure['txt_files'][:5]:
                report += f"  • {os.path.basename(txt)}\n"
            report += "\n"
        
        if structure['json_files']:
            report += "JSON files (first 5):\n"
            for json_file in structure['json_files'][:5]:
                report += f"  • {os.path.basename(json_file)}\n"
            report += "\n"
        
        # Try to identify common patterns
        patterns = self.identify_naming_patterns(structure)
        if patterns:
            report += "DETECTED PATTERNS:\n"
            report += "-" * 20 + "\n"
            for pattern in patterns:
                report += f"  • {pattern}\n"
        
        return report
    
    def identify_naming_patterns(self, structure):
        """Try to identify common naming patterns"""
        patterns = []
        
        # Get base names without extensions
        img_bases = set()
        txt_bases = set()
        json_bases = set()
        
        for img in structure['images']:
            base = os.path.splitext(os.path.basename(img))[0]
            img_bases.add(base)
        
        for txt in structure['txt_files']:
            base = os.path.splitext(os.path.basename(txt))[0]
            txt_bases.add(base)
        
        for json_file in structure['json_files']:
            base = os.path.splitext(os.path.basename(json_file))[0]
            json_bases.add(base)
        
        # Check for exact matches
        exact_matches = img_bases.intersection(txt_bases).intersection(json_bases)
        if exact_matches:
            patterns.append(f"Exact name matching possible ({len(exact_matches)} files)")
        
        # Check for common prefixes/suffixes
        common_prefixes = set()
        common_suffixes = set()
        
        for img_base in img_bases:
            for txt_base in txt_bases:
                if img_base in txt_base:
                    suffix = txt_base.replace(img_base, '')
                    if suffix:
                        common_suffixes.add(f"TXT suffix: '{suffix}'")
                
                if txt_base in img_base:
                    prefix = img_base.replace(txt_base, '')
                    if prefix:
                        common_prefixes.add(f"IMG prefix: '{prefix}'")
        
        patterns.extend(list(common_prefixes)[:3])  # Limit to 3
        patterns.extend(list(common_suffixes)[:3])
        
        return patterns
    
    def update_matching_preview(self, folder_path, structure):
        """Update preview tree dengan matching results"""
        # Clear existing items
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        
        strategy = self.matching_strategy.get()
        matches = self.find_file_matches(structure, strategy)
        
        # Populate tree with matches
        for match in matches:
            img_name = os.path.basename(match['image']) if match['image'] else 'N/A'
            gt_name = os.path.basename(match['ground_truth']) if match['ground_truth'] else 'MISSING'
            pred_name = os.path.basename(match['predictions']) if match['predictions'] else 'MISSING'
            
            # Determine status
            if match['ground_truth'] and match['predictions']:
                status = '✓ Complete'
                tag = 'complete'
            elif match['ground_truth'] or match['predictions']:
                status = '⚠ Partial'
                tag = 'partial'
            else:
                status = '✗ No Annotations'
                tag = 'missing'
            
            item = self.preview_tree.insert('', 'end', values=(img_name, gt_name, pred_name, status))
            self.preview_tree.set(item, 'Image', img_name)
        
        # Configure tags for colors
        self.preview_tree.tag_configure('complete', background='lightgreen')
        self.preview_tree.tag_configure('partial', background='lightyellow')
        self.preview_tree.tag_configure('missing', background='lightcoral')
    
    def find_file_matches(self, structure, strategy):
        """Find matching files berdasarkan strategy"""
        matches = []
        
        if strategy == "exact_name":
            matches = self.find_exact_name_matches(structure)
        elif strategy == "flexible":
            matches = self.find_flexible_matches(structure)
        elif strategy == "manual":
            matches = self.find_manual_pattern_matches(structure)
        
        return matches
    
    def find_exact_name_matches(self, structure):
        """Find matches dengan exact name matching"""
        matches = []
        
        for img_path in structure['images']:
            img_base = os.path.splitext(os.path.basename(img_path))[0]
            img_dir = os.path.dirname(img_path)
            
            # Find matching txt file
            gt_path = None
            for txt_path in structure['txt_files']:
                txt_base = os.path.splitext(os.path.basename(txt_path))[0]
                if txt_base == img_base:
                    gt_path = txt_path
                    break
            
            # Find matching json file
            pred_path = None
            for json_path in structure['json_files']:
                json_base = os.path.splitext(os.path.basename(json_path))[0]
                if json_base == img_base:
                    pred_path = json_path
                    break
            
            matches.append({
                'image': img_path,
                'ground_truth': gt_path,
                'predictions': pred_path
            })
        
        return matches
    
    def find_flexible_matches(self, structure):
        """Find matches dengan flexible matching (substring matching)"""
        matches = []
        
        for img_path in structure['images']:
            img_base = os.path.splitext(os.path.basename(img_path))[0]
            
            # Find best matching txt file
            gt_path = None
            best_gt_score = 0
            for txt_path in structure['txt_files']:
                txt_base = os.path.splitext(os.path.basename(txt_path))[0]
                score = self.calculate_name_similarity(img_base, txt_base)
                if score > best_gt_score and score > 0.5:  # Minimum 50% similarity
                    best_gt_score = score
                    gt_path = txt_path
            
            # Find best matching json file
            pred_path = None
            best_pred_score = 0
            for json_path in structure['json_files']:
                json_base = os.path.splitext(os.path.basename(json_path))[0]
                score = self.calculate_name_similarity(img_base, json_base)
                if score > best_pred_score and score > 0.5:
                    best_pred_score = score
                    pred_path = json_path
            
            matches.append({
                'image': img_path,
                'ground_truth': gt_path,
                'predictions': pred_path
            })
        
        return matches
    
    def find_manual_pattern_matches(self, structure):
        """Find matches menggunakan manual patterns"""
        import fnmatch
        
        matches = []
        gt_pattern = self.gt_pattern_var.get()
        pred_pattern = self.pred_pattern_var.get()
        
        for img_path in structure['images']:
            img_base = os.path.splitext(os.path.basename(img_path))[0]
            
            # Find matching ground truth file
            gt_path = None
            gt_search_pattern = gt_pattern.replace('*', img_base)
            for txt_path in structure['txt_files']:
                txt_name = os.path.basename(txt_path)
                if fnmatch.fnmatch(txt_name, gt_search_pattern) or img_base in txt_name:
                    gt_path = txt_path
                    break
            
            # Find matching predictions file
            pred_path = None
            pred_search_pattern = pred_pattern.replace('*', img_base)
            for json_path in structure['json_files']:
                json_name = os.path.basename(json_path)
                if fnmatch.fnmatch(json_name, pred_search_pattern) or img_base in json_name:
                    pred_path = json_path
                    break
            
            matches.append({
                'image': img_path,
                'ground_truth': gt_path,
                'predictions': pred_path
            })
        
        return matches
    
    def calculate_name_similarity(self, name1, name2):
        """Calculate similarity between two filenames"""
        # Simple similarity based on common substrings
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        if name1_lower == name2_lower:
            return 1.0
        
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.8
        
        # Count common characters
        common_chars = 0
        for char in set(name1_lower):
            if char in name2_lower:
                common_chars += min(name1_lower.count(char), name2_lower.count(char))
        
        max_length = max(len(name1_lower), len(name2_lower))
        return common_chars / max_length if max_length > 0 else 0
    
    def start_smart_batch_processing(self, folder_path, dialog):
        """Start batch processing dengan smart matching"""
        try:
            structure = self.analyze_folder_structure(folder_path)
            matches = self.find_file_matches(structure, self.matching_strategy.get())
            
            # Filter only complete matches
            complete_matches = [m for m in matches if m['ground_truth'] and m['predictions']]
            
            if not complete_matches:
                messagebox.showwarning("Warning", "No complete matches found (files with both GT and predictions)")
                return
            
            # Confirm processing
            result = messagebox.askyesno("Confirm Processing", 
                f"Found {len(complete_matches)} complete matches out of {len(matches)} images.\n\n"
                f"Proceed with batch processing?")
            
            if not result:
                return
            
            # Close dialog and start processing
            dialog.destroy()
            
            # Create custom batch results structure
            batch_results = {
                'processed_files': [],
                'failed_files': [],
                'missing_gt_files': [m['image'] for m in matches if not m['ground_truth']],
                'missing_pred_files': [m['image'] for m in matches if not m['predictions']],
                'overall_metrics': self.validation_gui.validation_results.__class__(),
                'individual_results': {}
            }
            
            # Process each complete match
            self.process_smart_batch_matches(complete_matches, batch_results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Smart batch processing failed: {str(e)}")
    
    def process_smart_batch_matches(self, matches, batch_results):
        """Process the matched files"""
        total_matches = len(matches)
        
        # Create progress dialog
        progress_dialog = tk.Toplevel(self.validation_gui.root)
        progress_dialog.title("Processing Batch Files")
        progress_dialog.geometry("500x200")
        progress_dialog.transient(self.validation_gui.root)
        progress_dialog.grab_set()
        
        progress_frame = ttk.Frame(progress_dialog, padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        progress_label = ttk.Label(progress_frame, text="Initializing...")
        progress_label.pack(pady=5)
        
        progress_bar = ttk.Progressbar(progress_frame, mode='determinate', maximum=total_matches)
        progress_bar.pack(fill=tk.X, pady=5)
        
        status_label = ttk.Label(progress_frame, text="")
        status_label.pack(pady=5)
        
        # Process each match
        for i, match in enumerate(matches):
            try:
                # Update progress
                progress_bar['value'] = i
                filename = os.path.basename(match['image'])
                progress_label.config(text=f"Processing: {filename}")
                status_label.config(text=f"{i+1}/{total_matches}")
                progress_dialog.update()
                
                # Process single file
                result = self.process_single_match(match, batch_results)
                
                if result['success']:
                    batch_results['processed_files'].append(result)
                else:
                    batch_results['failed_files'].append({
                        'file': match['image'],
                        'error': result['error']
                    })
                
            except Exception as e:
                batch_results['failed_files'].append({
                    'file': match['image'],
                    'error': str(e)
                })
        
        # Finalize
        progress_bar['value'] = total_matches
        progress_label.config(text="Generating report...")
        progress_dialog.update()
        
        # Generate report
        self.validation_gui.generate_batch_report(batch_results, os.path.dirname(matches[0]['image']))
        
        # Close progress dialog
        progress_dialog.destroy()
        
        # Show completion
        self.validation_gui.show_batch_completion_dialog(batch_results, progress_dialog)
    
    def process_single_match(self, match, batch_results):
        """Process a single matched set of files"""
        try:
            from PIL import Image
            import numpy as np
            
            result = {
                'image_path': match['image'],
                'gt_path': match['ground_truth'],
                'pred_path': match['predictions'],
                'success': False,
                'metrics': None,
                'error': None
            }
            
            # Load image
            pil_image = Image.open(match['image'])
            pil_image = pil_image.convert("RGB")
            image = np.array(pil_image)
            
            # Load ground truth
            gt_annotations = self.load_ground_truth_smart(match['ground_truth'])
            if not gt_annotations:
                result['error'] = "No valid ground truth annotations"
                return result
            
            # Load predictions
            pred_annotations = self.load_predictions_smart(match['predictions'])
            if not pred_annotations:
                result['error'] = "No valid prediction annotations"
                return result
            
            # Run validation
            file_validation = self.validation_gui.validation_results.__class__()
            matches_found, unmatched_gt, unmatched_pred = self.validation_gui.comparison.match_annotations(
                gt_annotations, pred_annotations
            )
            
            # Process matches
            for match_result in matches_found:
                file_validation.add_result(match_result['pred_class'], match_result['gt_class'], matched=True)
                batch_results['overall_metrics'].add_result(match_result['pred_class'], match_result['gt_class'], matched=True)
            
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
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
    
    def load_ground_truth_smart(self, file_path):
        """Smart loading untuk ground truth files"""
        try:
            annotations = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 9:
                        continue
                    
                    _, label, *_, x1, y1, x2, y2 = parts
                    x1, y1 = map(float, [x1, y1])
                    
                    # Convert center point to bounding box
                    box_size = 128
                    half = box_size / 2
                    x1_box = x1 - half
                    y1_box = y1 - half
                    x2_box = x1_box + box_size
                    y2_box = y1_box + box_size
                    
                    # Map label to class (default to falciparum for parasites)
                    if label == 'Parasite':
                        class_idx = 0  # falciparum
                        class_name = 'falciparum'
                    elif label == 'Parasitized':
                        class_idx = 1  # vivax
                        class_name = 'vivax'
                    else:
                        class_idx = 2  # uninfected
                        class_name = 'uninfected'
                    
                    annotations.append({
                        'bbox': (x1_box, y1_box, x2_box, y2_box),
                        'class': class_idx,
                        'class_name': class_name,
                        'label': label
                    })
            
            return annotations
            
        except Exception as e:
            print(f"Error loading GT file {file_path}: {e}")
            return []
    
    def load_predictions_smart(self, file_path):
        """Smart loading untuk prediction files"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            annotations = []
            
            if 'annotations' in data:
                for ann in data['annotations']:
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
            
            return annotations
            
        except Exception as e:
            print(f"Error loading predictions file {file_path}: {e}")
            return []

# Fungsi untuk menambahkan ke MalariaValidationGUI class
def add_smart_batch_loader_to_gui(validation_gui):
    """Add smart batch loader ke existing validation GUI"""
    
    # Create smart batch loader instance
    smart_loader = SmartBatchLoader(validation_gui)
    
    # Replace existing load_batch_folder method
    def load_smart_batch_folder():
        """Load batch folder with smart matching"""
        from tkinter import filedialog
        
        folder_path = filedialog.askdirectory(title="Select folder containing images and annotations")
        
        if folder_path:
            smart_loader.show_smart_batch_dialog(folder_path)
    
    # Add button to existing GUI
    if hasattr(validation_gui, 'setup_gui'):
        # Find the file_frame and add smart batch button
        root = validation_gui.root
        
        # Create new button (you can add this manually to the file_frame in your code)
        def add_smart_batch_button():
            # Find the file loading frame
            for widget in root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for subwidget in widget.winfo_children():
                        if isinstance(subwidget, ttk.LabelFrame) and subwidget['text'] == 'Load Data':
                            ttk.Button(subwidget, text="Smart Batch Loader", 
                                     command=load_smart_batch_folder).pack(fill=tk.X, pady=2)
                            break
        
        # You can call this function to add the button
        # add_smart_batch_button()
    
    return smart_loader

# Instructions untuk penggunaan:
print("""
CARA MENGGUNAKAN SMART BATCH LOADER:

1. Tambahkan kode ini ke model_val.py dengan cara:
   - Import SmartBatchLoader class
   - Dalam MalariaValidationGUI.__init__, tambahkan:
     self.smart_loader = SmartBatchLoader(self)
   
2. Tambahkan button baru di file_frame:
   ttk.Button(file_frame, text="Smart Batch Loader", 
             command=self.load_smart_batch_folder).pack(fill=tk.X, pady=2)

3. Tambahkan method baru di MalariaValidationGUI:
   def load_smart_batch_folder(self):
       folder_path = filedialog.askdirectory(title="Select folder for smart batch processing")
       if folder_path:
           self.smart_loader.show_smart_batch_dialog(folder_path)

FITUR SMART BATCH LOADER:
✓ Analisa struktur folder otomatis
✓ Deteksi pola penamaan file
✓ 3 strategi matching: exact, flexible, manual
✓ Preview matching results sebelum processing
✓ Handling file tanpa suffix konsisten
✓ Progress tracking untuk batch processing
✓ Error handling yang robust

STRATEGI MATCHING:
1. Exact Name: image.jpg ↔ image.txt ↔ image.json
2. Flexible: Cari file dengan nama paling mirip
3. Manual Pattern: Specify pattern sendiri (*.txt, *_gt.txt, dll)
""")