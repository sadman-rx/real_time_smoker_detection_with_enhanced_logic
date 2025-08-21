import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from smoking_detection import SmokerLabel, detect_smoking, load_models
import os
from tkinter import messagebox
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Try to import tkinterdnd2, fallback to basic tkinter if not available
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DRAG_DROP_ENABLED = True
except ImportError:
    logger.warning("tkinterdnd2 not available, drag and drop will be disabled")
    DRAG_DROP_ENABLED = False

class ModernDropZone(ttk.Frame):
    def __init__(self, parent, main_gui, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # Store reference to main GUI
        self.gui = main_gui
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('Drop.TFrame', background='#f0f0f0')
        
        # Configure frame
        self.configure(style='Drop.TFrame', padding=20)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create drop zone label
        self.drop_label = ttk.Label(
            self,
            text="Click to browse for image" if not DRAG_DROP_ENABLED else "Drag and drop image here\nor click to browse",
            justify="center"
        )
        self.drop_label.grid(pady=50)
        
        # Configure drag and drop if available
        if DRAG_DROP_ENABLED:
            self.drop_target_register(DND_FILES)
            self.dnd_bind('<<Drop>>', self.handle_drop)
        
        # Bind click event
        self.bind("<Button-1>", self.browse_file)
        self.drop_label.bind("<Button-1>", self.browse_file)

    def handle_drop(self, event):
        file_path = event.data
        logger.debug(f"Received dropped file: {file_path}")
        if file_path:
            # Handle Windows file path format
            file_path = file_path.strip('{}')
            # Remove additional quotes if present
            if file_path.startswith('"') and file_path.endswith('"'):
                file_path = file_path[1:-1]
            logger.debug(f"Processed file path: {file_path}")
            if self.gui is not None:
                self.gui.process_image(file_path)
            else:
                logger.error("No GUI reference")
                messagebox.showerror("Error", "System not properly initialized")
    
    def browse_file(self, event):
        try:
            logger.info("Opening file dialog...")
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
            )
            logger.info(f"Selected file through browse: {file_path}")
            if file_path:
                logger.info("File selected, calling process_image")
                if self.gui is not None:
                    self.gui.process_image(file_path)
                else:
                    logger.error("No GUI reference")
                    messagebox.showerror("Error", "System not properly initialized")
            else:
                logger.info("No file selected")
        except Exception as e:
            logger.error(f"Error browsing file: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Error selecting file: {str(e)}")

class SmokerDetectionGUI(tk.Tk if not DRAG_DROP_ENABLED else TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self._initialize_gui()

    def _initialize_gui(self):
        """Initialize all GUI components"""
        # Load models at startup
        self.status_var = tk.StringVar()
        self.status_var.set("Loading models...")
        
        # Load models
        try:
            logger.info("Loading models...")
            self.object_model, self.landmark_model = load_models()
            logger.info("Models loaded successfully")
            self.status_var.set("Ready to process images")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
            self.status_var.set(f"Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            return

        # Configure main window
        self.title("Smoking Detection System")
        self.geometry("1400x900")  # Increased window size
        self.configure(bg='#ffffff')
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Create header
        header = ttk.Label(
            self,
            text="Smoking Detection System",
            font=('Helvetica', 24),
            padding=20
        )
        header.grid(row=0, column=0, sticky="ew")
        
        # Create main content area
        self.content = ttk.Frame(self, padding=20)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_columnconfigure(1, weight=2)  # Give more space to results
        self.content.grid_rowconfigure(0, weight=1)
        
        # Create left panel (drop zone and reset button)
        self.left_panel = ttk.Frame(self.content)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=10)
        self.left_panel.grid_rowconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=0)  # Don't expand button row
        
        # Add drop zone to left panel
        self.drop_zone = ModernDropZone(self.left_panel, self)  # Pass self as main_gui
        self.drop_zone.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        # Add reset button to left panel
        self.reset_button = ttk.Button(
            self.left_panel,
            text="Reset",
            command=self.reset_interface,
            style='Action.TButton'
        )
        self.reset_button.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        # Create right panel (results)
        self.right_panel = ttk.Frame(self.content)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=10)
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)
        
        # Create image label in right panel
        self.result_label = ttk.Label(self.right_panel)
        self.result_label.grid(row=0, column=0, sticky="nsew")
        
        # Status bar
        self.status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            padding=5
        )
        self.status_bar.grid(row=2, column=0, sticky="ew")

        # Configure styles
        self.style = ttk.Style()
        self.style.configure('Action.TButton', font=('Helvetica', 12))

    def reset_interface(self):
        """Reset the interface to its initial state"""
        logger.info("Resetting interface")
        # Clear the result image
        self.result_label.configure(image='')
        self.result_label.image = None
        # Reset status
        self.status_var.set("Ready")

    def process_image(self, image_path):
        def process():
            try:
                logger.info(f"Starting to process image: {image_path}")
                self.status_var.set("Processing image...")
                
                # Check if file exists
                if not os.path.exists(image_path):
                    logger.error(f"File does not exist: {image_path}")
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                # Check if file is an image
                try:
                    with Image.open(image_path) as img:
                        logger.info(f"Image opened successfully: {img.format}, size={img.size}, mode={img.mode}")
                except Exception as e:
                    logger.error(f"Failed to open image: {str(e)}")
                    raise ValueError(f"Invalid image file: {str(e)}")
                
                # Process the image using smoking detection
                logger.info("Loading models for detection...")
                if not hasattr(self, 'object_model') or not hasattr(self, 'landmark_model'):
                    logger.error("Models not properly initialized")
                    raise RuntimeError("Detection models not properly initialized")
                
                logger.info("Calling detect_smoking function")
                annotated_image, results = detect_smoking(image_path, self.object_model, self.landmark_model, save_output=False)
                logger.info("detect_smoking function completed")
                
                if annotated_image is None:
                    logger.error("detect_smoking returned None")
                    raise ValueError("No results from smoking detection")
                
                # Convert result to PhotoImage
                logger.info("Converting result to PhotoImage")
                result_pil = Image.fromarray(annotated_image)
                
                # Resize while maintaining aspect ratio
                display_size = (800, 800)  # Increased display size
                result_pil.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                result_tk = ImageTk.PhotoImage(result_pil)
                
                # Update GUI
                logger.info("Updating GUI with processed image")
                self.result_label.configure(image=result_tk)
                self.result_label.image = result_tk  # Keep a reference
                
                self.status_var.set("Processing complete")
                logger.info("Image processing completed successfully")
                
            except FileNotFoundError as e:
                logger.error(f"File not found: {str(e)}")
                self.status_var.set("Error: File not found")
                messagebox.showerror("Error", f"File not found: {image_path}")
            except ValueError as e:
                logger.error(f"Invalid image: {str(e)}")
                self.status_var.set("Error: Invalid image file")
                messagebox.showerror("Error", f"Invalid image file: {str(e)}")
            except RuntimeError as e:
                logger.error(f"Runtime error: {str(e)}")
                self.status_var.set("Error: System not properly initialized")
                messagebox.showerror("Error", f"System error: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}", exc_info=True)
                self.status_var.set(f"Error: {str(e)}")
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")

        # Run processing in a separate thread
        threading.Thread(target=process, daemon=True).start()

def main():
    logger.info("Starting Smoking Detection GUI")
    app = SmokerDetectionGUI()
    # Set style
    style = ttk.Style()
    style.configure('TFrame', background='#ffffff')
    style.configure('TLabel', background='#ffffff')
    
    logger.info("Starting main event loop")
    app.mainloop()

if __name__ == "__main__":
    main()
