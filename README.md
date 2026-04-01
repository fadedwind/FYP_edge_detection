# Edge Detection Toolkits

A comprehensive web application for edge detection using both classical algorithms (Sobel, Canny) and deep learning methods (HED, PiDiNet), with support for BSDS500 dataset evaluation.

## 1. Requirements: Software

### Backend Requirements

- **Python**: 3.7 or higher
- **Python Packages** (install via `pip install -r requirements.txt`):
  - `opencv-python` - Image processing and computer vision operations
  - `numpy` - Numerical computing
  - `Pillow` - Image manipulation
  - `matplotlib` - Plotting and visualization (for PR curves)
  - `scipy` - Scientific computing
  - `flask` - Web framework for API
  - `flask-cors` - Cross-origin resource sharing support
  - `torch` - PyTorch (required for HED and PiDiNet)
  - `torchvision` - PyTorch vision utilities
  - `ultralytics` - YOLO models (optional, for vehicle detection)
  - `timm` - PyTorch image models (optional)

**Note**: PyTorch installation may require platform-specific wheels. Please refer to [PyTorch official website](https://pytorch.org/) for the appropriate installation command based on your CUDA/CPU environment.

**Example installations:**
```bash
# CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Frontend Requirements

- **Node.js**: 14.x or higher
- **npm**: 6.x or higher (comes with Node.js)

**Frontend Dependencies** (install via `npm install` in `vue-frontend/` directory):
- `vue` (^3.3.4) - Frontend framework
- `axios` (^1.6.0) - HTTP client
- `vue-router` (^4.2.5) - Routing
- `vite` (^5.0.0) - Build tool
- `@vitejs/plugin-vue` (^4.5.0) - Vite Vue plugin

## 2. Pretrained Models

The project requires pretrained model files for deep learning algorithms. Place all model files in the `models/` directory at the project root.

### HED (Holistically-Nested Edge Detection)

**Required Files:**
- `hed_deploy.prototxt` - Network architecture definition (Caffe prototxt format)
- `hed_pretrained_bsds.caffemodel` or `hed_bsds.caffemodel` - Pretrained weights (Caffe model format)

**Model Location:**
```
models/
├── hed_deploy.prototxt
└── hed_pretrained_bsds.caffemodel  (or hed_bsds.caffemodel)
```

**Download Sources:**
- HED models are typically available from the original HED project repositories
- The Caffe model can be downloaded from official HED releases or converted from PyTorch checkpoints

### PiDiNet (Pixel Difference Networks)

**Required Files:**
The system supports multiple PiDiNet model variants. Place any of the following `.pth` files in the `models/` directory:

**Lightweight Models (Fast, Recommended for Testing):**
- `table5_pidinet-tiny.pth` - Tiny version (fastest)
- `table5_pidinet-tiny-l.pth` - Tiny-large version
- `table5_pidinet-small.pth` - Small version
- `table5_pidinet-small-l.pth` - Small-large version

**Standard Models (Balanced Speed and Accuracy):**
- `pidinet_bsds.pth` - Standard model trained on BSDS500
- `table5_pidinet.pth` - Standard PiDiNet model

**Large Models (High Accuracy, Slower):**
- `table5_pidinet-l.pth` - Large version
- `table6_pidinet.pth` - Table 6 variant
- `table7_pidinet.pth` - Table 7 variant
- `table5_baseline.pth` - Baseline model

**Model Location:**
```
models/
└── [any of the above .pth files]
```

**Model Selection Priority:**
The system automatically selects models in the following priority order (if `prefer_fast=True`):
1. Lightweight models (tiny/small variants) - for faster inference
2. Standard models - for balanced performance
3. Large models - for maximum accuracy

**Download Sources:**
- PiDiNet models are available from the official PiDiNet repository
- Pre-trained weights can be downloaded from the project's releases or model zoo

### Optional: BSDS500 Dataset

For BSDS500 quick test functionality, the dataset should be placed in the following structure:

```
BSR_bsds500/
└── BSDS500/
    └── data/
        ├── groundTruth/
        │   ├── test/
        │   ├── train/
        │   └── val/
        └── images/
            ├── test/
            ├── train/
            └── val/
```

**Note**: The BSDS500 dataset is optional. The application can work without it for single image and batch processing (without ground truth evaluation).

## 3. Preparation for Testing

### Step 1: Clone or Download the Project

```bash
git clone <repository-url>
cd FYP_edge_detection_git-v1
```

### Step 2: Install Backend Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# If PyTorch is not installed, install it separately based on your system
# For CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA (example for CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Download Pretrained Models

**For HED:**
1. Download `hed_deploy.prototxt` and `hed_pretrained_bsds.caffemodel` (or `hed_bsds.caffemodel`)
2. Place both files in the `models/` directory

**For PiDiNet:**
1. Download at least one PiDiNet model file (`.pth` format)
2. Recommended: Start with `table5_pidinet-tiny.pth` for faster testing
3. Place the model file(s) in the `models/` directory

**Verify Model Files:**
```bash
# Check if models directory exists and contains required files
ls models/
# Should show at least:
# - hed_deploy.prototxt
# - hed_pretrained_bsds.caffemodel (or hed_bsds.caffemodel)
# - [at least one PiDiNet .pth file]
```

### Step 4: Install Frontend Dependencies

```bash
cd vue-frontend
npm install
cd ..
```

### Step 5: (Optional) Prepare BSDS500 Dataset

If you want to use the BSDS500 quick test feature:

1. Download the BSDS500 dataset
2. Extract it to the project root directory
3. Ensure the directory structure matches:
   ```
   BSR_bsds500/BSDS500/data/
   ├── groundTruth/
   └── images/
   ```

### Step 6: Start the Backend Server

**Windows:**
```bash
start_backend.bat
```

**Linux/Mac:**
```bash
python edge_detection_api.py
```

The backend server will start on `http://127.0.0.1:5000`

### Step 7: Start the Frontend Development Server

```bash
cd vue-frontend
npm run dev
```

The frontend will be available at `http://localhost:3000` (or the port assigned by Vite)

### Step 8: Access the Application

Open your web browser and navigate to:
- Frontend: `http://localhost:3000`
- Backend API: `http://127.0.0.1:5000`

## Testing the Application

### Single Image Edge Detection

1. Open the application in your browser
2. Navigate to "Edge Detection" (classical algorithms) or "Deep Learning Edge Detection"
3. Upload an image using the file input or drag-and-drop
4. Select an algorithm (Sobel, Canny, HED, or PiDiNet)
5. Adjust parameters if needed
6. Click "Detect Edges" to process the image

### Batch Processing

1. Open the batch processing panel
2. Upload multiple images
3. Configure algorithm parameters
4. Click "Start Batch Processing"
5. Results will be saved in the `output/` directory

### BSDS500 Quick Test

1. Ensure BSDS500 dataset is properly set up
2. Navigate to the edge detection page
3. Open the batch processing panel
4. Click "BSDS500 Quick Test"
5. Select dataset subset (train/val/test)
6. The system will automatically:
   - Process all images in the selected subset
   - Evaluate against ground truth
   - Generate PR curves
   - Calculate ODS and OIS metrics
   - Save results in `output/` directory

## Troubleshooting

### Model Loading Issues

**HED Model Not Found:**
- Verify that `hed_deploy.prototxt` and `hed_pretrained_bsds.caffemodel` exist in `models/` directory
- Check file permissions and paths

**PiDiNet Model Not Found:**
- Ensure at least one `.pth` file exists in `models/` directory
- Check that the model file is not corrupted
- Try a different model variant if one fails to load

### PyTorch Installation Issues

- Visit [PyTorch official website](https://pytorch.org/) for platform-specific installation instructions
- Ensure Python version is compatible (3.7+)
- For CUDA support, verify CUDA toolkit is installed

### Backend Connection Issues

- Ensure backend server is running on port 5000
- Check for port conflicts
- Verify CORS is enabled in `edge_detection_api.py`

### Frontend Build Issues

- Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Check Node.js version compatibility
- Ensure all dependencies in `package.json` are installed

## Project Structure

```
.
├── edge_detection_api.py          # Flask backend API
├── boundary_evaluation.py          # BSDS500 evaluation module
├── hed.py                          # HED model implementation
├── pidinet.py                      # PiDiNet model implementation
├── requirements.txt                # Python dependencies
├── start_backend.bat               # Backend startup script (Windows)
├── models/                         # Pretrained model files directory
│   ├── hed_deploy.prototxt
│   ├── hed_pretrained_bsds.caffemodel
│   └── [PiDiNet .pth files]
├── vue-frontend/                   # Frontend Vue application
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── BSR_bsds500/                    # BSDS500 dataset (optional)
│   └── BSDS500/
└── output/                         # Output directory (auto-generated)
```

## License

Please refer to the original licenses of the included algorithms and models:
- HED: Check original HED project license
- PiDiNet: Check PiDiNet project license
- BSDS500: Check BSDS500 dataset license

## Acknowledgments

- HED: Holistically-Nested Edge Detection
- PiDiNet: Pixel Difference Networks for Efficient Edge Detection
- BSDS500: Berkeley Segmentation Dataset 500


