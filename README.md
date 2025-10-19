# 🎯 UniversalObjectDetector - AI-Powered Object Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![GUI](https://img.shields.io/badge/GUI-PyQt5-orange)

**An intelligent desktop application that uses deep learning to recognize and find any objects in your photo and video collections.**

## ✨ Features

- 🎯 **Universal Recognition** - Train the model to recognize people, animals, objects, buildings - anything!
- 📸 **Smart Photo Search** - Scan thousands of images to find your target object
- 🎬 **Video Analysis** - Automatically detect objects in video frames
- 🖼️ **User-Friendly GUI** - Intuitive PyQt5 desktop interface
- 🚀 **GPU Acceleration** - Fast processing with CUDA support
- 📊 **Customizable Sensitivity** - Adjustable confidence threshold
- 💾 **Model Training** - Easy training interface with transfer learning

## 🏗️ Architecture

```
UniversalObjectDetector/
├── 📁 training_data/          # Training dataset structure
├── 🧠 train_custom_detector.py # Model training script
├── 🖼️ find_my_object_gui.py   # Main GUI application
├── 📁 setup_training_folders.py # Dataset preparation
├── 📄 requirements.txt        # Python dependencies
└── 📄 README.md              # This file
```

## 🚀 Quick Start

### 1. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/WiktorProgramista/UniversalObjectDetector.git
cd UniversalObjectDetector

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Training Data

```bash
# Create folder structure for your target object
python setup_training_folders.py "cat"
```

### 3. Organize Your Images

Add images to these folders:
- `training_data/cat/` - 50-200 images of your target object
- `training_data/similar_objects/` - Similar objects (other cats/animals)
- `training_data/other_objects/` - Various different objects
- `training_data/background/` - Backgrounds without the target

### 4. Train Your Model

```bash
# Train the recognition model
python train_custom_detector.py
```

### 5. Start Searching!

```bash
# Launch the graphical interface
python find_my_object_gui.py
```

## 📁 Project Structure

### Core Application Files:
- `setup_training_folders.py` - Creates training folder structure
- `train_custom_detector.py` - Model training with PyTorch
- `find_my_object_gui.py` - Main GUI search interface

### Generated Files:
- `[target]_model.pth` - Trained PyTorch model
- `[target]_model_info.json` - Training information and metrics

## 🎯 Real-World Use Cases

### 🐾 Pet Recognition
- Find all photos of your specific pet in large collections
- Monitor pet activity in video recordings
- Organize pet photo albums automatically

### 👥 People Detection
- Find specific people in family photo archives
- Security monitoring and person recognition
- Organize photos by individuals

### 🏠 Property Monitoring
- Detect specific vehicles or objects
- Home security video analysis
- Inventory management through images

### 🎨 Collection Organization
- Find specific items in large photo collections
- Archive management and categorization
- Content-based image retrieval

## 🔧 Technical Details

### Model Architecture
- **Backbone**: ResNet-50 (pre-trained on ImageNet)
- **Classifier**: Custom binary classification head
- **Technique**: Transfer learning with fine-tuning
- **Output**: Probability score (0-1) for target presence

### Performance Metrics
- **Accuracy**: 85-95% (depending on training data quality)
- **Processing Speed**: 10-50 images/second (with GPU)
- **Video Analysis**: 2-10 frames/second extraction
- **Memory Usage**: 1-2GB RAM during operation

## 🛠️ Advanced Usage

### Custom Training Parameters

```python
from train_custom_detector import CustomObjectDetector

detector = CustomObjectDetector()
# Custom training with more epochs
detector.train('training_data', 'my_target', epochs=50)
```

### Batch Processing

```bash
# Search multiple folders sequentially
for folder in /path/to/folder1 /path/to/folder2 /path/to/folder3; do
    python find_my_object_gui.py --directory "$folder" --target "my_object"
done
```

### Confidence Threshold Guide
- **0.50-0.70**: High sensitivity (more results, potential false positives)
- **0.70-0.85**: Recommended balance (good accuracy and recall)
- **0.85-0.95**: High precision (fewer results, higher accuracy)

## 📊 System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- CPU only (slower processing)

### Recommended
- Python 3.9+
- 8GB RAM
- 5GB free disk space
- NVIDIA GPU with CUDA support
- SSD storage for faster file access

## 🐛 Troubleshooting

### Common Issues & Solutions

**Problem**: "Model not found" error
```bash
# Make sure you've trained the model first
python train_custom_detector.py
```

**Problem**: Slow performance
```bash
# Enable GPU acceleration if available
# The application automatically uses CUDA if detected
```

**Problem**: Installation errors
```bash
# Try installing PyTorch separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Problem**: No results found
- Lower the confidence threshold in the GUI
- Add more training images (50+ recommended)
- Ensure training images are diverse and high quality

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution
- Additional model architectures
- Performance optimizations
- New GUI features
- Documentation improvements
- Bug fixes and testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **WiktorProgramista** - Initial development - [WiktorProgramista](https://github.com/WiktorProgramista)

## 🙏 Acknowledgments

- **PyTorch Team** - For the excellent deep learning framework
- **OpenCV Community** - For comprehensive computer vision tools
- **PyQt Developers** - For the robust desktop application framework
- **ImageNet Contributors** - For the pre-trained models

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#🐛-troubleshooting) section above
2. Search existing [Issues](https://github.com/WiktorProgramista/UniversalObjectDetector/issues)
3. Create a new Issue with detailed description

---

**Happy object hunting! 🎯**
