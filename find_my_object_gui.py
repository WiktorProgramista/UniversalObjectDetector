# find_my_object_gui.py
import os
import json
import sys
from datetime import datetime
from PIL import Image, ImageOps
import torch
import cv2
import tempfile
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QScrollArea, QGridLayout, QSlider, 
                             QFileDialog, QMessageBox, QFrame, QProgressBar, QSizePolicy,
                             QCheckBox, QComboBox, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QPalette, QFont, QColor, QImage
from train_custom_detector import CustomObjectDetector

class PhotoWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, directory, model_path, confidence_threshold, target_name, include_videos=True):
        super().__init__()
        self.directory = directory
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_name = target_name
        self.include_videos = include_videos
        self.results = {
            'search_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'directory': directory,
            'target_name': target_name,
            'confidence_threshold': confidence_threshold,
            'include_videos': include_videos,
            'detected_photos': [],
            'detected_videos': [],
            'total_photos_scanned': 0,
            'total_videos_scanned': 0
        }

    def extract_video_frames(self, video_path, interval_seconds=2):
        """Ekstrahuje klatki z filmu w regularnych odstępach czasu"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0:
                return frames
            
            frame_interval = int(fps * interval_seconds)
            if frame_interval == 0:
                frame_interval = 1
            
            for frame_num in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
        except Exception as e:
            print(f"Błąd ekstrakcji klatek z {video_path}: {e}")
        
        return frames

    def analyze_video(self, video_path, detector):
        """Analizuje film i zwraca najlepsze klatki z wykrytym obiektem"""
        best_frames = []
        frames = self.extract_video_frames(video_path)
        
        for i, frame in enumerate(frames):
            try:
                image = Image.fromarray(frame)
                input_tensor = detector.val_transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    output = detector.model(input_tensor.to(detector.device))
                    confidence = output.item()

                if confidence >= self.confidence_threshold:
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        image.save(temp_file.name, 'JPEG', quality=85)
                        best_frames.append({
                            'frame_path': temp_file.name,
                            'confidence': confidence,
                            'frame_number': i,
                            'timestamp': i * 2
                        })
                        
            except Exception as e:
                print(f"Błąd analizy klatki {i} z {video_path}: {e}")
        
        return best_frames

    def run(self):
        try:
            detector = CustomObjectDetector()
            
            if not os.path.exists(self.model_path):
                self.error.emit(f"Model {self.model_path} nie istnieje!")
                return

            detector.create_model()
            checkpoint = torch.load(self.model_path, map_location=detector.device)
            detector.model.load_state_dict(checkpoint['model_state_dict'])
            detector.model.eval()

            # Szukaj w typowych folderach z mediami
            media_dirs = ['output/photos', 'output/videos', 'output/gifs', 'output', 
                         'photos', 'videos', 'images', 'media', 'DCIM', 'Pictures']
            found_dirs = []
            
            for media_dir in media_dirs:
                full_dir = os.path.join(self.directory, media_dir)
                if os.path.exists(full_dir):
                    found_dirs.append(full_dir)

            # Jeśli nie znaleziono specyficznych folderów, przeszukaj cały katalog
            if not found_dirs:
                found_dirs = [self.directory]

            total_files = 0
            for media_dir in found_dirs:
                for root, dirs, files in os.walk(media_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            total_files += 1
                        if self.include_videos and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            total_files += 1

            processed = 0
            for media_dir in found_dirs:
                for root, dirs, files in os.walk(media_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.results['total_photos_scanned'] += 1
                            processed += 1

                            try:
                                preview_image = Image.open(file_path).convert('RGB')
                                image_for_model = detector.val_transform(preview_image).unsqueeze(0)
                                
                                with torch.no_grad():
                                    output = detector.model(image_for_model.to(detector.device))
                                    confidence = output.item()

                                if confidence >= self.confidence_threshold:
                                    detected_photo = {
                                        'file_path': file_path,
                                        'confidence': confidence,
                                        'file_name': file,
                                        'directory': root,
                                        'type': 'photo',
                                        'preview_image': preview_image
                                    }
                                    self.results['detected_photos'].append(detected_photo)
                                    self.progress.emit(processed, total_files, f"Znaleziono {self.target_name}! {file}")

                            except Exception as e:
                                print(f"Błąd przetwarzania {file}: {e}")

                        elif self.include_videos and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            self.results['total_videos_scanned'] += 1
                            processed += 1
                            
                            self.progress.emit(processed, total_files, f"Analizuję film: {file}")
                            
                            try:
                                best_frames = self.analyze_video(file_path, detector)
                                
                                for frame_data in best_frames:
                                    video_result = {
                                        'file_path': file_path,
                                        'frame_path': frame_data['frame_path'],
                                        'confidence': frame_data['confidence'],
                                        'file_name': file,
                                        'directory': root,
                                        'type': 'video',
                                        'timestamp': frame_data['timestamp'],
                                        'frame_number': frame_data['frame_number']
                                    }
                                    self.results['detected_videos'].append(video_result)
                                    self.progress.emit(processed, total_files, f"{self.target_name} w filmie! {file} @ {frame_data['timestamp']}s")

                            except Exception as e:
                                print(f"Błąd przetwarzania filmu {file}: {e}")

                        if processed % 10 == 0:
                            self.progress.emit(processed, total_files, f"Przetwarzanie... {file}")

            # Posortuj wyniki
            self.results['detected_photos'].sort(key=lambda x: x['confidence'], reverse=True)
            self.results['detected_videos'].sort(key=lambda x: x['confidence'], reverse=True)
            self.finished.emit(self.results)

        except Exception as e:
            self.error.emit(str(e))

class VideoThumbnail(QFrame):
    def __init__(self, video_data, target_name, parent=None):
        super().__init__(parent)
        self.video_data = video_data
        self.target_name = target_name
        self.setup_ui()

    def setup_ui(self):
        self.setFixedSize(220, 280)
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("""
            QFrame {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: white;
            }
            QFrame:hover {
                border: 2px solid #FF6B00;
                background-color: #f8f9fa;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        self.video_icon = QLabel("🎬 FILM")
        self.video_icon.setAlignment(Qt.AlignCenter)
        self.video_icon.setStyleSheet("color: #FF6B00; font-size: 12px; font-weight: bold; background-color: #FFF3E0; border-radius: 4px; padding: 2px;")
        layout.addWidget(self.video_icon)
        
        self.image_container = QLabel()
        self.image_container.setFixedSize(200, 150)
        self.image_container.setAlignment(Qt.AlignCenter)
        self.image_container.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 4px;")
        self.image_container.setScaledContents(True)
        layout.addWidget(self.image_container)
        
        self.name_label = QLabel()
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("color: black; font-size: 10px; font-weight: bold;")
        self.name_label.setWordWrap(True)
        self.name_label.setMaximumHeight(30)
        layout.addWidget(self.name_label)
        
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("color: #666; font-size: 9px;")
        layout.addWidget(self.time_label)
        
        self.confidence_label = QLabel()
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("color: black; font-size: 10px;")
        layout.addWidget(self.confidence_label)
        
        self.load_thumbnail()

    def load_thumbnail(self):
        try:
            if os.path.exists(self.video_data['frame_path']):
                pixmap = QPixmap(self.video_data['frame_path'])
                
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(190, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_container.setPixmap(scaled_pixmap)
                else:
                    self.show_error()
            else:
                self.show_error()
            
            short_name = os.path.basename(self.video_data['file_path'])
            if len(short_name) > 25:
                short_name = short_name[:22] + "..."
            self.name_label.setText(short_name)
            
            minutes = int(self.video_data['timestamp']) // 60
            seconds = int(self.video_data['timestamp']) % 60
            self.time_label.setText(f"Czas: {minutes:02d}:{seconds:02d}")
            
            self.confidence_label.setText(f"Pewność: {self.video_data['confidence']:.3f}")
            
            self.setToolTip(f"{self.video_data['file_name']}\nCzas: {minutes:02d}:{seconds:02d}\nPewność: {self.video_data['confidence']:.3f}")
            
        except Exception as e:
            print(f"Błąd ładowania miniatury filmu: {e}")
            self.show_error()

    def show_error(self):
        self.image_container.setText("Błąd\nładowania")
        self.image_container.setStyleSheet("color: red; font-weight: bold; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px;")
        self.name_label.setText("Błąd")
        self.time_label.setText("")
        self.confidence_label.setText("")

class PhotoThumbnail(QFrame):
    def __init__(self, photo_data, target_name, parent=None):
        super().__init__(parent)
        self.photo_data = photo_data
        self.target_name = target_name
        self.setup_ui()

    def setup_ui(self):
        self.setFixedSize(220, 250)
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("""
            QFrame {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: white;
            }
            QFrame:hover {
                border: 2px solid #007AFF;
                background-color: #f8f9fa;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        self.image_container = QLabel()
        self.image_container.setFixedSize(200, 200)
        self.image_container.setAlignment(Qt.AlignCenter)
        self.image_container.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 4px;")
        self.image_container.setScaledContents(True)
        layout.addWidget(self.image_container)
        
        self.name_label = QLabel()
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("color: black; font-size: 10px; font-weight: bold;")
        self.name_label.setWordWrap(True)
        self.name_label.setMaximumHeight(30)
        layout.addWidget(self.name_label)
        
        self.confidence_label = QLabel()
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("color: black; font-size: 10px;")
        layout.addWidget(self.confidence_label)
        
        self.load_thumbnail()

    def load_thumbnail(self):
        try:
            file_path = self.photo_data['file_path']
            
            if os.path.exists(file_path):
                pixmap = QPixmap(file_path)
                
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(190, 190, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_container.setPixmap(scaled_pixmap)
                else:
                    self.show_error()
            else:
                self.show_error()
            
            short_name = os.path.basename(file_path)
            if len(short_name) > 25:
                short_name = short_name[:22] + "..."
            self.name_label.setText(short_name)
            self.confidence_label.setText(f"Pewność: {self.photo_data['confidence']:.3f}")
            
            self.setToolTip(f"{self.photo_data['file_name']}\nPewność: {self.photo_data['confidence']:.3f}")
            
        except Exception as e:
            print(f"Błąd ładowania miniatury zdjęcia: {e}")
            self.show_error()

    def show_error(self):
        self.image_container.setText("Błąd\nładowania")
        self.image_container.setStyleSheet("color: red; font-weight: bold; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px;")
        self.name_label.setText("Błąd")
        self.confidence_label.setText("")

class ObjectPhotoViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.results = None
        self.current_target = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("🔍 Custom Object Finder - Zdjęcia i Filmy")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            QLabel {
                color: black;
            }
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056CC;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QComboBox {
                padding: 6px;
                border: 2px solid #cccccc;
                border-radius: 6px;
                background-color: white;
                color: black;
            }
            QComboBox:focus {
                border: 2px solid #007AFF;
            }
            QCheckBox {
                color: black;
                font-weight: bold;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 2px solid #cccccc;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #007AFF;
                border: 2px solid #0056CC;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #007AFF, stop:1 #0056CC);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
                color: white;
                font-weight: bold;
                background-color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: #007AFF;
                border-radius: 3px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("🔍 Custom Object Finder - Zdjęcia i Filmy")
        header.setFont(QFont("Arial", 20, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: black; margin: 20px;")
        layout.addWidget(header)

        # Controls
        controls_layout = QHBoxLayout()
        
        # Target selection
        target_label = QLabel("Cel:")
        target_label.setStyleSheet("color: black; font-weight: bold;")
        controls_layout.addWidget(target_label)
        
        self.target_combo = QComboBox()
        self.target_combo.setFixedWidth(150)
        self.target_combo.currentTextChanged.connect(self.on_target_changed)
        controls_layout.addWidget(self.target_combo)
        
        # Directory selection
        self.dir_btn = QPushButton("Wybierz folder")
        self.dir_btn.clicked.connect(self.select_directory)
        controls_layout.addWidget(self.dir_btn)

        self.dir_label = QLabel("Nie wybrano folderu")
        self.dir_label.setStyleSheet("color: black; font-style: italic;")
        controls_layout.addWidget(self.dir_label)

        controls_layout.addStretch()

        # Include videos checkbox
        self.videos_checkbox = QCheckBox("🔍 Szukaj w filmach")
        self.videos_checkbox.setChecked(True)
        self.videos_checkbox.setToolTip("Wyszukuj obiekt w filmach MP4, AVI, MOV, MKV")
        self.videos_checkbox.stateChanged.connect(self.on_videos_checkbox_changed)
        controls_layout.addWidget(self.videos_checkbox)

        # Confidence slider
        confidence_text = QLabel("Próg pewności:")
        confidence_text.setStyleSheet("color: black;")
        controls_layout.addWidget(confidence_text)
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(50, 95)
        self.confidence_slider.setValue(85)
        self.confidence_slider.valueChanged.connect(self.on_confidence_change)
        controls_layout.addWidget(self.confidence_slider)

        self.confidence_label = QLabel("0.85")
        self.confidence_label.setFixedWidth(40)
        self.confidence_label.setStyleSheet("color: black;")
        controls_layout.addWidget(self.confidence_label)

        # Search button
        self.search_btn = QPushButton("Szukaj!")
        self.search_btn.clicked.connect(self.start_search)
        self.search_btn.setEnabled(False)
        controls_layout.addWidget(self.search_btn)

        layout.addLayout(controls_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: black; font-weight: bold;")
        layout.addWidget(self.progress_label)

        # Results area
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setVisible(False)
        self.results_scroll.setStyleSheet("background-color: white; border: none;")
        
        self.results_widget = QWidget()
        self.results_widget.setStyleSheet("background-color: white;")
        self.results_layout = QGridLayout(self.results_widget)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.results_layout.setSpacing(15)
        self.results_layout.setContentsMargins(20, 20, 20, 20)
        self.results_scroll.setWidget(self.results_widget)
        
        layout.addWidget(self.results_scroll)

        # Status bar
        self.status_bar = QLabel("Wybierz cel i folder do wyszukiwania")
        self.status_bar.setStyleSheet("background-color: #e9e9e9; padding: 8px; border-top: 1px solid #ccc; color: black;")
        layout.addWidget(self.status_bar)

        self.load_available_models()
        self.update_status()

    def load_available_models(self):
        """Ładuje dostępne modele z plików .pth"""
        self.target_combo.clear()
        for file in os.listdir('.'):
            if file.endswith('_model.pth'):
                target_name = file.replace('_model.pth', '')
                self.target_combo.addItem(target_name)

    def on_target_changed(self, target_name):
        """Aktualizuje interfejs po zmianie celu"""
        self.current_target = target_name
        self.update_status()

    def on_videos_checkbox_changed(self):
        """Aktualizuje status po zmianie checkboxa filmów"""
        self.update_status()

    def on_confidence_change(self, value):
        """Aktualizuje etykietę suwaka pewności"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        self.update_status()

    def select_directory(self):
        """Wybiera folder do przeszukania"""
        directory = QFileDialog.getExistingDirectory(self, "Wybierz folder z mediami")
        if directory:
            self.selected_directory = directory
            self.dir_label.setText(os.path.basename(directory))
            self.update_status()

    def update_status(self):
        """Aktualizuje status i przyciski"""
        has_target = self.current_target is not None
        has_directory = hasattr(self, 'selected_directory')
        self.search_btn.setEnabled(has_target and has_directory)
        
        if has_target and has_directory:
            confidence = self.confidence_slider.value() / 100.0
            include_videos = self.videos_checkbox.isChecked()
            video_text = "i filmach" if include_videos else ""
            self.status_bar.setText(f"🎯 Szukaj: {self.current_target} | 📁 {os.path.basename(self.selected_directory)} | "
                                  f"⚡ Pewność: {confidence:.2f} | 📹 {video_text}")
        else:
            self.status_bar.setText("Wybierz cel i folder do wyszukiwania")

    def start_search(self):
        """Rozpoczyna wyszukiwanie"""
        if not hasattr(self, 'selected_directory'):
            QMessageBox.warning(self, "Błąd", "Wybierz folder do przeszukania!")
            return

        self.results_scroll.setVisible(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Przygotowywanie...")

        model_path = f"{self.current_target}_model.pth"
        confidence = self.confidence_slider.value() / 100.0
        include_videos = self.videos_checkbox.isChecked()

        self.worker = PhotoWorker(
            self.selected_directory, 
            model_path, 
            confidence,
            self.current_target,
            include_videos
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.search_finished)
        self.worker.error.connect(self.search_error)
        self.worker.start()

    def update_progress(self, current, total, message):
        """Aktualizuje pasek postępu"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_label.setText(f"{message} ({current}/{total})")

    def search_finished(self, results):
        """Wyświetla wyniki wyszukiwania"""
        self.results = results
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self.results_scroll.setVisible(True)
        
        self.display_results()

    def search_error(self, error_message):
        """Wyświetla błąd wyszukiwania"""
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        QMessageBox.critical(self, "Błąd", f"Wystąpił błąd: {error_message}")

    def display_results(self):
        """Wyświetla znalezione zdjęcia i filmy"""
        # Wyczyść poprzednie wyniki
        for i in reversed(range(self.results_layout.count())): 
            widget = self.results_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        total_photos = len(self.results['detected_photos'])
        total_videos = len(self.results['detected_videos'])
        total_results = total_photos + total_videos
        
        if total_results == 0:
            no_results = QLabel(f"❌ Nie znaleziono {self.current_target} w wybranym folderze\n\n"
                              "Spróbuj:\n"
                              "• Obniżyć próg pewności\n"
                              "• Dodać więcej zdjęć do treningu\n"
                              "• Sprawdzić inne foldery")
            no_results.setFont(QFont("Arial", 12))
            no_results.setAlignment(Qt.AlignCenter)
            no_results.setStyleSheet("color: black; padding: 40px;")
            self.results_layout.addWidget(no_results, 0, 0, 1, 4)
            return

        # Display photos and videos in grid
        row, col = 0, 0
        max_cols = 4
        
        # Najpierw filmy
        for video in self.results['detected_videos']:
            thumbnail = VideoThumbnail(video, self.current_target)
            thumbnail.mousePressEvent = lambda event, vd=video: self.open_video_frame(vd)
            self.results_layout.addWidget(thumbnail, row, col, Qt.AlignCenter)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Potem zdjęcia
        for photo in self.results['detected_photos']:
            thumbnail = PhotoThumbnail(photo, self.current_target)
            thumbnail.mousePressEvent = lambda event, pd=photo: self.open_photo(pd)
            self.results_layout.addWidget(thumbnail, row, col, Qt.AlignCenter)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        total_found = len(self.results['detected_photos']) + len(self.results['detected_videos'])
        search_type = "zdjęciach i filmach" if self.results['include_videos'] else "zdjęciach"
        self.status_bar.setText(f"Wyświetlono {total_found} wyników w {search_type}!")

    def open_photo(self, photo_data):
        """Otwiera zdjęcie w pełnym rozmiarze"""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"📸 {os.path.basename(photo_data['file_path'])}")
            dialog.setGeometry(200, 200, 1000, 800)
            dialog.setStyleSheet("background-color: white;")
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            
            # Załaduj i wyświetl obraz
            pixmap = QPixmap(photo_data['file_path'])
            if not pixmap.isNull():
                image_label = QLabel()
                scaled_pixmap = pixmap.scaled(900, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
                image_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(image_label)
            
            # Informacje o pliku
            file_info = QLabel(
                f"Plik: {os.path.basename(photo_data['file_path'])}\n"
                f"Ścieżka: {photo_data['file_path']}\n"
                f"Rozmiar: {pixmap.width()} x {pixmap.height()}\n"
                f"Pewność: {photo_data['confidence']:.3f}\n"
                f"Rozmiar pliku: {os.path.getsize(photo_data['file_path']) / 1024:.1f} KB"
            )
            file_info.setAlignment(Qt.AlignCenter)
            file_info.setStyleSheet("color: black; margin: 10px; font-size: 12px;")
            layout.addWidget(file_info)
            
            # Przycisk zamknięcia
            close_btn = QPushButton("Zamknij")
            close_btn.clicked.connect(dialog.close)
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #007AFF;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-weight: bold;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #0056CC;
                }
            """)
            layout.addWidget(close_btn, 0, Qt.AlignCenter)
            
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie można załadować zdjęcia: {e}")

    def open_video_frame(self, video_data):
        """Pokazuje klatkę z filmu z opcją odtworzenia"""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"🎬 {os.path.basename(video_data['file_path'])}")
            dialog.setGeometry(200, 200, 1000, 800)
            dialog.setStyleSheet("background-color: white;")
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            
            # Załaduj i wyświetl klatkę
            if os.path.exists(video_data['frame_path']):
                pixmap = QPixmap(video_data['frame_path'])
                if not pixmap.isNull():
                    image_label = QLabel()
                    scaled_pixmap = pixmap.scaled(900, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    image_label.setPixmap(scaled_pixmap)
                    image_label.setAlignment(Qt.AlignCenter)
                    layout.addWidget(image_label)
            
            # Informacje o filmie
            minutes = int(video_data['timestamp']) // 60
            seconds = int(video_data['timestamp']) % 60
            
            file_info = QLabel(
                f"Film: {os.path.basename(video_data['file_path'])}\n"
                f"Ścieżka: {video_data['file_path']}\n"
                f"Czas: {minutes:02d}:{seconds:02d}\n"
                f"Pewność: {video_data['confidence']:.3f}\n"
                f"Rozmiar pliku: {os.path.getsize(video_data['file_path']) / (1024*1024):.1f} MB"
            )
            file_info.setAlignment(Qt.AlignCenter)
            file_info.setStyleSheet("color: black; margin: 10px; font-size: 12px;")
            layout.addWidget(file_info)
            
            # Przycisk odtworzenia filmu
            play_btn = QPushButton("Odtwórz film")
            play_btn.clicked.connect(lambda: self.play_video(video_data['file_path']))
            play_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF6B00;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-weight: bold;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #E55A00;
                }
            """)
            layout.addWidget(play_btn, 0, Qt.AlignCenter)
            
            # Przycisk zamknięcia
            close_btn = QPushButton("Zamknij")
            close_btn.clicked.connect(dialog.close)
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #007AFF;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-weight: bold;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #0056CC;
                }
            """)
            layout.addWidget(close_btn, 0, Qt.AlignCenter)
            
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie można załadować klatki: {e}")

    def play_video(self, video_path):
        """Otwiera film w domyślnej aplikacji"""
        try:
            import subprocess
            import platform
            
            if platform.system() == 'Windows':
                os.startfile(video_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', video_path])
            else:  # Linux
                subprocess.run(['xdg-open', video_path])
                
        except Exception as e:
            QMessageBox.information(self, "Odtwarzanie filmu", 
                                  f"Nie można automatycznie otworzyć filmu.\n\nOtwórz ręcznie: {video_path}")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Sprawdź czy są dostępne modele
    models = [f for f in os.listdir('.') if f.endswith('_model.pth')]
    if not models:
        print("❌ Nie znaleziono żadnych modeli!")
        print("\n🎯 Najpierw wytrenuj model:")
        print("1. python setup_training_folders.py [nazwa_celu]")
        print("2. Dodaj zdjęcia do folderów")
        print("3. python train_custom_detector.py")
        return
    
    viewer = ObjectPhotoViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()