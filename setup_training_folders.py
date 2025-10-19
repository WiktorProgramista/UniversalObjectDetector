# setup_training_folders.py
import os
import shutil

def create_training_structure(target_name="my_object"):
    """Tworzy strukturę folderów do trenowania modelu dla dowolnego obiektu"""
    
    folders = [
        f'training_data/{target_name}',    # TYLKO Twój cel (osoba, przedmiot, zwierzę)
        'training_data/similar_objects',   # Podobne obiekty
        'training_data/other_objects',     # Inne obiekty
        'training_data/background'         # Tło bez celu
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("✅ Stworzono strukturę folderów:")
    print(f"\n📁 training_data/")
    print(f"├── 📸 {target_name}/        # Umieść zdjęcia Twojego celu")
    print("├── 🔄 similar_objects/   # Umieść zdjęcia podobnych obiektów")  
    print("├── 🎯 other_objects/     # Umieść zdjęcia innych obiektów")
    print("└── 🌄 background/        # Umieść zdjęcia tła bez celu")
    
    print(f"\n🎯 INSTRUKCJA dla: {target_name}")
    print(f"1. Do '{target_name}/' dodaj 50-200 zdjęć Twojego celu z różnych kątów")
    print("2. Do 'similar_objects/' dodaj zdjęcia podobnych obiektów")
    print("3. Do 'other_objects/' dodaj zdjęcia różnych innych obiektów")
    print("4. Do 'background/' dodaj zdjęcia krajobrazów, tła bez celu")
    print("\n⏰ Następnie uruchom: python train_custom_detector.py")

if __name__ == "__main__":
    import sys
    target_name = sys.argv[1] if len(sys.argv) > 1 else "my_object"
    create_training_structure(target_name)