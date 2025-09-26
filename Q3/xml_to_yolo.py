import os
import xml.etree.ElementTree as ET

# --- CONFIGURATION ---
# IMPORTANT: Adjust this path to the single directory containing both your images and XML files.
BASE_DIR = 'C:\\ALL\\FindWork\\Dhvani Hackathon Mohammed Salim\\TrafficDataset\\train\\Final Train Dataset'
# Define the output directory for the new YOLO-format files
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'labels')

def get_all_classes(base_dir):
    """
    Scans all XML files in the directory to find and return a set of all unique class names.
    """
    all_classes = set()
    for file in os.listdir(base_dir):
        if file.endswith('.xml'):
            xml_path = os.path.join(base_dir, file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text.lower()
                    all_classes.add(name)
            except ET.ParseError as e:
                print(f"Error parsing XML file {xml_path}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred with file {xml_path}: {e}")
    return sorted(list(all_classes))

def convert_xml_to_yolo(xml_filepath, class_to_id):
    """
    Parses a single XML file and converts its annotations to YOLO format
    using the provided class-to-ID mapping.
    """
    try:
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        # Get image dimensions from the XML
        size = root.find('size')
        if size is None:
            print(f"Warning: <size> tag not found in {xml_filepath}. Skipping.")
            return None
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        yolo_annotations = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            if name in class_to_id:
                class_id = class_to_id[name]

                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))

                # Calculate normalized YOLO coordinates
                center_x = (xmin + xmax) / 2.0 / img_width
                center_y = (ymin + ymax) / 2.0 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Check if coordinates are within a valid range
                if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    print(f"Warning: Invalid coordinates found in {xml_filepath} for object '{name}'. Skipping.")
                    continue

                yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with file {xml_filepath}: {e}")
        return None

def main():
    """Main function to process all XML files in the directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Automatically discover all classes
    print("Scanning XML files to discover all unique classes...")
    all_classes = get_all_classes(BASE_DIR)
    
    if not all_classes:
        print("No classes found in the XML files. Please check your dataset.")
        return

    CLASS_TO_ID = {name: i for i, name in enumerate(all_classes)}
    
    print("\nDiscovered classes:")
    for i, cls in enumerate(all_classes):
        print(f"  - {i}: {cls}")

    # Step 2: Save the classes to a 'classes.txt' file for YOLO
    classes_file_path = os.path.join(OUTPUT_DIR, 'classes.txt')
    with open(classes_file_path, 'w') as f:
        for cls in all_classes:
            f.write(cls + '\n')
    print(f"\nSaved class list to '{classes_file_path}'.")

    # Step 3: Convert XML annotations to YOLO format
    print(f"\nStarting conversion of XML files in '{BASE_DIR}'...")
    for file in os.listdir(BASE_DIR):
        if file.endswith('.xml'):
            xml_path = os.path.join(BASE_DIR, file)
            yolo_annotations = convert_xml_to_yolo(xml_path, CLASS_TO_ID)

            if yolo_annotations:
                # Create the corresponding .txt file
                txt_filename = os.path.splitext(file)[0] + '.txt'
                txt_path = os.path.join(OUTPUT_DIR, txt_filename)
                
                with open(txt_path, 'w') as f:
                    for line in yolo_annotations:
                        f.write(line + '\n')
                
                print(f"Successfully converted '{file}' to '{txt_filename}'")
    
    print("\nConversion complete.")
    print(f"The YOLO-formatted labels are now in the '{OUTPUT_DIR}' directory.")
    print("You can now use these files to train your YOLO model.")
    print("\nNote: The `images` directory for training should also be `C:\\ALL\\FindWork\\Dhvani Hackathon Mohammed Salim\\TrafficDataset\\train\\Final Train Dataset`.")

if __name__ == '__main__':
    main()
