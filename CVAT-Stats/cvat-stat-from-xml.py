import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import pandas as pd

def export_experiment_stats_to_excel(xml_file_path, output_excel_path='experiment_stats.xlsx'):
    # Parse the CVAT XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Dictionary to dynamically hold counts for any folder it encounters
    stats = defaultdict(lambda: {'frames': 0, 'annotations': 0})
    
    # Iterate through every frame in the XML
    for image in root.findall('image'):
        # CVAT stores the relative path in the 'name' attribute 
        file_path = image.get('name')
        
        # Extract the directory path natively based on the hierarchy
        folder_path = str(Path(file_path).parent)
        
        # Count the frame
        stats[folder_path]['frames'] += 1
        
        # Count all geometric shapes inside this frame
        annotation_count = 0
        for child in image:
            if child.tag in ['box', 'polygon', 'polyline', 'points', 'cuboid', 'ellipse']:
                annotation_count += 1
                
        stats[folder_path]['annotations'] += annotation_count

    # Convert the stats dictionary into a list of rows for the spreadsheet
    table_data = []
    for folder, counts in sorted(stats.items()):
        table_data.append({
            'Directory Hierarchy': folder,
            'Total Frames': counts['frames'],
            'Total Annotations': counts['annotations']
        })
        
    # Create a pandas DataFrame (the table structure)
    df = pd.DataFrame(table_data)
    
    # Export the table to an Excel file, removing the default row numbers (index=False)
    df.to_excel(output_excel_path, index=False)
    
    print(f"Success! Exported {len(df)} folders to {output_excel_path}")

# --- How to run it ---
# Pass the path of your XML file, and optionally, what you want the Excel file to be named.
# export_experiment_stats_to_excel('annotations.xml', 'annotation_report.xlsx')

export_experiment_stats_to_excel('/home/emre/Desktop/NATO/code/CVAT-Stats/cvat-for-img-final/annotations.xml')