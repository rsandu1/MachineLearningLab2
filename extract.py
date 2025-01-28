import os
import json

def extract_project_contents(directory):
    """
    Recursively extracts the contents of all files in the given directory.

    Args:
        directory (str): The path to the root directory of the project.

    Returns:
        dict: A dictionary containing file paths as keys and file contents as values.
    """
    project_data = {}

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Open and read the file's content
                with open(file_path, 'r', encoding='utf-8') as f:
                    project_data[file_path] = f.read()
            except (UnicodeDecodeError, PermissionError):
                # Skip binary files or files we don't have permission to read
                project_data[file_path] = "[NON-TEXT FILE OR UNREADABLE]"

    return project_data


if __name__ == "__main__":
    # Specify the directory to scan (current directory by default)
    project_directory = os.getcwd()
    
    # Extract project contents
    print(f"Extracting files from: {project_directory}")
    project_contents = extract_project_contents(project_directory)
    
    # Output file
    output_file = os.path.join(project_directory, "output.json")
    
    # Save the extracted data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(project_contents, json_file, indent=4)
    
    print(f"Project contents have been saved to: {output_file}")