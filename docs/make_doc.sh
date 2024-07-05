#!/bin/bash
# Shell script to automatically generate the documentation with sphynx.

# First, delete the old documentation
folder_path="."

# List of files to keep
declare -a allowed_files=("conf.py" "Makefile" "make.bat" "index.rst" "notebooks.rst" "paper_scripts.rst" "make_doc.sh")

# Get the list of files from the current directory
files_in_folder=$(find "$folder_path" -maxdepth 1 -type f)

echo "Deleting old documentation source..."

# Loop through the files and delete the ones not in the allowed list
for file in $files_in_folder; do
    # Extract only the filename from the full path
    filename=$(basename "$file")

    # Check if the filename is in the allowed list
    if [[ ! " ${allowed_files[@]} " =~ " $filename " ]]; then
        # Not found in the allowed list, so delete the file
        echo "Deleting $file"
        rm "$file"
    fi
done

echo "Deletion process completed."

# Now, generate apidoc
cd ..; sphinx-apidoc  -o docs src/
cd docs


# Update "index.rst": this is only needed if index.rst has been deleted
#index_rst="$folder_path/index.rst"
#if [ -f "$index_rst" ]; then
#    # Search for the line "   :caption: Contents:"
#    line_number=$(grep -n "   :caption: Contents:" "$index_rst" | cut -d ":" -f 1)
#    if [ -n "$line_number" ]; then
#        # Add "   modules" two lines below the found line
#        insert_line=$((line_number + 2))
#        sed -i "${insert_line}i \ \  modules" "$index_rst"
#    fi
#fi
#echo "Added modules to index.rst"

# Finally, run make html
make html

echo "Documentation successfully updated!"
