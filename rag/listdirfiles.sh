#!/bin/zsh

setopt extended_glob null_glob

# Default exclude patterns for files
exclude_patterns=(
  "*.png"
  "*.jpg"
  "*.jpeg"
  "*.gif"
  "*.bmp"
  "*.svg"
  "*.ico"
  "*.pdf"
  "*.zip"
  "*.tar"
  "*.tar.gz"
  "*.mp3"
  "*.mp4"
  "*.avi"
  "*.mov"
  "*.mkv"
  "*.exe"
  "*.dll"
  "*.so"
  "*.bin"
  "*.obj"
  "*.class"
  "*.pyc"
  "*.DS_Store"
  "*.ttf"
  "*.woff"
  "*.woff2"
  "*.eot"
)

# Default exclude directories
exclude_dirs=(
  ".git"
  "node_modules"
  "dist"
  "build"
  "__pycache__"
  ".idea"
  ".vscode"
  ".pytest_cache"
  "venv"
  "env"
  ".env"
  "coverage"
  "target"
  ".gradle"
  ".settings"
  "vendor"
  "tmp"
  "logs"
  ".nyc_output"
  ".cache"
  ".next"
  "out"
)

# Function to display usage
function usage() {
  echo "Usage: $0 [-d directory]"
  exit 1
}

# Parse options
while getopts ":d:" opt; do
  case $opt in
    d)
      dir="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

shift $((OPTIND -1))

# Set directory to current if not specified
if [[ -z "$dir" ]]; then
  dir="."
fi

# Remove trailing slash from directory path
dir="${dir%/}"

# Build exclude patterns for files and directories
exclude_file_pattern="(${(j:|:)~exclude_patterns})"
exclude_dir_pattern="(${(j:|:)~exclude_dirs})"

# Function to recursively collect files
function collect_files() {
  local current_dir="$1"

  for entry in "$current_dir"/*(N); do
    # Skip if entry doesn't exist
    [[ -e "$entry" ]] || continue

    # Get relative path
    relative_entry="${entry#$dir/}"

    if [[ -d "$entry" ]]; then
      # Exclude directories matching the pattern
      for ex_dir in $exclude_dirs; do
        if [[ "${entry:t}" == "$ex_dir" ]]; then
          # Skip this directory
          continue 2
        fi
      done
      # Recursively collect files from subdirectories
      collect_files "$entry"
    elif [[ -f "$entry" ]]; then
      # Exclude files matching the pattern
      for ex_pattern in $exclude_patterns; do
        if [[ "$entry" == ${(~)ex_pattern} ]]; then
          # Skip this file
          continue 2
        fi
      done
      # Check if the file is a text file
      if file --mime-type "$entry" | grep -q 'text/'; then
        files+="$entry"
      fi
    fi
  done
}

# Initialize files array
typeset -a files
files=()

# Start collecting files
collect_files "$dir"

# Print content of files
for file in $files; do
  # Get relative path
  relative_file="${file#$dir/}"
  echo "Content of $relative_file:"
  cat "$file"
  echo ""
done
