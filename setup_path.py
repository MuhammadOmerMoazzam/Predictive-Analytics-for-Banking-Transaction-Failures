import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)

print(f"Project root added to Python path: {project_root}")
print(f"Available paths in sys.path: {sys.path[:3]}...")  # Show first 3 paths