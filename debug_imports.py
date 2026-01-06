import sys
import os

print("Current working directory:", os.getcwd())
print("Current Python path:")
for i, path in enumerate(sys.path[:10]):  # Print first 10 paths
    print(f"  {i}: {path}")

print("\nLooking for 'src' directory...")
src_path = os.path.join(os.getcwd(), 'src')
if os.path.exists(src_path):
    print(f"SUCCESS: Found 'src' directory at: {src_path}")
    print("  Contents:", os.listdir(src_path))
else:
    print("ERROR: 'src' directory not found in current directory")

# Try to add the current directory to the path explicitly
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
    print(f"\nAdded current directory to Python path: {os.getcwd()}")

# Check again
print("\nAfter adding current directory to path:")
try:
    import src
    print("SUCCESS: Successfully imported 'src' module")
    print(f"  Location: {src.__path__ if hasattr(src, '__path__') else 'N/A'}")
except ImportError as e:
    print(f"ERROR: Still can't import 'src': {e}")

# Try importing specific modules
try:
    from src.simulator import transaction_simulator
    print("SUCCESS: Successfully imported 'src.simulator'")
except ImportError as e:
    print(f"ERROR: Can't import 'src.simulator': {e}")

try:
    from src.simulator.transaction_simulator import TransactionSimulator
    print("SUCCESS: Successfully imported 'TransactionSimulator'")
except ImportError as e:
    print(f"ERROR: Can't import 'TransactionSimulator': {e}")