"""Initialize the src module and set up paths"""
import sys
import os

# Add the FlightDock root to path if not already there
flightdock_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if flightdock_root not in sys.path:
    sys.path.insert(0, flightdock_root)
