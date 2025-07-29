#!/usr/bin/env python3
"""
Run the domain evolution flow dashboard.

Usage:
    python run_dashboard.py
"""

if __name__ == "__main__":
    from visualization.dashboard.app import app
    app.run_server(debug=True)