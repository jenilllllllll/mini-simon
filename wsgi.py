"""
WSGI Configuration for PythonAnywhere
This file serves as the entry point for the PythonAnywhere web app.

Place this file at: /home/YOUR_USERNAME/mini-simon/wsgi.py
And configure your PythonAnywhere web app to point to this file.
"""

import sys
import os

# Add your project directory to the path
# Replace 'YOUR_USERNAME' with your actual PythonAnywhere username
# The path should match where you cloned the repository
PROJECT_HOME = os.path.dirname(os.path.abspath(__file__))
if PROJECT_HOME not in sys.path:
    sys.path.insert(0, PROJECT_HOME)

# Set environment variables for production
os.environ['PYTHONANYWHERE_DOMAIN'] = os.getenv('PYTHONANYWHERE_DOMAIN', 'pythonanywhere.com')
os.environ['PYTHONANYWHERE_SITE'] = os.getenv('PYTHONANYWHERE_SITE', 'mini-simon')

# Import the FastAPI app from web_main
from web_main import app as application

# For PythonAnywhere's WSGI compatibility, we need to wrap the FastAPI app
# PythonAnywhere uses a WSGI interface, so we use WSGIMiddleware
from starlette.middleware.wsgi import WSGIMiddleware
from fastapi import FastAPI

# If running on PythonAnywhere, wrap FastAPI with WSGIMiddleware
# This allows FastAPI to work with PythonAnywhere's WSGI server
if os.getenv('PYTHONANYWHERE_DOMAIN'):
    # Create a WSGI-compatible application
    application = WSGIMiddleware(application)
