from flask import Flask, render_template
import json
import os

app = Flask(__name__)

@app.route('/')
def view_report():
    # Construct the path to report.json, assuming it's in the parent directory of biaswipe_viewer
    # os.path.dirname(__file__) gives the directory of webserver.py (i.e., biaswipe_viewer)
    # os.path.join( ... , '..', 'report.json') goes one level up to the project root
    report_path = os.path.join(os.path.dirname(__file__), '..', 'report.json')

    report_data = None
    error_message = None

    try:
        with open(report_path, 'r') as f:
            report_data = json.load(f)
    except FileNotFoundError:
        error_message = f"Error: Report file not found at '{os.path.abspath(report_path)}'. Please generate it first using the BiasWipe CLI."
    except json.JSONDecodeError:
        error_message = f"Error: Could not decode JSON from '{os.path.abspath(report_path)}'. The file might be corrupted or not valid JSON."
    except Exception as e:
        error_message = f"An unexpected error occurred while trying to load the report: {str(e)}"

    return render_template('report_display.html', report_data=report_data, error_message=error_message)

if __name__ == '__main__':
    # Runs the Flask development server
    # Debug=True enables auto-reloading on code changes and provides a debugger
    # host='0.0.0.0' would make it accessible from network, default is 127.0.0.1 (localhost)
    app.run(debug=True)
