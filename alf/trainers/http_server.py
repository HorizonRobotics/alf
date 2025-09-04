# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import http.server
import json
import numpy as np
import pprint
import socketserver
from typing import Callable
import urllib.parse
import base64

ROUTES = {}  # Dictionary to store endpoint-to-handler mapping
HELP_TEXT = {}  # Dictionary to store help text for each endpoint


class CustomRequestHandler(http.server.BaseHTTPRequestHandler):

    def get_int_argument(self, name: str, default: int):
        """Get an integer argument from the query parameters.

        Args:
            name: The name of the query parameter.
            default: The default value to return if the query parameter is not found.
        """
        value = self._query_params.get(name, [str(default)])[0]
        try:
            return int(value)
        except ValueError:
            return default

    def get_string_argument(self, name: str, default: str):
        """Get a string argument from the query parameters.

        Args:
            name: The name of the query parameter.
            default: The default value to return if the query parameter is not found.
        """
        return self._query_params.get(name, [default])[0]

    def get_float_argument(self, name: str, default: float):
        """Get a float argument from the query parameters.

        Args:
            name: The name of the query parameter.
            default: The default value to return if the query parameter is not found.
        """
        value = self._query_params.get(name, [str(default)])[0]
        try:
            return float(value)
        except ValueError:
            return default

    def get_list_argument(self, name: str, default: list):
        """Get a list argument from the query parameters.

        Args:
            name: The name of the query parameter.
            default: The default value to return if the query parameter is not found.
        """
        return self._query_params.get(name, default)

    def send_image(self, image: np.ndarray):
        """Send an image response.

        Args:
            image (np.ndarray): A numpy array representing the image.
                The image shape should be (height, width, 3) or (height, width).
                The color channel should be RGB.
        """
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode(".jpg", image)
        image_bytes = buffer.tobytes()

        # Send response headers
        try:
            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.send_header("Content-Length", str(len(image_bytes)))
            self.end_headers()

            # Send image data
            self.wfile.write(image_bytes)
        except BrokenPipeError:
            # Ignore broken pipe error
            pass

    def send_json(self, data: dict):
        """Send JSON response.

        Args:
            data: A dictionary to be converted to JSON and sent as response
        """
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def send_html(self, html: str, response_code=200):
        """Send HTML response.

        For example:
        html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Simple HTTP Server</title>
            </head>
            <body>
                <h1>Welcome to My Simple HTTP Server</h1>
                <p>This is a basic HTML page served by Python.</p>
                <img src="/image" alt="Example Image" width="300">
            </body>
            </html>
        '''

        Args:
            html: The HTML content to be sent as response.
            response_code: The HTTP response code.
        """
        self.send_response(response_code)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def send_text(self, text: str, response_code: int = 200):
        """Send text response.

        Args:
            text: The text to be sent as response.
            response_code: The HTTP response code.
        """
        self.send_response(response_code)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        self._query_params = urllib.parse.parse_qs(parsed_url.query)
        if parsed_url.path in ROUTES:
            handler = ROUTES[parsed_url.path]
            handler(self)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")


def register_endpoint(path: str,
                      handler: Callable[[CustomRequestHandler], None],
                      help_text: str = ""):
    """Registers a new endpoint with a custom handler function.

    Args:
        path: The endpoint path. Note that path should start with '/'.
        handler: The handler function that will be called when the endpoint is
            accessed.
        help_text: The help text that will be displayed when the user accesses
            the home page.
    """
    ROUTES[path] = handler
    HELP_TEXT[path] = help_text


def start_server(port):
    with socketserver.TCPServer(("", port), CustomRequestHandler) as httpd:
        print(f"Serving at port {port}")
        httpd.serve_forever()


def home_handler(request):
    # respond with all the available endpoints and their help text
    request.send_text(pprint.pformat(HELP_TEXT))


def render_handler(request):
    """Handle /render endpoint by getting environment render image."""
    try:
        import alf
        env = alf.get_env()
        image = env.render()

        if image is None:
            request.send_html(
                "<html><body><h1>No image available</h1></body></html>")
            return

        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            _, buffer = cv2.imencode(".jpg", image_rgb)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            html = f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Environment Render</title>
            </head>
            <body>
                <h1>Environment Render</h1>
                <img src="data:image/jpeg;base64,{image_base64}" alt="Environment Render" style="max-width: 100%; height: auto;">
            </body>
            </html>
            '''
            request.send_html(html)
        else:
            request.send_html(
                "<html><body><h1>Invalid image format</h1></body></html>")
    except Exception as e:
        error_html = f"<html><body><h1>Error rendering environment</h1><p>{str(e)}</p></body></html>"
        request.send_html(error_html, 500)


# Registering endpoints
register_endpoint("/", home_handler)
register_endpoint("/render", render_handler, "Get environment render image")
