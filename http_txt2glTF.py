import asyncio
import hashlib
import json
import mimetypes
import os
import pathlib
import queue
import random
import ssl
import threading
from urllib.parse import urlparse, parse_qs

from http.server import HTTPServer, BaseHTTPRequestHandler

import websockets

from util import txt2mesh
from util import gltf_util

resultDir = None
requestQueue = queue.Queue(3)
resultIndex = {}


class MyHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed_path = urlparse(self.path)
        print(parsed_path)
        absolutePath = pathlib.Path(parsed_path.path).resolve()
        queries = parse_qs(parsed_path.query)
        try:
            if parsed_path.path == '/request':
                prompt = queries['prompt'][0]
                key = ''.join(random.choices('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', k=24))
                try:
                    requestQueue.put({
                            'prompt': prompt,
                            'key': key,
                        })
                except queue.Full:
                    print('[ERROR] do_GET(): /request: request queue is Full.')
                    key = None
                    pass
                if key is not None:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                            'code': 200,
                            'key': key,
                        }).encode())
                else:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                            'code': 500,
                            'message': 'Point-E is busy',
                        }).encode())
            elif parsed_path.path == '/getResult':
                key = queries['key'][0]
                filepath = resultIndex.get(key)
                if filepath is None:
                    filepath = ""
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', mimetypes.guess_type(filepath))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                            'code': 404,
                            'message': 'Not Found',
                        }).encode())
            else:
                ## GET specified path
                if str(absolutePath).startswith(resultDir) and os.path.exists(absolutePath):
                    ## Secure Contain Protect (inside resultDir)
                    with open(absolutePath, 'rb') as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', mimetypes.guess_type(absolutePath))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                            'code': 404,
                            'message': 'Not Found',
                        }).encode())
        except Exception as err:
            print('[ERROR] MyHTTPRequestHandler.do_GET(): ', end='')
            print(err)
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                    'code': 500,
                    'message': 'Internal Server Error',
                }).encode())

    def do_POST(self):
        parsed_path = urlparse(self.path)
        print(parsed_path)
        absolutePath = pathlib.Path(parsed_path.path).resolve()
        queries = parse_qs(parsed_path.query)
        content_length = int(self.headers['content-length'])
        content = self.rfile.read(content_length).decode('utf-8')
        try:
            if parsed_path.path == '/request':
                prompt = content
                key = ''.join(random.choices('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', k=24))
                try:
                    requestQueue.put({
                            'prompt': prompt,
                            'key': key,
                        })
                except queue.Full:
                    print('[ERROR] do_GET(): /request: request queue is Full.')
                    key = None
                    pass
                if key is not None:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(key.encode())
                else:
                    self.send_response(500)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write('Point-E is busy'.encode())
        except Exception as err:
            print('[ERROR] MyHTTPRequestHandler.do_POST(): ', end='')
            print(err)
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write('Internal Server Error'.encode())

def startHTTPServer(server_address):
    print('Start HTTP Server...')
    server = HTTPServer(server_address, MyHTTPRequestHandler)
    server.serve_forever()
    print('End HTTP Server.')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='Output Directory', type=str, default='./results')
    args = parser.parse_args()

    resultDir = pathlib.Path(args.output).resolve()
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    httpAddress = ('0.0.0.0', 8080)
    thread_httpServer = threading.Thread(
            target=startHTTPServer,
            args=(httpAddress,),
            daemon=True)
    thread_httpServer.start()

    # Initialize Point-E models
    point_e = txt2mesh.point_e_wrapper(guidance_scale=3.0)

    while True:
        request = None
        try:
            request = requestQueue.get(block=True)
            print(request)
        except Exception as err:
            print('[ERROR] Get Request: ', end='')
            print(err)

        try:
            print('Generate new 3D Model with the prompt "{}"...'.format(request['prompt']))
            pc = point_e.sample(request['prompt'])
            mesh = point_e.get_mesh(pc)
        except Exception as err:
            print('[ERROR] Generate Point Cloud and Convert to Mesh: ', end='')
            print(err)

        try:
            # Write the mesh to a PLY file to import into some other program.
            output_path = os.path.join(resultDir, '{}.glb'.format(request['key']))
            gltf_util.write_gltf(
                    output_path,
                    mesh)
            resultIndex[request['key']] = output_path
        except Exception as err:
            print('[ERROR] Save glTF: ', end='')
            print(err)

