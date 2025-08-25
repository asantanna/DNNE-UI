import os
import sys
import asyncio
import traceback
import time
from datetime import datetime
from pathlib import Path

import nodes
import folder_paths
import execution
import uuid
import urllib
import json
import glob
import struct
import ssl
import socket
import ipaddress
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from io import BytesIO

import aiohttp
from aiohttp import web
import logging

import mimetypes
from comfy.cli_args import args
import comfy.utils
import comfy.model_management
import node_helpers
from comfyui_version import __version__
from app.frontend_management import FrontendManager

from app.user_manager import UserManager
from app.model_manager import ModelFileManager
from app.custom_node_manager import CustomNodeManager
from typing import Optional, Union
from api_server.routes.internal.internal_routes import InternalRoutes

class BinaryEventTypes:
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2
    TEXT = 3

async def send_socket_catch_exception(function, message):
    try:
        await function(message)
    except (aiohttp.ClientError, aiohttp.ClientPayloadError, ConnectionResetError, BrokenPipeError, ConnectionError) as err:
        logging.warning("send error: {}".format(err))

@web.middleware
async def cache_control(request: web.Request, handler):
    response: web.Response = await handler(request)
    if request.path.endswith('.js') or request.path.endswith('.css') or request.path.endswith('index.json'):
        response.headers.setdefault('Cache-Control', 'no-cache')
    return response


@web.middleware
async def compress_body(request: web.Request, handler):
    accept_encoding = request.headers.get("Accept-Encoding", "")
    response: web.Response = await handler(request)
    if not isinstance(response, web.Response):
        return response
    if response.content_type not in ["application/json", "text/plain"]:
        return response
    if response.body and "gzip" in accept_encoding:
        response.enable_compression()
    return response


def create_cors_middleware(allowed_origin: str):
    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            # Pre-flight request. Reply successfully:
            response = web.Response()
        else:
            response = await handler(request)

        response.headers['Access-Control-Allow-Origin'] = allowed_origin
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, PUT, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    return cors_middleware

def is_loopback(host):
    if host is None:
        return False
    try:
        if ipaddress.ip_address(host).is_loopback:
            return True
        else:
            return False
    except ValueError:
        # Not a valid IP address, will try hostname resolution below
        pass

    loopback = False
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            r = socket.getaddrinfo(host, None, family, socket.SOCK_STREAM)
            for family, _, _, _, sockaddr in r:
                if not ipaddress.ip_address(sockaddr[0]).is_loopback:
                    return loopback
                else:
                    loopback = True
        except socket.gaierror:
            # Unable to resolve hostname, continue checking other address families
            pass

    return loopback


def create_origin_only_middleware():
    @web.middleware
    async def origin_only_middleware(request: web.Request, handler):
        #this code is used to prevent the case where a random website can queue comfy workflows by making a POST to 127.0.0.1 which browsers don't prevent for some dumb reason.
        #in that case the Host and Origin hostnames won't match
        #I know the proper fix would be to add a cookie but this should take care of the problem in the meantime
        if 'Host' in request.headers and 'Origin' in request.headers:
            host = request.headers['Host']
            origin = request.headers['Origin']
            host_domain = host.lower()
            parsed = urllib.parse.urlparse(origin)
            origin_domain = parsed.netloc.lower()
            host_domain_parsed = urllib.parse.urlsplit('//' + host_domain)

            #limit the check to when the host domain is localhost, this makes it slightly less safe but should still prevent the exploit
            loopback = is_loopback(host_domain_parsed.hostname)

            if parsed.port is None: #if origin doesn't have a port strip it from the host to handle weird browsers, same for host
                host_domain = host_domain_parsed.hostname
            if host_domain_parsed.port is None:
                origin_domain = parsed.hostname

            if loopback and host_domain is not None and origin_domain is not None and len(host_domain) > 0 and len(origin_domain) > 0:
                if host_domain != origin_domain:
                    logging.warning("WARNING: request with non matching host and origin {} != {}, returning 403".format(host_domain, origin_domain))
                    return web.Response(status=403)

        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)

        return response

    return origin_only_middleware

class PromptServer():
    def __init__(self, loop):
        PromptServer.instance = self

        mimetypes.init()
        mimetypes.add_type('application/javascript; charset=utf-8', '.js')
        mimetypes.add_type('image/webp', '.webp')

        self.user_manager = UserManager()
        self.model_file_manager = ModelFileManager()
        self.custom_node_manager = CustomNodeManager()
        self.internal_routes = InternalRoutes(self)
        self.supports = ["custom_nodes_from_web"]
        self.prompt_queue = execution.PromptQueue(self)
        self.loop = loop
        self.messages = asyncio.Queue()
        self.client_session:Optional[aiohttp.ClientSession] = None
        self.number = 0
        self.current_workflow_name = None  # Track currently loaded workflow
        self.start_time = time.time()  # Track server start time for uptime

        middlewares = [cache_control]
        if args.enable_compress_response_body:
            middlewares.append(compress_body)

        if args.enable_cors_header:
            middlewares.append(create_cors_middleware(args.enable_cors_header))
        else:
            middlewares.append(create_origin_only_middleware())

        max_upload_size = round(args.max_upload_size * 1024 * 1024)
        self.app = web.Application(client_max_size=max_upload_size, middlewares=middlewares)
        self.sockets = dict()
        self.web_root = (
            FrontendManager.init_frontend(args.front_end_version)
            if args.front_end_root is None
            else args.front_end_root
        )
        logging.info(f"[Prompt Server] web root: {self.web_root}")
        routes = web.RouteTableDef()
        self.routes = routes
        self.last_node_id = None
        self.client_id = None

        self.on_prompt_handlers = []
        
        # Agent server connection
        self.agent_ws = None
        self.agent_connected = False
        self.agent_clients = {}  # Cache of connected clients
        self.agent_connection_status = "disconnected"
        self.agent_reconnect_task = None
        # Combined workflow tracking - single source of truth
        self.active_workflows = {}  # {workflow_id: {file_handle, sequence, name, client_id, client_hostname, start_time}}

        @routes.get('/ws')
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            sid = request.rel_url.query.get('clientId', '')
            if sid:
                # Reusing existing session, remove old
                self.sockets.pop(sid, None)
            else:
                sid = uuid.uuid4().hex

            self.sockets[sid] = ws

            try:
                # Send initial state to the new client
                await self.send("status", { "status": self.get_queue_info(), 'sid': sid }, sid)
                # On reconnect if we are the currently executing client send the current node
                if self.client_id == sid and self.last_node_id is not None:
                    await self.send("executing", { "node": self.last_node_id }, sid)

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        # Handle incoming messages from frontend
                        try:
                            data = json.loads(msg.data)
                            msg_type = data.get('type')
                            
                            if msg_type == 'request_logs':
                                # Handle request for historical logs
                                workflow_id = data.get('workflow_id')
                                client_id = data.get('client_id')  # For requesting latest logs when no workflow_id
                                logging.debug(f"🔍 Received request_logs for workflow_id: {workflow_id}, client_id: {client_id}")
                                await self.send_workflow_history(ws, workflow_id, client_id)
                            elif msg_type == 'request_telemetry':
                                # Handle request for telemetry data
                                workflow_id = data.get('workflow_id')
                                client_id = data.get('client_id')
                                telemetry_type = data.get('telemetry_type', 'violations')  # 'violations' or 'data'
                                logging.debug(f"📊 Received request_telemetry for workflow_id: {workflow_id}, type: {telemetry_type}")
                                await self.send_telemetry_history(ws, workflow_id, client_id, telemetry_type)
                            elif msg_type == 'check_workflow_exists':
                                # Check if workflow exists on remote client
                                client_id = data.get('client_id')
                                workflow_id = data.get('workflow_id')
                                logging.debug(f"🔍 Checking if workflow {workflow_id} exists on {client_id}")
                                await self.check_workflow_exists(ws, client_id, workflow_id)
                            elif msg_type == 'run_workflow':
                                # Run an existing workflow on remote client
                                client_id = data.get('client_id')
                                workflow_id = data.get('workflow_id')
                                runner_args = data.get('runner_args', '')
                                logging.debug(f"▶️ Running workflow {workflow_id} on {client_id}")
                                await self.run_existing_workflow(ws, client_id, workflow_id, runner_args)
                            elif msg_type == 'widget_callback':
                                # Handle widget callback from frontend
                                widget_id = data.get('widget_id')
                                event = data.get('event')
                                event_params = data.get('event_params', {})
                                
                                logging.debug(f"🔄 Widget callback received: {widget_id} - {event}")
                                logging.debug(f"    Event params: {event_params}")
                                
                                # Route to appropriate node handler
                                if widget_id:
                                    node_type = widget_id.split('.')[0] if '.' in widget_id else None
                                    
                                    # Import node handlers dynamically
                                    response = None
                                    try:
                                        # Check if it's IsaacGymEnvsNode
                                        if node_type == 'IsaacGymEnvsNode':
                                            logging.debug(f"    Routing to IsaacGymEnvsNode handler")
                                            from custom_nodes.isaac_gym_envs_visnode import IsaacGymEnvsNode
                                            if hasattr(IsaacGymEnvsNode, 'handle_widget_callback'):
                                                response = await IsaacGymEnvsNode.handle_widget_callback(data)
                                                logging.debug(f"    Handler returned response: {response.get('type') if response else None}")
                                        
                                        # Send response if we got one
                                        if response:
                                            logging.debug(f"    Sending response to frontend: {response['type']}")
                                            # Wrap the response data properly for the frontend
                                            await ws.send_json({
                                                'type': 'widget_callback_response',
                                                'data': {
                                                    'widget_id': response.get('widget_id'),
                                                    'code_payload': response.get('code_payload'),
                                                    'chain': response.get('chain', True)
                                                }
                                            })
                                        else:
                                            # Send default response
                                            logging.debug(f"    Sending default response")
                                            await ws.send_json({
                                                'type': 'widget_callback_response',
                                                'data': {
                                                    'widget_id': widget_id,
                                                    'chain': True
                                                }
                                            })
                                    except Exception as e:
                                        logging.error(f"Error handling widget callback: {e}")
                                        await ws.send_json({
                                            'type': 'widget_callback_response',
                                            'data': {
                                                'widget_id': widget_id,
                                                'error': str(e),
                                                'chain': True
                                            }
                                        })
                            elif msg_type == 'request_runner_args':
                                # Handle request for runner arguments configuration
                                import os
                                runner_args_path = os.path.join(
                                    os.path.dirname(__file__), 
                                    'export_system/templates/framework/runner_args.json'
                                )
                                try:
                                    with open(runner_args_path, 'r') as f:
                                        runner_args = json.load(f)
                                    
                                    await ws.send_json({
                                        'type': 'runner_args',
                                        'data': runner_args
                                    })
                                    logging.debug("Sent runner_args configuration to frontend")
                                except FileNotFoundError:
                                    logging.error(f"runner_args.json not found at {runner_args_path}")
                                    await ws.send_json({
                                        'type': 'error',
                                        'message': 'Runner arguments configuration not found'
                                    })
                                except Exception as e:
                                    logging.error(f"Error loading runner_args.json: {e}")
                                    await ws.send_json({
                                        'type': 'error',
                                        'message': f'Error loading runner arguments: {str(e)}'
                                    })
                        except json.JSONDecodeError:
                            logging.warning(f'Invalid JSON from client: {msg.data}')
                        except Exception as e:
                            logging.error(f'Error handling client message: {e}')
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logging.warning('ws connection closed with exception %s' % ws.exception())
            finally:
                self.sockets.pop(sid, None)
            return ws

        @routes.get("/")
        async def get_root(request):
            response = web.FileResponse(os.path.join(self.web_root, "index.html"))
            response.headers['Cache-Control'] = 'no-cache'
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        @routes.get("/embeddings")
        def get_embeddings(request):
            embeddings = folder_paths.get_filename_list("embeddings")
            return web.json_response(list(map(lambda a: os.path.splitext(a)[0], embeddings)))

        @routes.get("/models")
        def list_model_types(request):
            model_types = list(folder_paths.folder_names_and_paths.keys())

            return web.json_response(model_types)

        @routes.get("/models/{folder}")
        async def get_models(request):
            folder = request.match_info.get("folder", None)
            if not folder in folder_paths.folder_names_and_paths:
                return web.Response(status=404)
            files = folder_paths.get_filename_list(folder)
            return web.json_response(files)

        @routes.get("/extensions")
        async def get_extensions(request):
            files = glob.glob(os.path.join(
                glob.escape(self.web_root), 'extensions/**/*.js'), recursive=True)

            extensions = list(map(lambda f: "/" + os.path.relpath(f, self.web_root).replace("\\", "/"), files))

            for name, dir in nodes.EXTENSION_WEB_DIRS.items():
                files = glob.glob(os.path.join(glob.escape(dir), '**/*.js'), recursive=True)
                extensions.extend(list(map(lambda f: "/extensions/" + urllib.parse.quote(
                    name) + "/" + os.path.relpath(f, dir).replace("\\", "/"), files)))

            return web.json_response(extensions)

        def get_dir_by_type(dir_type):
            if dir_type is None:
                dir_type = "input"

            if dir_type == "input":
                type_dir = folder_paths.get_input_directory()
            elif dir_type == "temp":
                type_dir = folder_paths.get_temp_directory()
            elif dir_type == "output":
                type_dir = folder_paths.get_output_directory()

            return type_dir, dir_type

        def compare_image_hash(filepath, image):
            hasher = node_helpers.hasher()

            # function to compare hashes of two images to see if it already exists, fix to #3465
            if os.path.exists(filepath):
                a = hasher()
                b = hasher()
                with open(filepath, "rb") as f:
                    a.update(f.read())
                    b.update(image.file.read())
                    image.file.seek(0)
                return a.hexdigest() == b.hexdigest()
            return False

        def image_upload(post, image_save_function=None):
            image = post.get("image")
            overwrite = post.get("overwrite")
            image_is_duplicate = False

            image_upload_type = post.get("type")
            upload_dir, image_upload_type = get_dir_by_type(image_upload_type)

            if image and image.file:
                filename = image.filename
                if not filename:
                    return web.Response(status=400)

                subfolder = post.get("subfolder", "")
                full_output_folder = os.path.join(upload_dir, os.path.normpath(subfolder))
                filepath = os.path.abspath(os.path.join(full_output_folder, filename))

                if os.path.commonpath((upload_dir, filepath)) != upload_dir:
                    return web.Response(status=400)

                if not os.path.exists(full_output_folder):
                    os.makedirs(full_output_folder)

                split = os.path.splitext(filename)

                if overwrite is not None and (overwrite == "true" or overwrite == "1"):
                    pass
                else:
                    i = 1
                    while os.path.exists(filepath):
                        if compare_image_hash(filepath, image): #compare hash to prevent saving of duplicates with same name, fix for #3465
                            image_is_duplicate = True
                            break
                        filename = f"{split[0]} ({i}){split[1]}"
                        filepath = os.path.join(full_output_folder, filename)
                        i += 1

                if not image_is_duplicate:
                    if image_save_function is not None:
                        image_save_function(image, post, filepath)
                    else:
                        with open(filepath, "wb") as f:
                            f.write(image.file.read())

                return web.json_response({"name" : filename, "subfolder": subfolder, "type": image_upload_type})
            else:
                return web.Response(status=400)

        @routes.post("/upload/image")
        async def upload_image(request):
            post = await request.post()
            return image_upload(post)


        @routes.post("/upload/mask")
        async def upload_mask(request):
            post = await request.post()

            def image_save_function(image, post, filepath):
                original_ref = json.loads(post.get("original_ref"))
                filename, output_dir = folder_paths.annotated_filepath(original_ref['filename'])

                if not filename:
                    return web.Response(status=400)

                # validation for security: prevent accessing arbitrary path
                if filename[0] == '/' or '..' in filename:
                    return web.Response(status=400)

                if output_dir is None:
                    type = original_ref.get("type", "output")
                    output_dir = folder_paths.get_directory_by_type(type)

                if output_dir is None:
                    return web.Response(status=400)

                if original_ref.get("subfolder", "") != "":
                    full_output_dir = os.path.join(output_dir, original_ref["subfolder"])
                    if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                        return web.Response(status=403)
                    output_dir = full_output_dir

                file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    with Image.open(file) as original_pil:
                        metadata = PngInfo()
                        if hasattr(original_pil,'text'):
                            for key in original_pil.text:
                                metadata.add_text(key, original_pil.text[key])
                        original_pil = original_pil.convert('RGBA')
                        mask_pil = Image.open(image.file).convert('RGBA')

                        # alpha copy
                        new_alpha = mask_pil.getchannel('A')
                        original_pil.putalpha(new_alpha)
                        original_pil.save(filepath, compress_level=4, pnginfo=metadata)

            return image_upload(post, image_save_function)

        @routes.get("/view")
        async def view_image(request):
            if "filename" in request.rel_url.query:
                filename = request.rel_url.query["filename"]
                filename,output_dir = folder_paths.annotated_filepath(filename)

                if not filename:
                    return web.Response(status=400)

                # validation for security: prevent accessing arbitrary path
                if filename[0] == '/' or '..' in filename:
                    return web.Response(status=400)

                if output_dir is None:
                    type = request.rel_url.query.get("type", "output")
                    output_dir = folder_paths.get_directory_by_type(type)

                if output_dir is None:
                    return web.Response(status=400)

                if "subfolder" in request.rel_url.query:
                    full_output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])
                    if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                        return web.Response(status=403)
                    output_dir = full_output_dir

                filename = os.path.basename(filename)
                file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    if 'preview' in request.rel_url.query:
                        with Image.open(file) as img:
                            preview_info = request.rel_url.query['preview'].split(';')
                            image_format = preview_info[0]
                            if image_format not in ['webp', 'jpeg'] or 'a' in request.rel_url.query.get('channel', ''):
                                image_format = 'webp'

                            quality = 90
                            if preview_info[-1].isdigit():
                                quality = int(preview_info[-1])

                            buffer = BytesIO()
                            if image_format in ['jpeg'] or request.rel_url.query.get('channel', '') == 'rgb':
                                img = img.convert("RGB")
                            img.save(buffer, format=image_format, quality=quality)
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type=f'image/{image_format}',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    if 'channel' not in request.rel_url.query:
                        channel = 'rgba'
                    else:
                        channel = request.rel_url.query["channel"]

                    if channel == 'rgb':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                r, g, b, a = img.split()
                                new_img = Image.merge('RGB', (r, g, b))
                            else:
                                new_img = img.convert("RGB")

                            buffer = BytesIO()
                            new_img.save(buffer, format='PNG')
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    elif channel == 'a':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                _, _, _, a = img.split()
                            else:
                                a = Image.new('L', img.size, 255)

                            # alpha img
                            alpha_img = Image.new('RGBA', img.size)
                            alpha_img.putalpha(a)
                            alpha_buffer = BytesIO()
                            alpha_img.save(alpha_buffer, format='PNG')
                            alpha_buffer.seek(0)

                            return web.Response(body=alpha_buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})
                    else:
                        # Get content type from mimetype, defaulting to 'application/octet-stream'
                        content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'

                        # For security, force certain extensions to download instead of display
                        file_extension = os.path.splitext(filename)[1].lower()
                        if file_extension in {'.html', '.htm', '.js', '.css'}:
                            content_type = 'application/octet-stream'  # Forces download

                        return web.FileResponse(
                            file,
                            headers={
                                "Content-Disposition": f"filename=\"{filename}\"",
                                "Content-Type": content_type
                            }
                        )

            return web.Response(status=404)

        @routes.get("/view_metadata/{folder_name}")
        async def view_metadata(request):
            folder_name = request.match_info.get("folder_name", None)
            if folder_name is None:
                return web.Response(status=404)
            if not "filename" in request.rel_url.query:
                return web.Response(status=404)

            filename = request.rel_url.query["filename"]
            if not filename.endswith(".safetensors"):
                return web.Response(status=404)

            safetensors_path = folder_paths.get_full_path(folder_name, filename)
            if safetensors_path is None:
                return web.Response(status=404)
            out = comfy.utils.safetensors_header(safetensors_path, max_size=1024*1024)
            if out is None:
                return web.Response(status=404)
            dt = json.loads(out)
            if not "__metadata__" in dt:
                return web.Response(status=404)
            return web.json_response(dt["__metadata__"])

        @routes.get("/system_stats")
        async def system_stats(request):
            device = comfy.model_management.get_torch_device()
            device_name = comfy.model_management.get_torch_device_name(device)
            cpu_device = comfy.model_management.torch.device("cpu")
            ram_total = comfy.model_management.get_total_memory(cpu_device)
            ram_free = comfy.model_management.get_free_memory(cpu_device)
            vram_total, torch_vram_total = comfy.model_management.get_total_memory(device, torch_total_too=True)
            vram_free, torch_vram_free = comfy.model_management.get_free_memory(device, torch_free_too=True)

            system_stats = {
                "system": {
                    "os": os.name,
                    "ram_total": ram_total,
                    "ram_free": ram_free,
                    "comfyui_version": __version__,
                    "python_version": sys.version,
                    "pytorch_version": comfy.model_management.torch_version,
                    "embedded_python": os.path.split(os.path.split(sys.executable)[0])[1] == "python_embeded",
                    "argv": sys.argv
                },
                "devices": [
                    {
                        "name": device_name,
                        "type": device.type,
                        "index": device.index,
                        "vram_total": vram_total,
                        "vram_free": vram_free,
                        "torch_vram_total": torch_vram_total,
                        "torch_vram_free": torch_vram_free,
                    }
                ]
            }
            return web.json_response(system_stats)

        @routes.get("/health")
        async def health_check(request):
            """Health check endpoint for uniform server health monitoring."""
            import time
            
            # Calculate uptime
            uptime = time.time() - self.start_time if hasattr(self, 'start_time') else 0
            
            # Check agent server connection
            agent_connected = hasattr(self, 'agent_ws') and self.agent_ws and not self.agent_ws.closed
            
            health_status = {
                "status": "healthy",
                "uptime": uptime,
                "version": __version__,  # This is already imported at the top of the file
                "agent_connected": agent_connected,
                "agent_clients": len(self.agent_clients) if hasattr(self, 'agent_clients') else 0,
                "active_workflows": len(self.active_workflows) if hasattr(self, 'active_workflows') else 0
            }
            
            return web.json_response(health_status)

        @routes.get("/prompt")
        async def get_prompt(request):
            return web.json_response(self.get_queue_info())

        def node_info(node_class):
            obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
            info = {}
            info['input'] = obj_class.INPUT_TYPES()
            info['input_order'] = {key: list(value.keys()) for (key, value) in obj_class.INPUT_TYPES().items()}
            info['output'] = obj_class.RETURN_TYPES
            info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [False] * len(obj_class.RETURN_TYPES)
            info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
            info['name'] = node_class
            info['display_name'] = nodes.NODE_DISPLAY_NAME_MAPPINGS[node_class] if node_class in nodes.NODE_DISPLAY_NAME_MAPPINGS.keys() else node_class
            info['description'] = obj_class.DESCRIPTION if hasattr(obj_class,'DESCRIPTION') else ''
            info['python_module'] = getattr(obj_class, "RELATIVE_PYTHON_MODULE", "nodes")
            info['category'] = 'sd'
            if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE == True:
                info['output_node'] = True
            else:
                info['output_node'] = False

            if hasattr(obj_class, 'CATEGORY'):
                info['category'] = obj_class.CATEGORY

            if hasattr(obj_class, 'OUTPUT_TOOLTIPS'):
                info['output_tooltips'] = obj_class.OUTPUT_TOOLTIPS

            if getattr(obj_class, "DEPRECATED", False):
                info['deprecated'] = True
            if getattr(obj_class, "EXPERIMENTAL", False):
                info['experimental'] = True

            if hasattr(obj_class, 'API_NODE'):
                info['api_node'] = obj_class.API_NODE
            
            # Include COLOR and BGCOLOR if defined on the node class
            if hasattr(obj_class, 'COLOR'):
                info['color'] = obj_class.COLOR
            if hasattr(obj_class, 'BGCOLOR'):
                info['bgcolor'] = obj_class.BGCOLOR
                
            return info

        @routes.get("/object_info")
        async def get_object_info(request):
            with folder_paths.cache_helper:
                out = {}
                for x in nodes.NODE_CLASS_MAPPINGS:
                    try:
                        out[x] = node_info(x)
                    except Exception:
                        logging.error(f"[ERROR] An error occurred while retrieving information for the '{x}' node.")
                        logging.error(traceback.format_exc())
                return web.json_response(out)

        @routes.get("/object_info/{node_class}")
        async def get_object_info_node(request):
            node_class = request.match_info.get("node_class", None)
            out = {}
            if (node_class is not None) and (node_class in nodes.NODE_CLASS_MAPPINGS):
                out[node_class] = node_info(node_class)
            return web.json_response(out)

        @routes.get("/history")
        async def get_history(request):
            max_items = request.rel_url.query.get("max_items", None)
            if max_items is not None:
                max_items = int(max_items)
            return web.json_response(self.prompt_queue.get_history(max_items=max_items))

        @routes.get("/history/{prompt_id}")
        async def get_history_prompt_id(request):
            prompt_id = request.match_info.get("prompt_id", None)
            return web.json_response(self.prompt_queue.get_history(prompt_id=prompt_id))

        @routes.get("/queue")
        async def get_queue(request):
            queue_info = {}
            current_queue = self.prompt_queue.get_current_queue_volatile()
            queue_info['queue_running'] = current_queue[0]
            queue_info['queue_pending'] = current_queue[1]
            return web.json_response(queue_info)
        
        # HTTP endpoints for logs removed - use WebSockets only
        
        @routes.post("/prompt")
        async def post_prompt(request):
            logging.debug("got prompt - exporting workflow")
            json_data =  await request.json()
            json_data = self.trigger_on_prompt(json_data)
            
            # Debug: log what fields are available
            logging.debug(f"json_data keys: {list(json_data.keys())}")
            if "extra_data" in json_data:
                extra_data = json_data.get("extra_data", {})
                logging.debug(f"extra_data keys: {list(extra_data.keys())}")
                if "extra_pnginfo" in extra_data:
                    pnginfo = extra_data["extra_pnginfo"]
                    if "workflow" in pnginfo:
                        workflow_nodes = pnginfo["workflow"].get("nodes", [])
                        logging.debug(f"Found full workflow in extra_data with {len(workflow_nodes)} nodes")
                        # Check for Label nodes
                        label_count = sum(1 for n in workflow_nodes if n.get("type") == "Label")
                        logging.debug(f"Full workflow contains {label_count} Label nodes")

            if "number" in json_data:
                number = float(json_data['number'])
            else:
                number = self.number
                if "front" in json_data:
                    if json_data['front']:
                        number = -number

                self.number += 1

            if "prompt" in json_data:
                prompt = json_data["prompt"]
                export_target = json_data.get("export_target", "local")
                run_after_export = json_data.get("run_after_export", False)
                workflow_path = json_data.get("workflow_path")
                runner_args = json_data.get("runner_args", "")
                
                # FAIL FAST: Extract workflow name from path if provided
                if workflow_path:
                    # Path format is like "workflows/MNIST_Test.json"
                    # Extract just the filename without extension
                    import os
                    workflow_name = os.path.splitext(os.path.basename(workflow_path))[0]
                    if workflow_name:
                        self.current_workflow_name = workflow_name
                        logging.debug(f"Updated workflow name from path: {workflow_name}")
                    else:
                        error_msg = f"Invalid workflow path: {workflow_path}"
                        logging.error(error_msg)
                        return web.json_response({
                            "error": error_msg,
                            "node_errors": {}
                        }, status=400)
                
                logging.info(f"Export target: {export_target}, Run after export: {run_after_export}")
                
                # EXPORT WORKFLOW INSTEAD OF EXECUTE
                try:
                    # Import the export system
                    import os
                    from export_system.graph_exporter import GraphExporter
                    from export_system.node_exporters import register_all_exporters
                    
                    # Create exporter
                    exporter = GraphExporter()
                    register_all_exporters(exporter)

                    # FAIL FAST: Workflow name must be tracked
                    if not self.current_workflow_name:
                        error_msg = "Cannot export: No workflow is currently loaded. Please load a workflow first."
                        logging.error(error_msg)
                        return web.json_response({
                            "error": error_msg,
                            "node_errors": {}
                        }, status=400)
                    
                    workflow_name = self.current_workflow_name
                    logging.info(f"Exporting workflow: {workflow_name}")
                    
                    # Sanitize workflow name for filesystem
                    import re
                    safe_name = re.sub(r'[^\w\s-]', '', workflow_name)
                    safe_name = re.sub(r'[-\s]+', '-', safe_name)
                    if not safe_name:
                        safe_name = "workflow"
                    
                    # Try to get full workflow from extra_data (includes Label nodes)
                    full_workflow = None
                    if "extra_data" in json_data:
                        extra_data = json_data.get("extra_data", {})
                        if "extra_pnginfo" in extra_data:
                            pnginfo = extra_data["extra_pnginfo"]
                            if "workflow" in pnginfo:
                                full_workflow = pnginfo["workflow"]
                                logging.debug(f"Using full workflow from extra_data with {len(full_workflow.get('nodes', []))} nodes")
                    
                    if full_workflow:
                        # Use the full workflow which includes Label nodes
                        workflow = full_workflow
                        # Add/update metadata
                        if "metadata" not in workflow:
                            workflow["metadata"] = {}
                        workflow["metadata"].update({
                            "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "workflow_id": str(number),
                            "workflow_name": workflow_name
                        })
                    else:
                        # Fallback: Create workflow structure from prompt (Label nodes won't be available)
                        logging.info("Building workflow from prompt data (Label nodes will not be available)")
                        workflow = {
                            "nodes": [],
                            "links": [],
                            "metadata": {
                                "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "workflow_id": str(number),
                                "workflow_name": workflow_name
                            }
                        }
                        
                        # Convert prompt format to workflow format
                        # The prompt is a dict of node_id -> node_data
                        for node_id, node_data in prompt.items():
                            workflow["nodes"].append({
                                "id": node_id,
                                "class_type": node_data.get("class_type", "Unknown"),
                                "inputs": node_data.get("inputs", {})
                            })
                        
                        # Extract links from node inputs
                        link_id = 1
                        for node_id, node_data in prompt.items():
                            inputs = node_data.get("inputs", {})
                            for input_name, input_value in inputs.items():
                                # Check if input is a link (usually a list like [node_id, slot])
                                if isinstance(input_value, list) and len(input_value) == 2:
                                    source_node_id, source_slot = input_value
                                    # Find target slot index (you may need to adjust this)
                                    target_slot = 0  # Default, might need mapping
                                    workflow["links"].append([
                                        link_id,
                                        str(source_node_id),
                                        source_slot,
                                        str(node_id),
                                        target_slot
                                    ])
                                    link_id += 1
                    
                    # Create export directory structure
                    export_base_dir = os.path.join("export_system", "exports")
                    workflow_export_dir = os.path.join(export_base_dir, safe_name)
                    workflow_export_path = Path(workflow_export_dir)
                    
                    # Export the workflow to the target directory
                    # Note: Do NOT create directory here - exporter handles directory creation
                    exported_runner_path = exporter.export_workflow(workflow, workflow_export_path)
                    
                    # Verify the export was successful
                    if not os.path.exists(exported_runner_path):
                        raise Exception(f"Export failed: runner.py not found at {exported_runner_path}")
                    
                    logging.debug(f"Export saved to: {workflow_export_dir}")
                    
                    # Handle remote export if target is not local
                    if export_target != "local":
                        # FAIL FAST: Remote export requires agent connection
                        if not self.agent_connected:
                            error_msg = f"Cannot export to remote target '{export_target}': Agent server not connected"
                            logging.error(f"[EXPORT FAILED] {error_msg}")
                            return web.json_response({
                                "error": error_msg,
                                "node_errors": {}
                            }, status=400)
                        
                        try:
                            logging.info(f"Deploying to remote client: {export_target}")
                            
                            # Read all exported files
                            files_to_deploy = {}
                            for root, dirs, files in os.walk(workflow_export_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    relative_path = os.path.relpath(file_path, workflow_export_dir)
                                    with open(file_path, 'rb') as f:
                                        files_to_deploy[relative_path] = f.read().decode('utf-8')
                            
                            # Generate content-based workflow ID
                            import hashlib
                            import json as json_module
                            workflow_json = json_module.dumps(prompt, sort_keys=True)
                            content_hash = hashlib.sha256(workflow_json.encode()).hexdigest()[:12]
                            workflow_id = f"wf_{content_hash}"
                            
                            # Send deploy command to agent server
                            deploy_msg = {
                                "type": "deploy_workflow",
                                "workflow_id": workflow_id,
                                "workflow_name": safe_name,
                                "client_id": export_target,
                                "files": files_to_deploy,
                                "run_after_deploy": run_after_export,
                                "runner_args": runner_args
                            }
                            
                            await self.agent_ws.send_json(deploy_msg)
                            logging.info(f"Deployment request sent to agent server")
                            
                            return web.json_response({
                                "success": True,
                                "export_path": workflow_export_dir,
                                "files": ["runner.py"],
                                "message": f"Workflow exported locally and deploying to {export_target}",
                                "remote_deployment": "initiated"
                            })
                            
                        except Exception as e:
                            error_msg = f"Remote deployment to '{export_target}' failed: {e}"
                            logging.error(f"[EXPORT FAILED] {error_msg}")
                            # Return error - remote deployment was requested but failed
                            return web.json_response({
                                "error": error_msg,
                                "node_errors": {},
                                "export_path": workflow_export_dir,
                                "message": f"Workflow exported locally but remote deployment failed"
                            }, status=400)
                    
                    # Return success for local export
                    return web.json_response({
                        "success": True,
                        "export_path": workflow_export_dir,
                        "files": ["runner.py"],
                        "message": f"Workflow exported to: {workflow_export_dir}"
                    })
                    
                except Exception as e:
                    logging.error(f"Export failed: {str(e)}")
                    import traceback
                    tb_str = traceback.format_exc()
                    logging.error(f"Traceback:\n{tb_str}")
                    traceback.print_exc()
                    return web.json_response({
                        "error": f"Export failed: {str(e)}",
                        "details": tb_str
                    }, status=500)
                
        @routes.post("/queue")
        async def post_queue(request):
            json_data =  await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_queue()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    delete_func = lambda a: a[1] == id_to_delete
                    self.prompt_queue.delete_queue_item(delete_func)

            return web.Response(status=200)

        @routes.post("/interrupt")
        async def post_interrupt(request):
            await nodes.interrupt_processing()
            return web.Response(status=200)

        @routes.post("/free")
        async def post_free(request):
            json_data = await request.json()
            unload_models = json_data.get("unload_models", False)
            free_memory = json_data.get("free_memory", False)
            if unload_models:
                self.prompt_queue.set_flag("unload_models", unload_models)
            if free_memory:
                self.prompt_queue.set_flag("free_memory", free_memory)
            return web.Response(status=200)

        @routes.post("/history")
        async def post_history(request):
            json_data =  await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_history()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    self.prompt_queue.delete_history_item(id_to_delete)

            return web.Response(status=200)
        
        @routes.post("/remote_command")
        async def handle_remote_command(request):
            """Handle remote command requests for server control."""
            import time
            from datetime import datetime
            
            try:
                json_data = await request.json()
                logging.info(f"[Remote Command] Received request: {json_data}")
                
                # Simple auth check (can be enhanced later)
                auth_token = json_data.get("auth")
                expected_token = os.environ.get("DNNE_REMOTE_AUTH")
                if expected_token and auth_token != expected_token:
                    return web.json_response({
                        "success": False,
                        "message": "Authentication failed",
                        "timestamp": datetime.now().isoformat()
                    }, status=401)
                
                command = json_data.get("command")
                args = json_data.get("args", {})
                request_id = json_data.get("request_id")
                
                logging.debug(f"[Remote Command] Received: {command} with args: {args}")
                
                # Command dispatch
                if command == "restart":
                    # Handle server restart
                    delay = args.get("delay", 2)
                    reason = args.get("reason", "Remote command")
                    preserve_args = args.get("preserve_args", True)
                    
                    logging.info(f"[Remote Command] Server restart requested: {reason}")
                    
                    # Schedule restart
                    async def do_restart():
                        await asyncio.sleep(delay)
                        logging.info("[Remote Command] Executing restart...")
                        
                        # Close connections gracefully
                        if hasattr(self, 'agent_ws') and self.agent_ws:
                            await self.agent_ws.close()
                        
                        # Start new server window using dnne.bat
                        import subprocess
                        import platform
                        
                        if platform.system() == "Windows":
                            # Get the directory where server.py is located
                            # __file__ gives us the actual file path as Windows sees it
                            current_file = os.path.abspath(__file__)
                            server_dir = os.path.dirname(current_file)
                            
                            # dnne.bat is in the same directory as server.py
                            dnne_bat = os.path.join(server_dir, "dnne.bat")
                            
                            # If the path looks like a WSL path, we need to use the Windows path
                            # The server is running on Windows, so __file__ should already be a Windows path
                            if not os.path.exists(dnne_bat):
                                # Try to find dnne.bat in current working directory
                                cwd_dnne_bat = os.path.join(os.getcwd(), "dnne.bat")
                                if os.path.exists(cwd_dnne_bat):
                                    dnne_bat = cwd_dnne_bat
                                    server_dir = os.getcwd()
                                else:
                                    logging.error(f"[Remote Command] Cannot find dnne.bat at {dnne_bat} or {cwd_dnne_bat}")
                                    return
                            
                            # Check if we should restart agent server too
                            restart_agent = args.get("restart_agent_server", False)
                            cmd = [dnne_bat]
                            if restart_agent:
                                cmd.append("--restart-agent-server")
                                logging.info(f"[Remote Command] Will restart agent server too")
                            
                            # Add any additional command line arguments for DNNE
                            # Support both old 'extra_args' and new 'dnne_extra_args' for compatibility
                            dnne_extra_args = args.get("dnne_extra_args") or args.get("extra_args", [])
                            if dnne_extra_args:
                                if isinstance(dnne_extra_args, str):
                                    # Split the string into individual arguments
                                    cmd.extend(dnne_extra_args.split())
                                else:
                                    cmd.extend(dnne_extra_args)
                                logging.info(f"[Remote Command] Adding extra args for DNNE: {dnne_extra_args}")
                            
                            # Add any additional command line arguments for agent server
                            agent_server_extra_args = args.get("agent_server_extra_args", [])
                            if agent_server_extra_args:
                                # IMPORTANT: We use the --arg=value format instead of --arg value because
                                # Windows batch files don't properly preserve quoted arguments when using %*
                                # to pass arguments through. With space-separated format, quotes get mangled.
                                # Python's argparse accepts both formats, so we use = for Windows compatibility.
                                # Example: --agent-server-args="--enable-test-port" works
                                #          --agent-server-args "--enable-test-port" fails (quotes lost)
                                if isinstance(agent_server_extra_args, str):
                                    cmd.append(f"--agent-server-args={agent_server_extra_args}")
                                else:
                                    cmd.append(f"--agent-server-args={' '.join(agent_server_extra_args)}")
                                logging.info(f"[Remote Command] Adding extra args for agent server: {agent_server_extra_args}")
                            
                            logging.info(f"[Remote Command] Starting new server with: {' '.join(cmd)}")
                            logging.info(f"[Remote Command] Working directory: {server_dir}")
                            
                            # Start dnne.bat in a new window
                            # Just call dnne.bat directly - it will handle opening its own window
                            subprocess.Popen(
                                cmd,
                                cwd=server_dir,
                                shell=False,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                            )
                            
                            # Give the new process a moment to start
                            await asyncio.sleep(0.5)
                            
                            # Exit current process
                            logging.info("[Remote Command] Shutting down current server...")
                            os._exit(0)
                        else:
                            # On Unix-like systems, restart with current args
                            os.execv(sys.executable, [sys.executable] + sys.argv)
                    
                    asyncio.create_task(do_restart())
                    
                    return web.json_response({
                        "success": True,
                        "command": command,
                        "message": f"Server will restart in {delay} seconds",
                        "data": {"delay": delay, "reason": reason},
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif command == "get_status":
                    # Get server status
                    uptime = time.time() - self.start_time if hasattr(self, 'start_time') else 0
                    
                    return web.json_response({
                        "success": True,
                        "command": command,
                        "message": "Server status retrieved",
                        "data": {
                            "uptime": uptime,
                            "version": __version__,
                            "agent_connected": self.agent_connected if hasattr(self, 'agent_connected') else False,
                            "agent_status": self.agent_connection_status if hasattr(self, 'agent_connection_status') else "unknown",
                            "queue_size": len(self.prompt_queue.get_current_queue_volatile()[1]),
                            "node_count": len(nodes.NODE_CLASS_MAPPINGS)
                        },
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif command == "reload_nodes":
                    # Reload custom nodes
                    try:
                        from comfy.cli_args import args as cli_args
                        
                        # Clear the node mappings
                        nodes.NODE_CLASS_MAPPINGS.clear()
                        nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
                        
                        # Re-initialize nodes
                        nodes.init_extra_nodes(
                            init_custom_nodes=not cli_args.disable_all_custom_nodes,
                            init_api_nodes=not cli_args.disable_api_nodes
                        )
                        
                        return web.json_response({
                            "success": True,
                            "command": command,
                            "message": "Nodes reloaded successfully",
                            "data": {
                                "node_count": len(nodes.NODE_CLASS_MAPPINGS)
                            },
                            "request_id": request_id,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        return web.json_response({
                            "success": False,
                            "command": command,
                            "message": f"Failed to reload nodes: {str(e)}",
                            "request_id": request_id,
                            "timestamp": datetime.now().isoformat()
                        }, status=500)
                
                elif command == "clear_cache":
                    # Clear various caches
                    cache_type = args.get("type", "all")
                    
                    if cache_type in ["all", "models"]:
                        comfy.model_management.cleanup_models()
                    
                    if cache_type in ["all", "nodes"]:
                        # Clear node cache if available
                        pass
                    
                    return web.json_response({
                        "success": True,
                        "command": command,
                        "message": f"Cache cleared: {cache_type}",
                        "data": {"cache_type": cache_type},
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif command == "get_logs":
                    # Get recent logs (simplified version)
                    # In a full implementation, we'd capture logs to a buffer
                    return web.json_response({
                        "success": True,
                        "command": command,
                        "message": "Log retrieval not fully implemented",
                        "data": {
                            "logs": ["Log capture not yet implemented"],
                            "note": "Future enhancement"
                        },
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat()
                    })
                
                else:
                    return web.json_response({
                        "success": False,
                        "command": command,
                        "message": f"Unknown command: {command}",
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat()
                    }, status=400)
                    
            except Exception as e:
                logging.error(f"[Remote Command] Error: {e}")
                logging.error(traceback.format_exc())
                return web.json_response({
                    "success": False,
                    "message": str(e),
                    "request_id": json_data.get("request_id") if 'json_data' in locals() else None,
                    "timestamp": datetime.now().isoformat()
                }, status=500)

    async def connect_to_agent_server(self):
        """Connect to the DNNE agent server as a UI client."""
        from dnne_config import DNNEConfig
        config = DNNEConfig()
        agent_port = config.get('dnne.agent_server.ui_port', 8767)
        agent_url = f"ws://localhost:{agent_port}/ws"
        
        logging.info(f"Starting connection to agent server at {agent_url}")
        
        try:
            self.agent_connection_status = "connecting"
            logging.info(f"Connecting to agent server at localhost:{agent_port}...")
            
            self.agent_ws = await self.client_session.ws_connect(agent_url)
            self.agent_connected = True
            self.agent_connection_status = "connected"
            logging.info("Connected to agent server successfully")
            
            # Start listening for messages
            asyncio.create_task(self.agent_message_loop())
            
        except Exception as e:
            logging.error(f" Failed to connect to agent server: {e}")
            self.agent_connected = False
            self.agent_connection_status = "error"
            # Schedule reconnection
            if self.agent_reconnect_task is None or self.agent_reconnect_task.done():
                self.agent_reconnect_task = asyncio.create_task(self.agent_reconnect_loop())
    
    async def agent_reconnect_loop(self):
        """Attempt to reconnect to the agent server periodically."""
        reconnect_delay = 5.0  # seconds
        
        while not self.agent_connected:
            await asyncio.sleep(reconnect_delay)
            await self.connect_to_agent_server()
    
    async def agent_message_loop(self):
        """Process messages from the agent server."""
        try:
            async for msg in self.agent_ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self.handle_agent_message(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logging.error(f' Agent ws error: {self.agent_ws.exception()}')
                    break
        except Exception as e:
            logging.error(f" Agent message loop error: {e}")
        finally:
            self.agent_connected = False
            self.agent_connection_status = "disconnected"
            # Schedule reconnection
            if self.agent_reconnect_task is None or self.agent_reconnect_task.done():
                self.agent_reconnect_task = asyncio.create_task(self.agent_reconnect_loop())
    
    async def handle_agent_message(self, message):
        """Handle messages from the agent server."""
        try:
            msg_type = message.get("type")
            logging.debug(f"Received agent message type: {msg_type}")
            
            # Process the message (rest of the function continues below)
            await self._process_agent_message(message, msg_type)
            
        except KeyError as e:
            logging.error(f"FAIL: Agent message missing required key: {e}")
            logging.error(f"Message was: {message}")
            # Don't crash the connection, but don't try to recover either
        except Exception as e:
            logging.error(f"FAIL: Agent message processing error: {e}")
            logging.error(f"Message was: {message}")
            # Don't crash the connection, but don't try to recover either
    
    async def _process_agent_message(self, message, msg_type):
        """Process agent message - separated for error handling."""
        # Track active workflows per client - moved from handle_agent_message
        if not hasattr(self, 'client_workflows'):
            self.client_workflows = {}  # {client_id: {workflow_id: {name, start_time}}}
        
        if msg_type == "server_state":
            # Initial state from agent server
            clients = message.get("clients", {})
            self.agent_clients = {}
            for client_id, info in clients.items():
                if info.get("connected"):
                    self.agent_clients[client_id] = {
                        "id": client_id,
                        "hostname": info.get("hostname"),
                        "platform": info.get("platform"),
                        "connected_at": info.get("connected_at")
                    }
            
            # Initialize workflow tracking
            workflows = message.get("workflows", {})
            self.client_workflows = {}
            for workflow_id, wf_info in workflows.items():
                client_id = wf_info.get("client_id")
                if client_id and wf_info.get("status") == "running":
                    if client_id not in self.client_workflows:
                        self.client_workflows[client_id] = {}
                    self.client_workflows[client_id][workflow_id] = {
                        "name": wf_info.get("name", "unknown"),
                        "start_time": wf_info.get("start_time")
                    }
            
            logging.debug(f" Received agent state: {len(self.agent_clients)} clients connected")
            
        elif msg_type == "client_connected":
            # New client connected
            client_id = message.get("client_id")
            info = message.get("info", {})
            self.agent_clients[client_id] = {
                "id": client_id,
                "hostname": info.get("hostname"),
                "platform": info.get("platform"),
                "connected_at": info.get("connected_at")
            }
            
            # Initialize empty workflow list for new client
            self.client_workflows[client_id] = {}
            
            logging.info(f" Agent client connected: {info.get('hostname')}")
            
            # Send client status update
            self.send_sync("client_status", {
                "status": "connected",
                "client_id": client_id,
                "client_hostname": info.get("hostname"),
                "platform": info.get("platform", "Unknown"),
                "timestamp": info.get("connected_at")
            })
            
        elif msg_type == "client_disconnected":
            # Client disconnected
            client_id = message.get("client_id")
            if client_id in self.agent_clients:
                hostname = self.agent_clients[client_id].get("hostname")
                del self.agent_clients[client_id]
                
                # Clean up workflow tracking
                if client_id in self.client_workflows:
                    del self.client_workflows[client_id]
                
                logging.info(f" Agent client disconnected: {hostname}")
                
                # Send client status update
                self.send_sync("client_status", {
                    "status": "disconnected",
                    "client_id": client_id,
                    "client_hostname": hostname,
                    "timestamp": datetime.now().isoformat()
                })
        
        elif msg_type == "workflow_deployed":
            # Workflow deployed to client
            workflow_id = message.get("workflow_id")
            client_id = message.get("client_id")
            workflow_name = message.get("workflow_name")
            
            # FAIL FAST: workflow_name is required
            if not workflow_name:
                logging.error(f" workflow_deployed message missing workflow_name for {workflow_id}")
                return
            
            # Store metadata for this workflow (will be moved to active_workflows when logging starts)
            client_info = self.agent_clients.get(client_id, {})
            hostname = client_info.get("hostname", client_id)
            
            # Create metadata file
            self._create_workflow_metadata(workflow_id, workflow_name, hostname)
            
            logging.info(f" Workflow {workflow_name} ({workflow_id}) deployed to {hostname}")
            self.send_sync("agent_update", {
                "action": "workflow_deployed",
                "workflow_id": workflow_id,
                "client_id": client_id
            })
            
        elif msg_type == "workflow_status":
            # Workflow status update
            status = message.get("status")
            workflow_id = message.get("workflow_id")
            client_id = message.get("client_id")
            workflow_name = message.get("workflow_name")
            
            logging.debug(f"Processing workflow_status: workflow_id={workflow_id}, status={status}, client_id={client_id}, workflow_name={workflow_name}")
            
            # FAIL FAST: workflow_name is required
            if not workflow_name:
                logging.error(f"workflow_status message missing workflow_name for {workflow_id}, status={status}")
                return
            
            logging.info(f"Workflow status: {status} for {workflow_name} ({workflow_id})")
            
            # Forward the workflow_status message to UI for components that need it (like DNNELogViewer)
            self.send_sync("workflow_status", {
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "client_id": client_id,
                "status": status
            })
            
            # Get client info
            client_info = self.agent_clients.get(client_id, {})
            hostname = client_info.get("hostname", "unknown")
            
            # Update workflow tracking
            if client_id not in self.client_workflows:
                self.client_workflows[client_id] = {}
            
            if status == "running":
                # Track workflow first, then start logging
                self.client_workflows[client_id][workflow_id] = {
                    "name": workflow_name,
                    "start_time": datetime.now().isoformat()
                }
                logging.debug(f"Status is 'running', calling _start_workflow_logging for {workflow_id}")
                self._start_workflow_logging(workflow_id, client_id)
                
            elif status in ["terminated", "completed", "failed"]:
                # Stop logging and remove from tracking
                self._stop_workflow_logging(workflow_id)
                if workflow_id in self.client_workflows.get(client_id, {}):
                    del self.client_workflows[client_id][workflow_id]
            
        elif msg_type == "workflow_log":
            # Log message from running workflow
            workflow_id = message.get("workflow_id")
            log_data = message.get("log", {})
            # _write_workflow_log now handles both file writing and WebSocket forwarding
            self._write_workflow_log(workflow_id, log_data)
            
        elif msg_type == "telemetry_update":
            # Telemetry data from workflow nodes
            workflow_id = message.get("workflow_id")
            batch = message.get("batch", [])
            self._write_telemetry_batch(workflow_id, batch)
            
        elif msg_type == "workflow_exists_response":
            # Response from agent about workflow existence check
            # Forward to all UI clients
            await self.send_to_all({
                "type": "workflow_exists_response",
                "data": message
            })
            
        else:
            # Unknown message type - log as error
            logging.error(f" Unknown agent message type: {msg_type}")
            logging.error(f" Full message: {message}")
    
    def _create_workflow_metadata(self, workflow_id, workflow_name, client_hostname):
        """Create metadata.json file for a deployed workflow."""
        import json as json_module
        from datetime import datetime
        from pathlib import Path
        
        # Create directory structure
        log_dir = Path("remote_clients") / client_hostname / f"{workflow_name}_{workflow_id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata
        metadata = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "export_timestamp": datetime.now().isoformat(),
            "client_hostname": client_hostname,
            "deployment_path": f"/tmp/dnne_work_areas/{workflow_id}"
        }
        
        # Write metadata file
        metadata_file = log_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json_module.dump(metadata, f, indent=2)
        
        logging.info(f" Created metadata for {workflow_name} at {log_dir}")
    
    def _start_workflow_logging(self, workflow_id, client_id):
        """Start logging for a workflow run."""
        from datetime import datetime
        from pathlib import Path
        
        logging.debug(f"_start_workflow_logging called: workflow_id={workflow_id}, client_id={client_id}")
        
        # Get workflow info from agent clients (temporary until fully migrated)
        client_info = self.agent_clients.get(client_id, {})
        client_hostname = client_info.get("hostname", client_id)
        logging.debug(f"Client hostname: {client_hostname}")
        
        # Get workflow name from client_workflows
        if not hasattr(self, 'client_workflows') or client_id not in self.client_workflows:
            logging.error(f"Cannot start logging: client {client_id} not in client_workflows")
            return
            
        if workflow_id not in self.client_workflows[client_id]:
            logging.error(f"Cannot start logging: workflow {workflow_id} not tracked for client {client_id}")
            return
            
        workflow_name = self.client_workflows[client_id][workflow_id].get("name")
        if not workflow_name:
            logging.error(f"Cannot start logging: workflow {workflow_id} missing name")
            return
        
        # Create log directory
        log_dir = Path("remote_clients") / client_hostname / f"{workflow_name}_{workflow_id}" / "run_logs"
        logging.debug(f"Creating log directory: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Log directory created successfully: {log_dir.exists()}")
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_dir / f"run_{timestamp}.log"
        logging.debug(f"Creating log file: {log_file}")
        
        # Open file for writing (line buffered) with UTF-8 encoding for emoji support
        try:
            file_handle = open(log_file, 'w', encoding='utf-8', buffering=1)
            
            # Store all workflow info in one place
            start_time = datetime.now()
            self.active_workflows[workflow_id] = {
                'file_handle': file_handle,
                'sequence': 0,  # Initialize sequence counter
                'name': workflow_name,
                'client_id': client_id,
                'client_hostname': client_hostname,
                'start_time': start_time,  # Store as datetime object for telemetry
                'start_time_str': start_time.isoformat(),  # Store string version for display
                'log_file': str(log_file),  # Store path for later reading
                'log_dir': log_dir  # Store log directory for telemetry
            }
            
            # Write header
            file_handle.write(f"# Workflow: {workflow_name}\n")
            file_handle.write(f"# ID: {workflow_id}\n")
            file_handle.write(f"# Client: {client_hostname}\n")
            file_handle.write(f"# Started: {self.active_workflows[workflow_id]['start_time_str']}\n")
            file_handle.write("#" * 60 + "\n\n")
            
            logging.info(f"Started logging for {workflow_name} to {log_file}")
        except Exception as e:
            logging.error(f" Failed to create log file: {e}")
    
    def _write_workflow_log(self, workflow_id, log_data):
        """Write a log entry to the workflow's log file."""
        from datetime import datetime
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            logging.debug(f"No active workflow found for {workflow_id} in _write_workflow_log")
            logging.debug(f"Active workflows: {list(self.active_workflows.keys())}")
            return  # No active workflow
        
        try:
            # Get and increment sequence number
            sequence = workflow['sequence']
            workflow['sequence'] = sequence + 1
            
            # Add sequence to log data for WebSocket (not for file!)
            log_data_with_seq = {**log_data, 'sequence': sequence}
            
            # Write to file WITHOUT sequence number (keep logs clean)
            file_handle = workflow['file_handle']
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            level = log_data.get("level", "info").upper()
            message = log_data.get("message", "")
            
            # Write formatted log entry (no sequence in file)
            file_handle.write(f"[{timestamp}] [{level}] {message}\n")
            file_handle.flush()  # Ensure immediate write
            
            # Send via WebSocket WITH sequence for deduplication
            logging.debug(f"Sending workflow_log via WebSocket for {workflow_id}, seq={sequence}")
            self.send_sync("workflow_log", {
                "workflow_id": workflow_id,
                "log": log_data_with_seq
            })
            logging.debug(f"Successfully sent workflow_log for {workflow_id}")
        except Exception as e:
            logging.error(f" Failed to write log: {e}")
            logging.error(f" Log data: {log_data}")
    
    def _stop_workflow_logging(self, workflow_id):
        """Stop logging for a workflow and close all files."""
        from datetime import datetime
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            logging.warning(f" Attempted to stop logging for unknown workflow {workflow_id}")
            return
        
        try:
            # Write footer to log file
            file_handle = workflow['file_handle']
            file_handle.write(f"\n# Stopped: {datetime.now().isoformat()}\n")
            file_handle.close()
            
            # Close all telemetry files if any
            if "telemetry_files" in workflow:
                for file_key, file_handle in workflow["telemetry_files"].items():
                    try:
                        file_handle.close()
                        logging.debug(f"📊 Closed telemetry file: {file_key}")
                    except Exception as e:
                        logging.error(f"📊 Failed to close telemetry file {file_key}: {e}")
            
            # Remove from active workflows
            del self.active_workflows[workflow_id]
            
            logging.info(f" Stopped logging for workflow {workflow_id}")
        except Exception as e:
            logging.error(f" Failed to close log file: {e}")
    
    def _write_telemetry_batch(self, workflow_id, batch):
        """Write telemetry batch to files with minimal processing."""
        from datetime import datetime
        from pathlib import Path
        import json as json_module
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            logging.error(f"FAIL: Cannot write telemetry for {workflow_id} - workflow not active")
            return
        
        # Fail fast if log_dir is missing - indicates corrupted state
        if "log_dir" not in workflow:
            logging.error(f"FAIL: Workflow {workflow_id} missing log_dir - corrupted state")
            logging.error(f"Workflow data: {workflow}")
            return
        
        # Create telemetry directory with timestamp
        log_dir = workflow["log_dir"]
        
        # Get or create telemetry directory for this run
        if "telemetry_dir" not in workflow:
            # Create timestamped telemetry directory
            timestamp = workflow.get("start_time", datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
            telemetry_base = log_dir.parent / "telemetry"
            telemetry_dir = telemetry_base / f"telem_{timestamp}"
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            workflow["telemetry_dir"] = telemetry_dir
            workflow["telemetry_files"] = {}  # Track open file handles
            logging.info(f"📊 Created telemetry directory: {telemetry_dir}")
        
        telemetry_dir = workflow["telemetry_dir"]
        telemetry_files = workflow["telemetry_files"]
        
        try:
            for packet in batch:
                if isinstance(packet, str):
                    # Pipe-delimited format for metrics
                    parts = packet.split('|')
                    if len(parts) >= 4:
                        metric_type, node_id, value, timestamp = parts[:4]
                        
                        # Get or create file handle for this node
                        if node_id not in telemetry_files:
                            file_path = telemetry_dir / f"node_{node_id}.dat"
                            telemetry_files[node_id] = open(file_path, "a")
                        
                        # Write metric: timestamp|type|value
                        telemetry_files[node_id].write(f"{timestamp}|{metric_type}|{value}\n")
                        telemetry_files[node_id].flush()
                        
                elif isinstance(packet, dict):
                    packet_type = packet.get("type")
                    
                    if packet_type == "violation_detail":
                        # Detailed violation
                        node_id = packet.get("node_id")
                        if not node_id:
                            continue
                        
                        # Get or create violations file for this node
                        violations_key = f"{node_id}_violations"
                        if violations_key not in telemetry_files:
                            file_path = telemetry_dir / f"node_{node_id}_violations.log"
                            telemetry_files[violations_key] = open(file_path, "a")
                        
                        # Write violation line
                        ts = datetime.fromtimestamp(packet.get("timestamp", 0))
                        violation_type = packet.get("violation_type", "unknown")
                        expected = packet.get("expected", 0)
                        actual = packet.get("actual", 0)
                        extra = packet.get("extra_args", "")
                        
                        if extra:
                            line = f"{ts.isoformat()} {violation_type}[{extra}] exp={expected} act={actual}\n"
                        else:
                            line = f"{ts.isoformat()} {violation_type} exp={expected} act={actual}\n"
                        
                        telemetry_files[violations_key].write(line)
                        telemetry_files[violations_key].flush()
                        
                    elif packet_type == "violation_summary":
                        # Summary of violations
                        for violation in packet.get("violations", []):
                            node_id = violation.get("node_id")
                            if not node_id:
                                continue
                            
                            violations_key = f"{node_id}_violations"
                            if violations_key not in telemetry_files:
                                file_path = telemetry_dir / f"node_{node_id}_violations.log"
                                telemetry_files[violations_key] = open(file_path, "a")
                            
                            # Write summary line
                            ts = datetime.now()
                            violation_type = violation.get("violation_type", "unknown")
                            count = violation.get("count", 0)
                            expected = violation.get("expected", 0)
                            actual_range = violation.get("actual_range", [0, 0])
                            last_actual = violation.get("last_actual", 0)
                            extra = violation.get("extra_args", "")
                            
                            if extra:
                                line = f"{ts.isoformat()} SUMMARY {violation_type}[{extra}] count={count} exp={expected} range={actual_range} last={last_actual}\n"
                            else:
                                line = f"{ts.isoformat()} SUMMARY {violation_type} count={count} exp={expected} range={actual_range} last={last_actual}\n"
                            
                            telemetry_files[violations_key].write(line)
                            telemetry_files[violations_key].flush()
                            
                    elif packet_type == "queue":
                        # Queue depth metrics
                        node_id = packet.get("node_id")
                        if node_id:
                            if node_id not in telemetry_files:
                                file_path = telemetry_dir / f"node_{node_id}.dat"
                                telemetry_files[node_id] = open(file_path, "a")
                            
                            timestamp = packet.get("timestamp", time.time())
                            queue_name = packet.get("queue", "unknown")
                            depth = packet.get("depth", 0)
                            
                            # Write as: timestamp|queue_{name}|depth
                            telemetry_files[node_id].write(f"{timestamp}|queue_{queue_name}|{depth}\n")
                            telemetry_files[node_id].flush()
                            
        except Exception as e:
            logging.error(f"📊 Failed to write telemetry: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    async def send_workflow_history(self, ws, workflow_id, client_id=None):
        """Send historical logs for a workflow to the client.
        If workflow_id is None, sends the most recent workflow logs for the given client.
        """
        logging.debug(f"📜 send_workflow_history called for workflow_id: {workflow_id}, client_id: {client_id}")
        
        # If no workflow_id provided, find the latest one for the client
        if not workflow_id and client_id:
            workflow_id = self._get_latest_workflow_id(client_id)
            if workflow_id:
                logging.debug(f"  Found latest workflow_id: {workflow_id} for client: {client_id}")
            else:
                logging.debug(f"  No workflows found for client: {client_id}")
                # Send empty response if no workflows found
                await ws.send_json({
                    "type": "workflow_log_history",
                    "data": {
                        "workflow_id": None,
                        "logs": "",
                        "last_sequence": -1
                    }
                })
                return
        
        try:
            workflow = self.active_workflows.get(workflow_id) if workflow_id else None
            logging.debug(f"  Active workflow found: {workflow is not None}")
            
            if workflow:
                # Active workflow - read the log file
                log_file = workflow['log_file']
                last_sequence = workflow['sequence'] - 1  # -1 because we increment before use
                
                # Read the log file
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                except Exception as e:
                    logging.error(f" Failed to read log file {log_file}: {e}")
                    log_content = ""
                
                # Send the historical logs
                await ws.send_json({
                    "type": "workflow_log_history",
                    "data": {
                        "workflow_id": workflow_id,
                        "logs": log_content,
                        "last_sequence": last_sequence
                    }
                })
                
                logging.debug(f" ✅ Sent historical logs for workflow {workflow_id} (last_seq: {last_sequence})")
            else:
                # Workflow not active - try to find historical logs
                logging.debug(f"  Searching for historical logs for workflow {workflow_id}")
                from pathlib import Path
                import glob
                
                log_content = ""
                last_sequence = -1
                
                # Look for log files in remote_clients directory
                # Pattern: remote_clients/{client}/{workflow_name}_{workflow_id}/run_logs/*.log
                log_pattern = f"remote_clients/*/*_{workflow_id}/run_logs/*.log"
                logging.debug(f"  Using pattern: {log_pattern}")
                log_files = glob.glob(log_pattern)
                logging.debug(f"  Found {len(log_files)} log file(s): {log_files}")
                
                if log_files:
                    # Get the most recent log file (by modification time)
                    latest_log = max(log_files, key=lambda f: Path(f).stat().st_mtime)
                    
                    try:
                        with open(latest_log, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                        logging.debug(f" 📁 Found historical log for workflow {workflow_id}: {latest_log}")
                        logging.debug(f"  Log content length: {len(log_content)} characters")
                    except Exception as e:
                        logging.error(f" Failed to read historical log file {latest_log}: {e}")
                
                # Send the historical logs (empty if not found)
                logging.debug(f"  Sending historical logs: {len(log_content)} characters")
                await ws.send_json({
                    "type": "workflow_log_history",
                    "data": {
                        "workflow_id": workflow_id,
                        "logs": log_content,
                        "last_sequence": last_sequence
                    }
                })
                
                if log_content:
                    logging.debug(f" ✅ Sent historical logs for workflow {workflow_id} (no sequence)")
                else:
                    logging.debug(f" ❌ No historical logs found for workflow {workflow_id}")
                
        except Exception as e:
            logging.error(f" Failed to send workflow history: {e}")
    
    async def send_telemetry_history(self, ws, workflow_id, client_id, telemetry_type):
        """Read and send telemetry data from disk"""
        from pathlib import Path
        import time
        
        logging.debug(f"📊 send_telemetry_history: workflow_id={workflow_id}, client_id={client_id}, type={telemetry_type}")
        
        # Try to find telemetry directory
        telemetry_dir = None
        
        # First check if it's an active workflow
        workflow = self.active_workflows.get(workflow_id) if workflow_id else None
        if workflow and 'telemetry_dir' in workflow:
            telemetry_dir = Path(workflow['telemetry_dir'])
            logging.debug(f"  Found active workflow telemetry dir: {telemetry_dir}")
        else:
            # Try to find historical telemetry
            if workflow_id:
                # Look for telemetry in the workflow's directory
                client_dir = Path(f"remote_clients/{client_id or 'local'}")
                if client_dir.exists():
                    # Find the workflow directory
                    for wf_dir in client_dir.glob(f"*_{workflow_id}"):
                        telem_base = wf_dir / "telemetry"
                        if telem_base.exists():
                            # Get the most recent telemetry directory
                            telem_dirs = list(telem_base.glob("telem_*"))
                            if telem_dirs:
                                telemetry_dir = sorted(telem_dirs)[-1]  # Most recent
                                logging.debug(f"  Found historical telemetry dir: {telemetry_dir}")
                                break
        
        # If no telemetry directory found, send empty response
        if not telemetry_dir or not telemetry_dir.exists():
            logging.debug(f"  No telemetry directory found")
            await ws.send_json({
                "type": "telemetry_history",
                "data": {
                    "workflow_id": workflow_id,
                    "telemetry_type": telemetry_type,
                    "content": f"No telemetry {telemetry_type} available.",
                    "timestamp": time.time()
                }
            })
            return
        
        telemetry_output = []
        
        try:
            if telemetry_type == 'violations':
                # Read only violation files
                violation_files = sorted(telemetry_dir.glob("*_violations.log"))
                if violation_files:
                    for violation_file in violation_files:
                        node_id = violation_file.stem.replace('node_', '').replace('_violations', '')
                        telemetry_output.append(f"\n{'='*60}\n")
                        telemetry_output.append(f"Node {node_id} Violations\n")
                        telemetry_output.append(f"{'='*60}\n")
                        with open(violation_file, 'r') as f:
                            content = f.read()
                            if content:
                                telemetry_output.append(content)
                            else:
                                telemetry_output.append("No violations recorded.\n")
                else:
                    telemetry_output.append("No violation files found.\n")
            else:  # telemetry_type == 'data'
                # Read only metric data files (exclude violations)
                metric_files = sorted([f for f in telemetry_dir.glob("node_*.dat") 
                                      if '_violations' not in str(f)])
                if metric_files:
                    for metric_file in metric_files:
                        node_id = metric_file.stem.replace('node_', '')
                        telemetry_output.append(f"\n{'='*60}\n")
                        telemetry_output.append(f"Node {node_id} Metrics\n")
                        telemetry_output.append(f"{'='*60}\n")
                        with open(metric_file, 'r') as f:
                            content = f.read()
                            if content:
                                # Format the metrics for readability
                                lines = content.strip().split('\n')
                                for line in lines[-100:]:  # Show last 100 entries
                                    telemetry_output.append(line + '\n')
                            else:
                                telemetry_output.append("No metrics recorded.\n")
                else:
                    telemetry_output.append("No metric files found.\n")
            
            # Send the telemetry data
            content = ''.join(telemetry_output) if telemetry_output else f"No {telemetry_type} available."
            await ws.send_json({
                "type": "telemetry_history",
                "data": {
                    "workflow_id": workflow_id,
                    "telemetry_type": telemetry_type,
                    "content": content,
                    "timestamp": time.time()
                }
            })
            
            logging.debug(f"  ✅ Sent telemetry {telemetry_type} ({len(content)} chars)")
            
        except Exception as e:
            logging.error(f"  ❌ Error reading telemetry: {e}")
            await ws.send_json({
                "type": "telemetry_history",
                "data": {
                    "workflow_id": workflow_id,
                    "telemetry_type": telemetry_type,
                    "content": f"Error reading telemetry: {str(e)}",
                    "timestamp": time.time()
                }
            })
    
    async def check_workflow_exists(self, ws, client_id, workflow_id):
        """Check if workflow exists on remote client"""
        from pathlib import Path
        
        if self.agent_ws and client_id != 'local':
            # Forward check to agent
            await self.agent_ws.send_json({
                "type": "check_workflow",
                "client_id": client_id,
                "workflow_id": workflow_id
            })
            # Agent will respond directly to UI
        else:
            # For local, check file system
            workflow_dir = Path(f"export_system/exports/{workflow_id}")
            exists = workflow_dir.exists() and (workflow_dir / "runner.py").exists()
            
            await ws.send_json({
                "type": "workflow_exists_response",
                "data": {
                    "client_id": client_id,
                    "workflow_id": workflow_id,
                    "exists": exists
                }
            })
            logging.debug(f"  Local workflow {workflow_id} exists: {exists}")
    
    async def run_existing_workflow(self, ws, client_id, workflow_id, runner_args):
        """Run an already deployed workflow"""
        if self.agent_ws and client_id != 'local':
            # Forward to agent
            await self.agent_ws.send_json({
                "type": "run_workflow",
                "client_id": client_id,
                "workflow_id": workflow_id,
                "runner_args": runner_args
            })
            logging.info(f"  Forwarded run command to agent for {workflow_id} on {client_id}")
        else:
            # For local, run directly
            # This would require implementing local execution logic
            # For now, just log
            logging.info(f"  Local run not yet implemented for {workflow_id}")
            # Could potentially start a subprocess here similar to how agent does it
    
    def _get_latest_workflow_id(self, client_id):
        """Get the workflow_id of the most recent workflow for a client."""
        logging.debug(f"📂 _get_latest_workflow_id called for client_id: {client_id}")
        try:
            from pathlib import Path
            import glob
            
            # If no client_id provided, we can't find logs
            if not client_id:
                logging.debug("  ❌ No client_id provided, cannot find latest logs")
                return None
            
            # Map client_id to hostname for directory lookup
            # client_id might be the hostname already or need lookup
            client_hostname = None
            for cid, client_info in self.agent_clients.items():
                if cid == client_id or client_info.get('hostname') == client_id:
                    client_hostname = client_info.get('hostname')
                    break
            
            if not client_hostname:
                logging.debug(f"  ❌ Could not find hostname for client_id: {client_id}")
                return None
            
            # Look for workflow directories for this client
            # Pattern: remote_clients/{client_hostname}/*_wf_*/
            client_dir = Path("remote_clients") / client_hostname
            if not client_dir.exists():
                logging.debug(f"  ❌ Client directory does not exist: {client_dir}")
                return None
            
            # Find all workflow directories for this client
            workflow_dirs = sorted(
                [d for d in client_dir.iterdir() if d.is_dir() and "_wf_" in d.name],
                key=lambda d: d.stat().st_mtime,
                reverse=True  # Most recent first
            )
            
            if not workflow_dirs:
                logging.debug(f"  ❌ No workflow directories found for client: {client_hostname}")
                return None
            
            # Get the most recent workflow directory
            latest_workflow_dir = workflow_dirs[0]
            workflow_name = latest_workflow_dir.name
            # Extract workflow_id from directory name (format: {name}_wf_{id})
            workflow_id = workflow_name.split("_wf_")[-1] if "_wf_" in workflow_name else None
            
            logging.debug(f"  📁 Found latest workflow: {workflow_name}")
            return workflow_id
            
        except Exception as e:
            logging.error(f" Failed to get latest workflow id: {e}")
            return None
    
    async def setup(self):
        timeout = aiohttp.ClientTimeout(total=None) # no timeout
        self.client_session = aiohttp.ClientSession(timeout=timeout)
        
        # Connect to agent server
        await self.connect_to_agent_server()

    def add_routes(self):
        self.user_manager.add_routes(self.routes)
        self.model_file_manager.add_routes(self.routes)
        self.custom_node_manager.add_routes(self.routes, self.app, nodes.LOADED_MODULE_DIRS.items())
        
        # Add DNNE-specific routes
        from dnne_hooks.routes import dnne_add_routes
        dnne_add_routes(self, self.routes)
        
        self.app.add_subapp('/internal', self.internal_routes.get_app())

        # Prefix every route with /api for easier matching for delegation.
        # This is very useful for frontend dev server, which need to forward
        # everything except serving of static files.
        # Currently both the old endpoints without prefix and new endpoints with
        # prefix are supported.
        api_routes = web.RouteTableDef()
        for route in self.routes:
            # Custom nodes might add extra static routes. Only process non-static
            # routes to add /api prefix.
            if isinstance(route, web.RouteDef):
                api_routes.route(route.method, "/api" + route.path)(route.handler, **route.kwargs)
        self.app.add_routes(api_routes)
        self.app.add_routes(self.routes)

        # Add routes from web extensions.
        for name, dir in nodes.EXTENSION_WEB_DIRS.items():
            self.app.add_routes([web.static('/extensions/' + name, dir)])

        workflow_templates_path = FrontendManager.templates_path()
        if workflow_templates_path:
            self.app.add_routes([
                web.static('/templates', workflow_templates_path)
            ])

        self.app.add_routes([
            web.static('/', self.web_root),
        ])

    def get_queue_info(self):
        prompt_info = {}
        exec_info = {}
        exec_info['queue_remaining'] = self.prompt_queue.get_tasks_remaining()
        prompt_info['exec_info'] = exec_info
        return prompt_info

    async def send(self, event, data, sid=None):
        if event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            await self.send_image(data, sid=sid)
        elif isinstance(data, (bytes, bytearray)):
            await self.send_bytes(event, data, sid)
        else:
            await self.send_json(event, data, sid)

    def encode_bytes(self, event, data):
        if not isinstance(event, int):
            raise RuntimeError(f"Binary event types must be integers, got {event}")

        packed = struct.pack(">I", event)
        message = bytearray(packed)
        message.extend(data)
        return message

    async def send_image(self, image_data, sid=None):
        image_type = image_data[0]
        image = image_data[1]
        max_size = image_data[2]
        if max_size is not None:
            if hasattr(Image, 'Resampling'):
                resampling = Image.Resampling.BILINEAR
            else:
                resampling = Image.ANTIALIAS

            image = ImageOps.contain(image, (max_size, max_size), resampling)
        type_num = 1
        if image_type == "JPEG":
            type_num = 1
        elif image_type == "PNG":
            type_num = 2

        bytesIO = BytesIO()
        header = struct.pack(">I", type_num)
        bytesIO.write(header)
        image.save(bytesIO, format=image_type, quality=95, compress_level=1)
        preview_bytes = bytesIO.getvalue()
        await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, preview_bytes, sid=sid)

    async def send_bytes(self, event, data, sid=None):
        message = self.encode_bytes(event, data)

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_bytes, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_bytes, message)

    async def send_json(self, event, data, sid=None):
        message = {"type": event, "data": data}

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_json, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_json, message)

    def send_sync(self, event, data, sid=None):
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))

    def queue_updated(self):
        self.send_sync("status", { "status": self.get_queue_info() })

    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)

    async def start(self, address, port, verbose=True, call_on_start=None):
        await self.start_multi_address([(address, port)], call_on_start=call_on_start)

    async def start_multi_address(self, addresses, call_on_start=None, verbose=True):
        runner = web.AppRunner(self.app, access_log=None)
        await runner.setup()
        ssl_ctx = None
        scheme = "http"
        if args.tls_keyfile and args.tls_certfile:
                ssl_ctx = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS_SERVER, verify_mode=ssl.CERT_NONE)
                ssl_ctx.load_cert_chain(certfile=args.tls_certfile,
                                keyfile=args.tls_keyfile)
                scheme = "https"

        if verbose:
            logging.info("Starting server\n")
        for addr in addresses:
            address = addr[0]
            port = addr[1]
            site = web.TCPSite(runner, address, port, ssl_context=ssl_ctx)
            await site.start()

            if not hasattr(self, 'address'):
                self.address = address #TODO: remove this
                self.port = port

            # Display localhost instead of 0.0.0.0 for better Windows compatibility
            if address == '0.0.0.0':
                address_print = 'localhost'
            elif ':' in address:
                address_print = "[{}]".format(address)
            else:
                address_print = address

            if verbose:
                logging.info("To see the GUI go to: {}://{}:{}".format(scheme, address_print, port))

        if call_on_start is not None:
            call_on_start(scheme, self.address, self.port)

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def trigger_on_prompt(self, json_data):
        for handler in self.on_prompt_handlers:
            try:
                json_data = handler(json_data)
            except Exception:
                logging.warning("[ERROR] An error occurred during the on_prompt_handler processing")
                logging.warning(traceback.format_exc())

        return json_data

    def send_progress_text(
        self, text: Union[bytes, bytearray, str], node_id: str, sid=None
    ):
        if isinstance(text, str):
            text = text.encode("utf-8")
        node_id_bytes = str(node_id).encode("utf-8")

        # Pack the node_id length as a 4-byte unsigned integer, followed by the node_id bytes
        message = struct.pack(">I", len(node_id_bytes)) + node_id_bytes + text

        self.send_sync(BinaryEventTypes.TEXT, message, sid)
