# Template variables - replaced during export
template_vars = {
    "NODE_ID": "data_streamer_1",
    "CLASS_NAME": "DataStreamerNode",
    "FILE_PATH": "./data/trajectory.csv",
    "SYNC_MODE": "none",
    "FREQUENCY_HZ": 100.0,
    "AUTO_FIRST_ROW": True,
    "LOOP_DATA": False,
    "EOF_MODE": "stop",
    "DELIMITER": ",",
    "SKIP_HEADER": True,
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Data Streamer - Streams CSV data row-by-row with configurable synchronization"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])  # No blocking inputs - DataStreamer generates data autonomously
        self.setup_outputs(["data", "done", "metadata"])
        
        # Create input queues for optional inputs that might be connected
        from asyncio import Queue
        self.input_queues["sync"] = Queue(maxsize=1)  # Trigger signal - size 1
        self.input_queues["reset"] = Queue(maxsize=1)  # Trigger signal - size 1
        
        # Configuration
        self.file_path = "{FILE_PATH}"
        self.sync_mode = "{SYNC_MODE}"
        self.frequency_hz = {FREQUENCY_HZ}
        self.auto_first_row = {AUTO_FIRST_ROW}
        self.loop_data = {LOOP_DATA}
        self.eof_mode = "{EOF_MODE}"
        self.delimiter = "{DELIMITER}"
        self.skip_header = {SKIP_HEADER}
        
        # Runtime state
        self.data = None
        self.metadata = {{}}
        self.current_row = 0
        self.total_rows = 0
        self.last_send_time = None
        self.first_row_sent = False
        self.eof_reached = False
        
    async def initialize(self):
        """Load CSV data and metadata on startup"""
        try:
            import pandas as pd
            import numpy as np
            import json
            import os
            
            # Load CSV data
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"CSV file not found: {{self.file_path}}")
            
            # Read CSV with pandas for efficiency
            df = pd.read_csv(self.file_path, delimiter=self.delimiter, 
                           header=0 if self.skip_header else None)
            
            # Convert to numpy array for fast access
            self.data = df.values.astype(np.float32)
            self.total_rows = len(self.data)
            
            if self.total_rows == 0:
                raise ValueError(f"CSV file is empty: {{self.file_path}}")
            
            # Load metadata if exists
            base_name = os.path.splitext(self.file_path)[0]
            metadata_path = f"{{base_name}}_metadata.json"
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                except Exception as e:
                    self.node_logger.warning(f"Could not load metadata: {{e}}")
            
            # Add runtime metadata
            self.metadata.update({{
                "file_path": self.file_path,
                "total_rows": self.total_rows,
                "num_columns": self.data.shape[1],
                "sync_mode": self.sync_mode,
                "frequency_hz": self.frequency_hz if self.sync_mode == "timed" else None,
                "column_names": list(df.columns) if self.skip_header else None
            }})
            
            # Send metadata once at startup
            await self.send_output("metadata", self.metadata)
            
            # For 'none' mode or auto_first_row, send first row immediately
            if self.sync_mode == "none" or (self.sync_mode == "external" and self.auto_first_row):
                await self._send_current_row()
                self.first_row_sent = True
            
            self.node_logger.info(f"Loaded {{self.total_rows}} rows from {{self.file_path}}")
            
            # Log streaming mode (actual streaming happens in run())
            if self.sync_mode == "none":
                self.node_logger.info("Streaming mode: none (as fast as possible)")
            elif self.sync_mode == "timed":
                self.node_logger.info(f"Streaming mode: timed at {{self.frequency_hz}}Hz")
            elif self.sync_mode == "external":
                self.node_logger.info("Streaming mode: external (waiting for sync trigger)")
        except Exception as e:
            self.node_logger.error(f"Failed to initialize data streamer: {{e}}")
            raise
    
    async def _send_current_row(self):
        """Send the current row as a tensor"""
        if self.data is None or self.eof_reached:
            return
        
        # Get current row and convert to tensor
        row_data = self.data[self.current_row]
        
        # Create tensor on correct device
        from framework.globals import Global as g
        device = torch.device(g.get_device())
        
        # Convert to tensor with proper shape [1, features] for batch consistency
        # Single row becomes batch of size 1
        tensor_data = torch.from_numpy(row_data).float().unsqueeze(0).to(device)
        
        # Set gradient requirements based on mode
        if not g.inference_mode:
            tensor_data.requires_grad_(True)
        
        # Send data
        await self.send_output("data", tensor_data)
        
        # Move to next row
        self.current_row += 1
        
        # Handle end of file
        if self.current_row >= self.total_rows:
            if self.loop_data:
                self.current_row = 0
                self.node_logger.debug("Looping back to start of data")
            else:
                await self._handle_eof()
    
    async def _handle_eof(self):
        """Handle end of file based on eof_mode"""
        if self.eof_mode == "pulse_done":
            await self.send_output("done", True)
            self.node_logger.info("End of file reached, sent done signal")
            if not self.loop_data:
                self.eof_reached = True
        elif self.eof_mode == "hold_last":
            # Keep current_row at last row
            self.current_row = self.total_rows - 1
            self.node_logger.info("End of file reached, holding last row")
        else:  # "stop"
            self.eof_reached = True
            self.node_logger.info("End of file reached, stopping stream")
    
    async def _continuous_stream(self):
        """Stream data continuously for 'none' mode"""
        while not self.eof_reached:
            if self.current_row < self.total_rows:
                await self._send_current_row()
            else:
                break
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.001)
    
    async def _timed_stream(self):
        """Stream data at fixed frequency for 'timed' mode"""
        import time
        interval = 1.0 / self.frequency_hz
        
        while not self.eof_reached:
            start_time = time.time()
            
            if self.current_row < self.total_rows or self.eof_mode == "hold_last":
                await self._send_current_row()
            else:
                break
            
            # Sleep for remaining time to maintain frequency
            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    async def run(self):
        """Override run for autonomous streaming modes"""
        self.running = True
        self.node_logger.info(f"Starting DataStreamer node {{self.node_id}}")
        
        try:
            # Initialize data
            await self.initialize()
            
            if self.sync_mode == "external":
                # Use default QueueNode behavior for external sync
                await super().run()
            else:
                # For 'none' and 'timed' modes, run the streaming task
                if self.sync_mode == "none":
                    await self._continuous_stream()
                elif self.sync_mode == "timed":
                    await self._timed_stream()
                    
        except asyncio.CancelledError:
            self.node_logger.info(f"Node {{self.node_id}} cancelled")
            raise
        except Exception as e:
            self.node_logger.error(f"Error in node {{self.node_id}}: {{e}}")
            raise
        finally:
            self.running = False
    
    async def compute(self, **kwargs) -> Dict[str, Any]:
        """Handle sync and reset inputs (only called in external mode)"""
        sync = kwargs.get('sync')
        reset = kwargs.get('reset')
        
        # Handle reset
        if reset is not None:
            self.current_row = 0
            self.eof_reached = False
            self.first_row_sent = False
            self.node_logger.info("Reset to beginning of data")
            
            # Send first row if auto_first_row is enabled
            if self.auto_first_row:
                await self._send_current_row()
                self.first_row_sent = True
            return {{}}
        
        # Handle sync input for 'external' mode
        if self.sync_mode == "external" and sync is not None:
            if not self.eof_reached:
                await self._send_current_row()
            return {{}}
        
        # For other modes, compute is a no-op
        return {{}}