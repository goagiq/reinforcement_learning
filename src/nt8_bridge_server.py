"""
NT8 Bridge Server

TCP socket server that communicates with NinjaTrader 8 strategy.
Handles market data reception and trade signal transmission.
"""

import socket
import json
import asyncio
import threading
from typing import Dict, Optional, Callable
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NT8BridgeServer:
    """TCP server for NT8 communication"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8888,
        on_market_data: Optional[Callable] = None,
        on_trade_request: Optional[Callable] = None
    ):
        self.host = host
        self.port = port
        self.on_market_data = on_market_data  # Callback for market data
        self.on_trade_request = on_trade_request  # Callback for trade requests
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.client_connected = False
        self.client_socket: Optional[socket.socket] = None
    
    def start(self):
        """Start the server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.running = True
            
            logger.info(f"NT8 Bridge Server started on {self.host}:{self.port}")
            logger.info("Waiting for NT8 strategy to connect...")
            
            # Accept connections in a thread
            threading.Thread(target=self._accept_connections, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    def _accept_connections(self):
        """Accept incoming connections from NT8"""
        while self.running:
            try:
                client_socket, address = self.socket.accept()
                logger.info(f"NT8 strategy connected from {address}")
                
                self.client_socket = client_socket
                self.client_connected = True
                
                # Handle client in a separate thread
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                ).start()
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address):
        """Handle communication with NT8 client"""
        buffer = ""
        
        while self.running and self.client_connected:
            try:
                # Receive data
                data = client_socket.recv(4096).decode('utf-8')
                
                if not data:
                    break
                
                buffer += data
                
                # Process complete messages (separated by newlines)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line:
                        try:
                            message = json.loads(line)
                            self._process_message(message, client_socket)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON received: {line}")
                
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error handling client {address}: {e}")
                break
        
        # Client disconnected
        logger.info(f"NT8 strategy disconnected: {address}")
        self.client_connected = False
        client_socket.close()
    
    def _process_message(self, message: Dict, client_socket: socket.socket):
        """Process incoming message from NT8"""
        msg_type = message.get("type")
        
        if msg_type == "market_data":
            # Market data from NT8
            logger.debug(f"Received market data: {message.get('instrument')}")
            if self.on_market_data:
                self.on_market_data(message.get("data"))
        
        elif msg_type == "trade_request":
            # NT8 requesting trade signal
            logger.debug("NT8 requesting trade signal")
            if self.on_trade_request:
                signal = self.on_trade_request(message.get("data"))
                self._send_message(client_socket, {
                    "type": "trade_signal",
                    "signal": signal,
                    "timestamp": datetime.now().isoformat()
                })
        
        elif msg_type == "heartbeat":
            # Keep-alive heartbeat
            self._send_message(client_socket, {
                "type": "heartbeat_ack",
                "timestamp": datetime.now().isoformat()
            })
        
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    def _send_message(self, client_socket: socket.socket, message: Dict):
        """Send message to NT8 client"""
        try:
            data = json.dumps(message) + '\n'
            client_socket.sendall(data.encode('utf-8'))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    def send_trade_signal(self, signal: Dict):
        """Send trade signal to NT8"""
        if self.client_connected and self.client_socket:
            self._send_message(self.client_socket, {
                "type": "trade_signal",
                "signal": signal,
                "timestamp": datetime.now().isoformat()
            })
        else:
            logger.warning("No NT8 client connected. Cannot send trade signal.")
    
    def send_status(self, status: str, data: Optional[Dict] = None):
        """Send status update to NT8"""
        if self.client_connected and self.client_socket:
            self._send_message(self.client_socket, {
                "type": "status",
                "status": status,
                "data": data or {},
                "timestamp": datetime.now().isoformat()
            })
    
    def stop(self):
        """Stop the server"""
        self.running = False
        self.client_connected = False
        
        if self.client_socket:
            self.client_socket.close()
        
        if self.socket:
            self.socket.close()
        
        logger.info("NT8 Bridge Server stopped")


# Example usage with mock handlers
if __name__ == "__main__":
    def handle_market_data(data: Dict):
        """Handle incoming market data"""
        print(f"Received market data: {data.get('instrument')} @ {data.get('close')}")
    
    def handle_trade_request(data: Dict) -> Dict:
        """Handle trade request, return trade signal"""
        # Example: Always return "hold" for testing
        return {
            "action": "hold",
            "position_size": 0.0,
            "confidence": 0.5
        }
    
    # Create and start server
    server = NT8BridgeServer(
        on_market_data=handle_market_data,
        on_trade_request=handle_trade_request
    )
    
    try:
        server.start()
        
        # Keep running
        print("Server running. Press Ctrl+C to stop.")
        while True:
            import time
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()

