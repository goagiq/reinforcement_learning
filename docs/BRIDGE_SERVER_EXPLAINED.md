# Bridge Server Explained

## 🎯 What Is It?

The **Bridge Server** is a **TCP socket communication layer** between **NinjaTrader 8** (C#) and **your Python RL system**. Think of it as a translator/bridge that lets them talk to each other.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────┐
│   NinjaTrader 8 (C#)    │
│                         │
│  RLTradingStrategy.cs   │◄─── Your NT8 Strategy
└───────────┬─────────────┘
            │
            │ TCP Socket (localhost:8888)
            │ JSON Messages
            │
┌───────────▼─────────────┐
│   Bridge Server (Python)│
│                         │
│  nt8_bridge_server.py   │◄─── In-memory socket handler
└───────────┬─────────────┘
            │
            │ Callbacks & Events
            │
┌───────────▼─────────────┐
│   RL Trading System     │
│                         │
│  - RL Agent             │◄─── Makes trading decisions
│  - Reasoning Engine     │◄─── Validates decisions
│  - Risk Manager         │◄─── Checks safety
│  - Live Trading         │◄─── Orchestrates everything
└─────────────────────────┘
```

---

## 🔄 How It Works

### **Phase 1: Connection Setup**

1. **Bridge Server starts** (Python)
   ```python
   # In src/nt8_bridge_server.py
   server = NT8BridgeServer(host="localhost", port=8888)
   server.start()  # Opens TCP socket, waits for connections
   ```

2. **NT8 Strategy connects** (C#)
   ```csharp
   // In nt8_strategy/RLTradingStrategy.cs
   tcpClient = new TcpClient();
   tcpClient.Connect("localhost", 8888);  // Connects to bridge
   ```

3. **Handshake**
   - NT8 sends: `{"type":"connection", "instrument":"ES", "timeframe":1}`
   - Bridge acknowledges and is ready for data

### **Phase 2: Data Flow (Bidirectional)**

#### **NT8 → Python: Market Data**

**Every bar update**, NT8 sends:
```json
{
  "type": "market_data",
  "instrument": "ES 12-24",
  "timestamp": "2024-01-15T14:30:00",
  "data": {
    "open": 4850.25,
    "high": 4851.00,
    "low": 4849.50,
    "close": 4850.75,
    "volume": 12345,
    "bar_index": 100
  }
}
```

**Bridge receives** → Calls your callback:
```python
def _handle_market_data(self, data):
    # Convert to MarketBar
    bar = MarketBar(
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        volume=data['volume'],
        timestamp=datetime.fromisoformat(data['timestamp'])
    )
    # Process with RL agent
    self._process_market_update(bar)
```

#### **Python → NT8: Trade Signals**

**When RL decides to trade**, Python sends:
```python
# In src/live_trading.py
signal = {
    "action": "buy",  # or "sell", "hold"
    "position_size": 0.75,  # Normalized -1.0 to 1.0
    "confidence": 0.85,
    "timestamp": "2024-01-15T14:30:00"
}
bridge_server.send_trade_signal(signal)
```

**Bridge transmits** → NT8 receives:
```csharp
// In RLTradingStrategy.cs
ProcessTradeSignal(signal) {
    string action = signal.action;  // "buy"
    double size = signal.position_size;  // 0.75
    
    // Execute trade
    if (action == "buy") {
        EnterLong(Convert.ToInt32(size * maxPositionSize));
    }
}
```

---

## 📨 Message Protocol

### **Message Types**

| Type | Direction | Purpose | Example |
|------|-----------|---------|---------|
| `connection` | NT8 → Python | Initial handshake | `{"type":"connection","instrument":"ES"}` |
| `market_data` | NT8 → Python | Bar updates | `{"type":"market_data","data":{OHLCV}}` |
| `trade_signal` | Python → NT8 | Trading instructions | `{"type":"trade_signal","signal":{...}}` |
| `trade_request` | NT8 → Python | Request signal | `{"type":"trade_request"}` |
| `heartbeat` | NT8 → Python | Keep-alive | `{"type":"heartbeat"}` |
| `heartbeat_ack` | Python → NT8 | Connection alive | `{"type":"heartbeat_ack"}` |
| `status` | Python → NT8 | Status updates | `{"type":"status","status":"running"}` |

### **Data Format**

- **Protocol**: TCP/IP sockets
- **Encoding**: UTF-8 JSON
- **Delimiter**: Newline (`\n`) between messages
- **Port**: 8888 (default, configurable)
- **Host**: localhost (same machine)

---

## 🧩 Key Components

### **1. NT8BridgeServer (Python)**

```python
class NT8BridgeServer:
    def start():
        # Bind to port 8888, listen for connections
        self.socket.bind(("localhost", 8888))
        self.socket.listen(1)
        
    def _handle_client():
        # Receive JSON messages
        data = client_socket.recv(4096).decode('utf-8')
        message = json.loads(data)
        
        # Route to appropriate handler
        if message['type'] == 'market_data':
            self.on_market_data(message['data'])
        elif message['type'] == 'trade_request':
            signal = self.on_trade_request(message['data'])
            self._send_message(client_socket, signal)
```

### **2. RLTradingStrategy (C#)**

```csharp
public class RLTradingStrategy : Strategy
{
    private TcpClient tcpClient;
    private NetworkStream stream;
    
    protected override void OnBarUpdate() {
        // Send market data every bar
        SendMarketData();
    }
    
    private void ReceiveMessages() {
        // Background thread listens for signals
        while (!shouldStop) {
            if (stream.DataAvailable) {
                string message = ReadFromStream();
                ProcessReceivedMessage(message);
            }
        }
    }
}
```

---

## 🎮 Integration in Live Trading

### **How It Connects**

```python
# In src/live_trading.py
class LiveTradingSystem:
    def __init__(self):
        # Create bridge server with callbacks
        self.bridge_server = NT8BridgeServer(
            host="localhost",
            port=8888,
            on_market_data=self._handle_market_data,    # NT8 → Python
            on_trade_request=self._handle_trade_request  # Python → NT8
        )
        self.bridge_server.start()
    
    def _handle_market_data(self, data):
        # Convert to MarketBar
        bar = MarketBar(...)
        
        # Feed to RL agent
        action = self.agent.select_action(state)
        
        # Apply reasoning & risk management
        final_action = self.reasoning_engine.validate(action)
        final_action = self.risk_manager.validate(final_action)
        
        # Send signal back to NT8
        self.bridge_server.send_trade_signal({
            "action": "buy",
            "position_size": final_action,
            "confidence": 0.85
        })
```

---

## 🔧 Configuration

### **Python Side** (`src/api_server.py`)

```python
bridge_config = {
    "host": "localhost",
    "port": 8888
}
```

### **NT8 Side** (`RLTradingStrategy.cs`)

```csharp
[NinjaScriptProperty]
public string ServerHost { get; set; } = "localhost";

[NinjaScriptProperty]
public int ServerPort { get; set; } = 8888;
```

**Configurable in NT8 UI:**
- Right-click strategy → Add Strategy
- In parameters window, expand "Parameters" section
- Look for **Server Host** and **Server Port** settings
- **GroupName = "Connection"** should create a "Connection" group, but NT8 may group under "Parameters"

---

## 🚀 Starting the Bridge

### **Option 1: Via Web UI**

1. Go to **Trading** tab
2. Click **"Start Bridge"**
3. Status shows **"Bridge Server Running"**

### **Option 2: Via API**

```bash
curl http://localhost:8200/api/bridge/start
```

### **Option 3: Direct Script**

```bash
python src/nt8_bridge_server.py
```

---

## 📊 Message Flow Example

```
┌─────────────────────────────────────────────────────────────────┐
│ Full Trading Cycle                                              │
└─────────────────────────────────────────────────────────────────┘

1. NT8 Strategy loads on chart
   ↓
2. Connects to Bridge Server (localhost:8888)
   ↓
3. Bridge: "Connection accepted"
   ↓
4. NT8 → Bridge: {"type":"connection","instrument":"ES"}
   ↓
5. New bar forms in NT8
   ↓
6. NT8 → Bridge: {"type":"market_data","data":{OHLCV}}
   ↓
7. Bridge → RL System: "New bar received"
   ↓
8. RL Agent processes, decides: BUY @ 0.75
   ↓
9. Reasoning Engine: "Approves"
   ↓
10. Risk Manager: "Within limits"
    ↓
11. Bridge → NT8: {"type":"trade_signal","signal":{"action":"buy","size":0.75}}
    ↓
12. NT8 Strategy: EnterLong(1)
    ↓
13. Order fills in NT8
    ↓
14. Repeat for next bar...
```

---

## 🛡️ Error Handling

### **Connection Issues**

**Bridge not running:**
```
NT8 Log: "Failed to connect to Python server: Connection refused"
→ Start bridge server first
```

**Port in use:**
```
Python Log: "Failed to start server: [Errno 48] Address already in use"
→ Kill existing process or change port
```

### **Message Parsing**

**Invalid JSON:**
```
Bridge Log: "Invalid JSON received: {corrupted_data}"
→ NT8 continues, drops message
```

**Disconnect handling:**
```
- Bridge detects disconnect
- NT8 attempts reconnect
- Both sides log disconnection
```

---

## 🔍 Debugging

### **Enable Verbose Logging**

**Python side:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**NT8 side:**
```csharp
Print($"Sending: {json}");  // Already includes logging
```

### **Monitor Messages**

**Bridge console:**
```
NT8 Bridge Server started on localhost:8888
Waiting for NT8 strategy to connect...
NT8 strategy connected from ('127.0.0.1', 54321)
Received market data: ES
Received market data: ES
Sending trade signal: {'action': 'buy', 'position_size': 0.75}
```

**NT8 Output Window:**
```
Connected to Python RL server at localhost:8888
Received signal: buy, size: 0.75, confidence: 0.85
Entering long: 1 contracts
```

---

## 🎓 Key Concepts

### **1. Non-Blocking Architecture**

- **NT8 thread**: Reads signals from bridge
- **Python thread**: Receives market data
- **Both async**: Never blocks each other

### **2. JSON Serialization**

- **Universal format**: Works across languages
- **Type-safe**: C# objects ↔ Python dicts
- **Human-readable**: Easy to debug

### **3. Callback Pattern**

```python
# Bridge doesn't know about RL - just calls callbacks
server = NT8BridgeServer(
    on_market_data=my_custom_handler  # You provide the handler
)
```

---

## ❓ FAQ

**Q: Why TCP instead of HTTP?**  
A: **Lower latency** - No HTTP overhead, direct socket communication

**Q: Why localhost only?**  
A: **Security** - RL runs on same machine as NT8, no network exposure

**Q: Can I use different port?**  
A: **Yes!** Configure in both NT8 strategy and Python config

**Q: What if NT8 disconnects?**  
A: **Auto-reconnect** - NT8 strategy retries on next bar

**Q: How fast is it?**  
A: **<1ms latency** - Local socket + JSON is extremely fast

**Q: Does it work with multiple instruments?**  
A: **Yes** - Each chart connects independently, separate streams

---

## 🎯 Summary

**Bridge Server = Communication Layer**

- NT8 sends market data → Bridge receives → RL processes
- RL sends trade signals → Bridge transmits → NT8 executes
- **Simple, fast, reliable**

**Think of it as**: A telephone line between NT8 (broker) and your RL brain, where:
- NT8 speaks: "Market updated"
- RL responds: "Trade this"
- Bridge translates: JSON messages

**That's it!** Everything else (RL decisions, reasoning, risk management) happens in Python, then gets sent back as simple JSON signals.

---

## 📚 Related Docs

- **[NT8_CONNECTION_GUIDE.md](NT8_CONNECTION_GUIDE.md)** - Step-by-step setup
- **[src/nt8_bridge_server.py](../src/nt8_bridge_server.py)** - Source code
- **[nt8_strategy/RLTradingStrategy.cs](../nt8_strategy/RLTradingStrategy.cs)** - NT8 strategy

