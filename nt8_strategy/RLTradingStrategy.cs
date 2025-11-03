/*
 * RLTradingStrategy.cs
 * 
 * NinjaTrader 8 strategy that connects to Python RL server.
 * 
 * Setup:
 * 1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\Strategies\
 * 2. Compile in NT8: Tools → Compile
 * 3. Configure and run on a chart
 * 
 * Requirements:
 * - Python server must be running (src/nt8_bridge_server.py)
 * - Port 8888 must be available
 * 
 * NOTE: This version uses manual JSON building to avoid Newtonsoft.Json dependency
 */

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class RLTradingStrategy : Strategy
    {
        #region Variables
        
        // Connection settings
        private string serverHost = "localhost";
        private int serverPort = 8888;
        private TcpClient tcpClient;
        private NetworkStream stream;
        private bool isConnected = false;
        private Thread receiveThread;
        private bool shouldStop = false;
        
        // Trading settings
        private bool enableAutoTrading = false;
        private double maxPositionSize = 1.0;
        private double currentPositionSize = 0.0;
        
        // Data streaming
        private int barsSinceLastUpdate = 0;
        private int updateFrequency = 1; // Send data every N bars
        
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"RL Trading Strategy - Connects to Python RL server for automated trading.";
                Name = "RL Trading Strategy";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 0;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 20;
            }
            else if (State == State.Configure)
            {
                // Initialize any custom settings here
            }
            else if (State == State.DataLoaded)
            {
                // Connect to Python server
                ConnectToServer();
            }
            else if (State == State.Terminated)
            {
                // Disconnect from server
                DisconnectFromServer();
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0) return;
            if (CurrentBars[0] < BarsRequiredToTrade) return;
            
            // Send market data to Python server
            if (isConnected && barsSinceLastUpdate >= updateFrequency)
            {
                SendMarketData();
                barsSinceLastUpdate = 0;
            }
            else
            {
                barsSinceLastUpdate++;
            }
        }

        #region Server Communication
        
        private void ConnectToServer()
        {
            try
            {
                tcpClient = new TcpClient();
                tcpClient.Connect(serverHost, serverPort);
                stream = tcpClient.GetStream();
                isConnected = true;
                shouldStop = false;
                
                Print("Connected to Python RL server at " + serverHost + ":" + serverPort);
                
                // Start receiving thread
                receiveThread = new Thread(ReceiveMessages);
                receiveThread.IsBackground = true;
                receiveThread.Start();
                
                // Send initial connection message
                SendConnectionMessage();
            }
            catch (Exception ex)
            {
                Print("Failed to connect to Python server: " + ex.Message);
                isConnected = false;
            }
        }
        
        private void DisconnectFromServer()
        {
            shouldStop = true;
            isConnected = false;
            
            if (receiveThread != null && receiveThread.IsAlive)
            {
                receiveThread.Join(1000);
            }
            
            if (stream != null)
            {
                stream.Close();
            }
            
            if (tcpClient != null)
            {
                tcpClient.Close();
            }
            
            Print("Disconnected from Python RL server");
        }
        
        private void SendConnectionMessage()
        {
            if (!isConnected || stream == null) return;
            
            try
            {
                StringBuilder json = new StringBuilder();
                json.Append("{");
                json.Append("\"type\":\"connection\",");
                json.Append("\"instrument\":\"").Append(EscapeJson(Instrument.FullName)).Append("\",");
                json.Append("\"timeframe\":").Append(BarsPeriod.Value).Append(",");
                json.Append("\"timestamp\":\"").Append(DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss")).Append("\"");
                json.Append("}");
                
                byte[] data = Encoding.UTF8.GetBytes(json.ToString() + "\n");
                stream.Write(data, 0, data.Length);
                stream.Flush();
            }
            catch (Exception ex)
            {
                Print("Error sending connection message: " + ex.Message);
                isConnected = false;
            }
        }
        
        private void SendMarketData()
        {
            if (!isConnected || stream == null) return;
            
            try
            {
                StringBuilder json = new StringBuilder();
                json.Append("{");
                json.Append("\"type\":\"market_data\",");
                json.Append("\"instrument\":\"").Append(EscapeJson(Instrument.FullName)).Append("\",");
                json.Append("\"timestamp\":\"").Append(Time[0].ToString("yyyy-MM-ddTHH:mm:ss")).Append("\",");
                json.Append("\"data\":{");
                json.Append("\"open\":").Append(Open[0]).Append(",");
                json.Append("\"high\":").Append(High[0]).Append(",");
                json.Append("\"low\":").Append(Low[0]).Append(",");
                json.Append("\"close\":").Append(Close[0]).Append(",");
                json.Append("\"volume\":").Append(Volume[0]).Append(",");
                json.Append("\"bar_index\":").Append(CurrentBars[0]);
                json.Append("}");
                json.Append("}");
                
                byte[] data = Encoding.UTF8.GetBytes(json.ToString() + "\n");
                stream.Write(data, 0, data.Length);
                stream.Flush();
            }
            catch (Exception ex)
            {
                Print("Error sending market data: " + ex.Message);
            }
        }
        
        private string EscapeJson(string value)
        {
            if (string.IsNullOrEmpty(value)) return "";
            
            return value.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r");
        }
        
        private void ReceiveMessages()
        {
            byte[] buffer = new byte[4096];
            string receivedData = "";
            
            while (!shouldStop && isConnected)
            {
                try
                {
                    if (stream.DataAvailable)
                    {
                        int bytesRead = stream.Read(buffer, 0, buffer.Length);
                        if (bytesRead > 0)
                        {
                            receivedData += Encoding.UTF8.GetString(buffer, 0, bytesRead);
                            
                            // Process complete messages (separated by newlines)
                            while (receivedData.Contains("\n"))
                            {
                                int newlineIndex = receivedData.IndexOf("\n");
                                string message = receivedData.Substring(0, newlineIndex);
                                receivedData = receivedData.Substring(newlineIndex + 1);
                                
                                if (!string.IsNullOrEmpty(message))
                                {
                                    ProcessReceivedMessage(message);
                                }
                            }
                        }
                    }
                    else
                    {
                        Thread.Sleep(100); // Small delay to prevent CPU spinning
                    }
                }
                catch (Exception ex)
                {
                    Print("Error receiving message: " + ex.Message);
                    isConnected = false;
                    break;
                }
            }
        }
        
        private void ProcessReceivedMessage(string messageJson)
        {
            try
            {
                // Simple JSON parsing without Newtonsoft.Json
                string msgType = ExtractJsonValue(messageJson, "type");
                
                if (msgType == "trade_signal")
                {
                    string signalJson = ExtractJsonValue(messageJson, "signal", true);
                    if (!string.IsNullOrEmpty(signalJson))
                    {
                        ProcessTradeSignal(signalJson);
                    }
                }
                else if (msgType == "status")
                {
                    string status = ExtractJsonValue(messageJson, "status");
                    Print("Server status: " + status);
                }
                else if (msgType == "heartbeat_ack")
                {
                    // Heartbeat acknowledgment - connection is alive
                }
            }
            catch (Exception ex)
            {
                Print("Error processing message: " + ex.Message);
            }
        }
        
        private string ExtractJsonValue(string json, string key, bool nestedObject = false)
        {
            try
            {
                string searchKey = "\"" + key + "\":";
                int keyIndex = json.IndexOf(searchKey);
                if (keyIndex == -1) return "";
                
                int valueStart = keyIndex + searchKey.Length;
                
                // Skip whitespace after colon
                while (valueStart < json.Length && char.IsWhiteSpace(json[valueStart]))
                {
                    valueStart++;
                }
                
                // Handle nested objects
                if (nestedObject && valueStart < json.Length && json[valueStart] == '{')
                {
                    // Find matching closing brace
                    int braceCount = 0;
                    int start = valueStart;
                    for (int i = start; i < json.Length; i++)
                    {
                        if (json[i] == '{') braceCount++;
                        if (json[i] == '}') braceCount--;
                        if (braceCount == 0)
                        {
                            return json.Substring(start, i - start + 1);
                        }
                    }
                }
                // Handle string values
                else if (valueStart < json.Length && json[valueStart] == '"')
                {
                    valueStart++; // Skip opening quote
                    int valueEnd = valueStart;
                    
                    // Find closing quote (handle escaped quotes)
                    while (valueEnd < json.Length && json[valueEnd] != '"')
                    {
                        if (json[valueEnd] == '\\') valueEnd += 2; // Skip escaped character
                        else valueEnd++;
                    }
                    
                    string value = json.Substring(valueStart, valueEnd - valueStart);
                    return UnescapeJson(value);
                }
                // Handle numeric or boolean values
                else
                {
                    int valueEnd = valueStart;
                    while (valueEnd < json.Length && json[valueEnd] != ',' && json[valueEnd] != '}' && json[valueEnd] != ']')
                    {
                        valueEnd++;
                    }
                    
                    return json.Substring(valueStart, valueEnd - valueStart).Trim();
                }
            }
            catch
            {
                return "";
            }
            
            return "";
        }
        
        private string UnescapeJson(string value)
        {
            return value.Replace("\\\"", "\"").Replace("\\\\", "\\").Replace("\\n", "\n").Replace("\\r", "\r");
        }
        
        private void ProcessTradeSignal(string signalJson)
        {
            if (!enableAutoTrading) return;
            
            try
            {
                string action = ExtractJsonValue(signalJson, "action").ToLower();
                double positionSize = 0.0;
                double confidence = 0.0;
                
                // Try to parse position_size
                string posSizeStr = ExtractJsonValue(signalJson, "position_size");
                if (!string.IsNullOrEmpty(posSizeStr))
                {
                    double.TryParse(posSizeStr, out positionSize);
                }
                
                // Try to parse confidence
                string confStr = ExtractJsonValue(signalJson, "confidence");
                if (!string.IsNullOrEmpty(confStr))
                {
                    double.TryParse(confStr, out confidence);
                }
                
                // Default to hold if no action
                if (string.IsNullOrEmpty(action)) action = "hold";
                
                // Clamp position size
                positionSize = Math.Max(-maxPositionSize, Math.Min(maxPositionSize, positionSize));
                
                Print($"Received signal: {action}, size: {positionSize:F2}, confidence: {confidence:F2}");
                
                // Execute trade based on signal
                double currentPos = Position.Quantity;
                double targetPos = positionSize * maxPositionSize; // Convert to contracts
                
                if (Math.Abs(targetPos - currentPos) > 0.01) // Only trade if significant change
                {
                    if (targetPos > currentPos)
                    {
                        // Enter long or increase position
                        EnterLong(Convert.ToInt32(Math.Abs(targetPos - currentPos)));
                        Print($"Entering long: {Math.Abs(targetPos - currentPos):F0} contracts");
                    }
                    else if (targetPos < currentPos)
                    {
                        // Exit or reverse
                        if (currentPos > 0)
                        {
                            ExitLong(Convert.ToInt32(Math.Abs(targetPos - currentPos)));
                            Print($"Exiting long: {Math.Abs(targetPos - currentPos):F0} contracts");
                        }
                        else
                        {
                            // Reverse to short (if supported)
                            EnterShort(Convert.ToInt32(Math.Abs(targetPos)));
                            Print($"Entering short: {Math.Abs(targetPos):F0} contracts");
                        }
                    }
                }
                
                currentPositionSize = positionSize;
            }
            catch (Exception ex)
            {
                Print("Error processing trade signal: " + ex.Message);
            }
        }
        
        #endregion
        
        #region Properties
        
        [NinjaScriptProperty]
        [Display(Name = "Server Host", Description = "Python server hostname", Order = 1, GroupName = "Connection")]
        public string ServerHost
        {
            get { return serverHost; }
            set { serverHost = value; }
        }
        
        [NinjaScriptProperty]
        [Range(1, 65535)]
        [Display(Name = "Server Port", Description = "Python server port", Order = 2, GroupName = "Connection")]
        public int ServerPort
        {
            get { return serverPort; }
            set { serverPort = value; }
        }
        
        [NinjaScriptProperty]
        [Display(Name = "Enable Auto Trading", Description = "Enable automatic trade execution", Order = 3, GroupName = "Trading")]
        public bool EnableAutoTrading
        {
            get { return enableAutoTrading; }
            set { enableAutoTrading = value; }
        }
        
        [NinjaScriptProperty]
        [Range(0.01, 10.0)]
        [Display(Name = "Max Position Size", Description = "Maximum position size (contracts)", Order = 4, GroupName = "Trading")]
        public double MaxPositionSize
        {
            get { return maxPositionSize; }
            set { maxPositionSize = value; }
        }
        
        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name = "Update Frequency", Description = "Send data every N bars", Order = 5, GroupName = "Trading")]
        public int UpdateFrequency
        {
            get { return updateFrequency; }
            set { updateFrequency = value; }
        }
        
        #endregion
    }
}
