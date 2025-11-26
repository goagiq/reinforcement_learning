/*
 * RLSignalIndicator.cs
 * 
 * NinjaTrader 8 Indicator that receives signals from Python RL server
 * and exposes Signal_Trend and Signal_Trade for trade management tools.
 * 
 * Setup:
 * 1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\Indicators\
 * 2. Compile in NT8: Tools → Compile
 * 3. Add to chart
 * 4. Configure Server Host and Port to match Python bridge server
 * 
 * Signal Values:
 * Signal_Trend:
 *   -2 = downtrend strong
 *   -1 = downtrend weak
 *    1 = uptrend weak
 *    2 = uptrend strong
 * 
 * Signal_Trade:
 *   -3 = downtrend strengthening
 *   -2 = downtrend pullback
 *   -1 = downtrend start
 *    0 = no signal
 *    1 = uptrend start
 *    2 = uptrend pullback
 *    3 = uptrend strengthening
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
using System.Windows.Media;
using System.Xml.Serialization;
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

namespace NinjaTrader.NinjaScript.Indicators
{
    public class RLSignalIndicator : Indicator
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
        
        // Signal values (exposed as indicator plots)
        private int signalTrend = 0;
        private int signalTrade = 0;
        
        // Data series for plotting
        private Series<int> signalTrendSeries;
        private Series<int> signalTradeSeries;
        
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"RL Signal Indicator - Receives signals from Python RL server and exposes Signal_Trend and Signal_Trade for trade management tools.";
                Name = "RL Signal Indicator";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;  // Separate panel
                DisplayInDataBox = true;
                DrawOnPricePanel = false;
                DrawHorizontalGridLines = true;
                DrawVerticalGridLines = true;
                PaintPriceMarkers = true;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;
                IsSuspendedWhileInactive = true;
                
                // Add plots (for data box and panel display)
                AddPlot(Brushes.Blue, "Signal_Trend");
                AddPlot(Brushes.Green, "Signal_Trade");
                
                // Set plot styles
                Plots[0].Width = 2;
                Plots[1].Width = 2;
            }
            else if (State == State.Configure)
            {
                // Initialize series
                signalTrendSeries = new Series<int>(this);
                signalTradeSeries = new Series<int>(this);
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
            
            // Update plots with current signal values (for data box)
            Values[0][0] = signalTrend;
            Values[1][0] = signalTrade;
            
            // Store in series for access
            signalTrendSeries[0] = signalTrend;
            signalTradeSeries[0] = signalTrade;
            
            // Note: Visual markers removed to avoid drawing API issues
            // The signals are accessible via Signal_Trend and Signal_Trade properties
            // which is the primary requirement for trade management tools
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
                
                Print("RL Signal Indicator: Connected to Python RL server at " + serverHost + ":" + serverPort);
                
                // Start receiving thread
                receiveThread = new Thread(ReceiveMessages);
                receiveThread.IsBackground = true;
                receiveThread.Start();
                
                // Send initial connection message
                SendConnectionMessage();
            }
            catch (Exception ex)
            {
                Print("RL Signal Indicator: Failed to connect to Python server: " + ex.Message);
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
            
            Print("RL Signal Indicator: Disconnected from Python RL server");
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
                Print("RL Signal Indicator: Error sending connection message: " + ex.Message);
                isConnected = false;
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
                    Print("RL Signal Indicator: Error receiving message: " + ex.Message);
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
                    Print("RL Signal Indicator: Server status: " + status);
                }
                else if (msgType == "heartbeat_ack")
                {
                    // Heartbeat acknowledgment - connection is alive
                }
            }
            catch (Exception ex)
            {
                Print("RL Signal Indicator: Error processing message: " + ex.Message);
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
            try
            {
                // Extract signal_trend and signal_trade
                string trendStr = ExtractJsonValue(signalJson, "signal_trend");
                string tradeStr = ExtractJsonValue(signalJson, "signal_trade");
                
                if (!string.IsNullOrEmpty(trendStr))
                {
                    int.TryParse(trendStr, out signalTrend);
                }
                
                if (!string.IsNullOrEmpty(tradeStr))
                {
                    int.TryParse(tradeStr, out signalTrade);
                }
                
                // Update indicator (will be reflected in OnBarUpdate)
                // Force update by calling OnBarUpdate if needed
                if (State == State.Historical || State == State.Realtime)
                {
                    // Values will be updated on next bar update
                }
            }
            catch (Exception ex)
            {
                Print("RL Signal Indicator: Error processing trade signal: " + ex.Message);
            }
        }
        
        #endregion
        
        #region Properties
        
        /// <summary>
        /// Signal_Trend value (-2 to 2)
        /// -2 = downtrend strong, -1 = downtrend weak, 1 = uptrend weak, 2 = uptrend strong
        /// </summary>
        [Browsable(false)]
        [XmlIgnore]
        public int Signal_Trend
        {
            get { return signalTrend; }
        }
        
        /// <summary>
        /// Signal_Trade value (-3 to 3)
        /// -3 = downtrend strengthening, -2 = downtrend pullback, -1 = downtrend start,
        /// 0 = no signal, 1 = uptrend start, 2 = uptrend pullback, 3 = uptrend strengthening
        /// </summary>
        [Browsable(false)]
        [XmlIgnore]
        public int Signal_Trade
        {
            get { return signalTrade; }
        }
        
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
        
        #endregion
    }
}

