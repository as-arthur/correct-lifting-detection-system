// ESP8266 + BMP280 -> LittleFS CSV + POST ke Flask + serve CSV
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ESP8266WebServer.h>
#include <time.h>
#include <LittleFS.h>

// ---------- CONFIG ----------
#define WIFI_SSID "Galaxy A21sFEC4"
#define WIFI_PASS "12345677"

#define SERVER_HOST "10.243.214.200"
#define SERVER_PORT 5000
#define SERVER_PATH "/data"

#define SAVE_TO_LITTLEFS false
#define SEND_TO_SERVER true
#define I2C_SCAN_ON_BOOT true

#define CSV_PATH "bmp.csv"
const unsigned long SAMPLE_INTERVAL = 2000; // ms, BMP280 lebih lambat
const long NTP_TIMEOUT_MS = 10000;

// Wiring BMP280: SDA -> D2 (GPIO4), SCL -> D1 (GPIO5) sesuai NodeMCU
#define BMP_SDA 5
#define BMP_SCL 4

// ---------- objects ----------
Adafruit_BMP280 bmp;
ESP8266WebServer server(80);

bool bmp_found = false;
unsigned long lastSample = 0;
bool ntpSynced = true;
String csvPath = String(CSV_PATH);

// ---------- helpers ----------
void initNTP() {
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  unsigned long start = millis();
  while (millis() - start < NTP_TIMEOUT_MS) {
    time_t now = time(nullptr);
    if (now > 1600000000UL) {
      ntpSynced = true;
      Serial.println("NTP synchronized.");
      return;
    }
    delay(200);
  }
  ntpSynced = false;
  Serial.println("NTP not synchronized (timeout).");
}

String getDateTimeISO(unsigned long &epoch_ms) {
  if (!ntpSynced) {
    epoch_ms = 0;
    return String("NTP_NOT_SET");
  }
  time_t raw = time(nullptr);
  raw += 7 * 3600; // UTC+7
  struct tm t;
  gmtime_r(&raw, &t);
  unsigned long ms = millis() % 1000;
  epoch_ms = ((unsigned long)raw) * 1000UL + ms;
  char buf[30];
  snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03lu",
           t.tm_year + 1900, t.tm_mon + 1, t.tm_mday,
           t.tm_hour, t.tm_min, t.tm_sec, ms);
  return String(buf);
}

void ensureCSVExists() {
#if SAVE_TO_LITTLEFS
  if (!LittleFS.exists(csvPath.c_str())) {
    File f = LittleFS.open(csvPath.c_str(), "w");
    if (f) {
      f.println("datetime,epoch_ms,ts_millis,temperature_c,pressure_hpa,altitude_m");
      f.close();
      Serial.printf("Created %s with header.\n", csvPath.c_str());
    } else {
      Serial.printf("Failed to create %s\n", csvPath.c_str());
    }
  }
#endif
}

void writeCSVToFile(const String &datetime, unsigned long epoch_ms, unsigned long ts_millis,
                    float temp_c, float pressure_hpa, float altitude_m) {
#if SAVE_TO_LITTLEFS
  File f = LittleFS.open(csvPath.c_str(), "a");
  if (!f) {
    Serial.printf("Failed open %s for append\n", csvPath.c_str());
    return;
  }
  f.print(datetime); f.print(",");
  f.print(epoch_ms); f.print(",");
  f.print(ts_millis); f.print(",");
  f.print(temp_c, 2); f.print(",");
  f.print(pressure_hpa, 2); f.print(",");
  f.println(altitude_m, 2);
  f.close();
#endif
}

bool sendHttpPostDebug(const String &full_url, const String &payload) {
  HTTPClient http;
  WiFiClient client;
  if (!http.begin(client, full_url)) {
    Serial.println("http.begin failed");
    return false;
  }
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(payload);
  if (code > 0) {
    String resp = http.getString();
    Serial.printf("HTTP %d: %s\n", code, resp.c_str());
    http.end();
    return true;
  } else {
    Serial.printf("HTTP POST failed, err=%d\n", code);
    http.end();
    return false;
  }
}

bool sendHttpManualTCP(const char *host, int port, const String &path, const String &payload) {
  WiFiClient client;
  Serial.printf("Manual connect to %s:%d ... ", host, port);
  if (!client.connect(host, port)) {
    Serial.println("FAILED");
    return false;
  }
  Serial.println("OK");
  String hostHdr = String(host) + ":" + String(port);
  String req = String("POST ") + path + " HTTP/1.1\r\n";
  req += "Host: " + hostHdr + "\r\n";
  req += "User-Agent: esp8266-manual\r\n";
  req += "Content-Type: application/json\r\n";
  req += "Connection: close\r\n";
  req += "Content-Length: " + String(payload.length()) + "\r\n\r\n";
  req += payload;
  client.print(req);
  unsigned long start = millis();
  while (!client.available() && millis() - start < 5000) yield();
  if (!client.available()) {
    Serial.println("No response (timeout)");
    client.stop();
    return false;
  }
  Serial.println("---- Response ----");
  while (client.available()) {
    Serial.println(client.readStringUntil('\n'));
  }
  Serial.println("---- End response ----");
  client.stop();
  return true;
}

void sendToServer(const String &datetime, unsigned long epoch_ms, unsigned long ts_millis,
                  float temp_c, float pressure_hpa, float altitude_m) {
  if (!SEND_TO_SERVER) return;
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected - skip POST");
    return;
  }

  String payload = "{";
  payload += "\"datetime\":\"" + datetime + "\"";
  payload += ",\"epoch_ms\":" + String(epoch_ms);
  payload += ",\"ts_millis\":" + String(ts_millis);
  payload += ",\"sensor\":\"3\"";
  payload += ",\"temperature_c\":" + String(temp_c, 2);
  payload += ",\"pressure_hpa\":" + String(pressure_hpa, 2);
  payload += ",\"altitude_m\":" + String(altitude_m, 2);
  payload += "}";

  String full_url = "http://" + String(SERVER_HOST) + ":" + String(SERVER_PORT) + String(SERVER_PATH);
  Serial.printf("POST -> %s\n", full_url.c_str());
  Serial.print("Payload: "); Serial.println(payload);

  bool ok = sendHttpPostDebug(full_url, payload);
  if (!ok) {
    Serial.println("HTTPClient failed, trying manual TCP...");
    ok = sendHttpManualTCP(SERVER_HOST, SERVER_PORT, String(SERVER_PATH), payload);
  }

  if (!ok) {
    Serial.println("All POST attempts failed.");
  }
}

// Web server handlers
void handleRoot() {
  String html = "<!doctype html><html><head><meta charset='utf-8'><title>ESP BMP280</title></head><body>"
                "<h3>ESP8266 BMP280 Logger</h3>"
                "<p><a href=\"/" + csvPath + "\">Download CSV</a></p>"
                "<p><a href=\"/status\">Status (JSON)</a></p>"
                "</body></html>";
  server.send(200, "text/html", html);
}

void handleCSV() {
#if SAVE_TO_LITTLEFS
  if (!LittleFS.exists(csvPath.c_str())) {
    server.send(404, "text/plain", "CSV not found");
    return;
  }
  File f = LittleFS.open(csvPath.c_str(), "r");
  if (!f) {
    server.send(500, "text/plain", "Unable to open CSV");
    return;
  }
  server.setContentLength(f.size());
  server.sendHeader("Content-Disposition", String("attachment; filename=\"") + csvPath.substring(1) + "\"");
  server.streamFile(f, "text/csv");
  f.close();
#else
  server.send(404, "text/plain", "CSV disabled");
#endif
}

void handleStatus() {
  String js = "{";
  js += "\"wifi_connected\":" + String(WiFi.status() == WL_CONNECTED ? "true" : "false");
  if (WiFi.status() == WL_CONNECTED) js += ",\"ip\":\"" + WiFi.localIP().toString() + "\"";
  js += ",\"bmp_found\":" + String(bmp_found ? "true" : "false");
  js += ",\"csv_path\":\"" + csvPath + "\"}";
  server.send(200, "application/json", js);
}

void i2cScanner() {
  Serial.println("I2C scanning...");
  byte count = 0;
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    byte err = Wire.endTransmission();
    if (err == 0) {
      Serial.printf("I2C device found at 0x%02X\n", addr);
      count++;
      delay(10);
    }
  }
  Serial.printf("I2C scan done. %d device(s) found.\n", count);
}

// ---------- setup & loop ----------
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n=== ESP8266 BMP280 Logger ===");

#if SAVE_TO_LITTLEFS
  if (!LittleFS.begin()) {
    Serial.println("LittleFS.begin() failed!");
  } else {
    Serial.println("LittleFS mounted.");
  }
#endif

  Wire.begin(BMP_SDA, BMP_SCL);
  Wire.setClock(400000);

  if (I2C_SCAN_ON_BOOT) i2cScanner();

  Serial.println("Initializing BMP280...");
  // Coba alamat 0x76 (default), fallback 0x77
  bmp_found = bmp.begin(0x76);
  if (!bmp_found) {
    bmp_found = bmp.begin(0x77);
    if (bmp_found) Serial.println("BMP280 OK at 0x77");
    else Serial.println("BMP280 NOT FOUND");
  } else {
    Serial.println("BMP280 OK at 0x76");
  }

  if (bmp_found) {
    // Default: oversampling x1, filter off
    bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,
                    Adafruit_BMP280::SAMPLING_X2, // temperature
                    Adafruit_BMP280::SAMPLING_X16, // pressure
                    Adafruit_BMP280::FILTER_OFF,
                    Adafruit_BMP280::STANDBY_MS_1000);
  }

  csvPath = String(CSV_PATH);
  ensureCSVExists();

  // WiFi
  Serial.printf("Connecting to WiFi '%s' ...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    Serial.print(".");
    delay(200);
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("WiFi connected. IP: "); Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("WiFi not connected after timeout.");
  }

  initNTP();

  server.on("/", HTTP_GET, handleRoot);
  server.on("/" + csvPath, HTTP_GET, handleCSV);
  server.on("/status", HTTP_GET, handleStatus);
  server.begin();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("Boot-time POST test to server...");
    String testUrl = "http://" + String(SERVER_HOST) + ":" + String(SERVER_PORT) + String(SERVER_PATH);
    HTTPClient http;
    WiFiClient testClient;
    if (http.begin(testClient, testUrl)) {
      http.addHeader("Content-Type", "application/json");
      String tp = "{\"boot\":\"ping\"}";
      int r = http.POST(tp);
      Serial.printf("Boot POST code=%d\n", r);
      if (r > 0) {
        String resp = http.getString();
        Serial.println("Boot POST response: " + resp);
      }
      http.end();
    } else {
      Serial.println("Boot test: http.begin FAILED");
    }
  } else {
    Serial.println("Skipping boot POST: WiFi not connected");
  }

  lastSample = millis();
}

void loop() {
  server.handleClient();

  unsigned long now = millis();
  if (now - lastSample < SAMPLE_INTERVAL) return;
  lastSample = now;

  if (!bmp_found) {
    Serial.println("BMP280 not found, skipping read");
    return;
  }

  unsigned long epoch_ms;
  String datetime = getDateTimeISO(epoch_ms);

  float temperature = bmp.readTemperature();      // Celsius
  float pressure = bmp.readPressure() / 100.0F;  // hPa
  float altitude = bmp.readAltitude(1013.25);    // meter (standard pressure)

  if (isnan(temperature) || isnan(pressure)) {
    Serial.println("BMP280 read error");
    return;
  }

  writeCSVToFile(datetime, epoch_ms, now, temperature, pressure, altitude);
  sendToServer(datetime, epoch_ms, now, temperature, pressure, altitude);
}