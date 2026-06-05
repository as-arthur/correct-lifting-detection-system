#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ESP8266WebServer.h>
#include <time.h>
#include <LittleFS.h>

// ---------- CONFIG ----------
#define WIFI_SSID "Unknown Device" //"Unknown Device" //"realme" 
#define WIFI_PASS "22222222" //"22222222" //"11111111"

#define SERVER_HOST "10.63.216.166" // Ganti ke IP laptop Flask
#define SERVER_PORT 5000
#define SERVER_PATH "/data"

#define SAVE_TO_LITTLEFS false
#define SEND_TO_SERVER true   // set false jika mau non-aktifkan POST
#define I2C_SCAN_ON_BOOT false // true => jalankan I2C scanner waktu boot (debug)
#define BUZZER_PIN D7
bool buzzerState = true; //Test buzzer : http://192.168.100.74(IP IoT)/buzzer?state=0/state=1

#define CSV_PATH "data.csv"
const unsigned long SAMPLE_INTERVAL = 200; // ms
const long NTP_TIMEOUT_MS = 10000;

// ---------- objects ----------
Adafruit_MPU6050 mpu1;
Adafruit_MPU6050 mpu2;
ESP8266WebServer server(80);

bool mpu1_found = false;
bool mpu2_found = false;
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
  epoch_ms = (uint64_t)raw * 1000ULL + ms;
  char buf[32];
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
      f.println("datetime,epoch_ms,ts_millis,mpu_id,ax,ay,az,gx,gy,gz");
      f.close();
      Serial.printf("Created %s with header.\n", csvPath.c_str());
    } else {
      Serial.printf("Failed to create %s\n", csvPath.c_str());
    }
  }
#endif
}

void writeCSVToFile(const String &datetime, unsigned long epoch_ms, unsigned long ts_millis,
                    uint8_t id, float ax_g, float ay_g, float az_g, float gx_dps, float gy_dps, float gz_dps) {
#if SAVE_TO_LITTLEFS
  File f = LittleFS.open(csvPath.c_str(), "a");
  if (!f) {
    Serial.printf("Failed open %s for append\n", csvPath.c_str());
    return;
  }
  f.print(datetime); f.print(",");
  f.print(epoch_ms); f.print(",");
  f.print(ts_millis); f.print(",");
  f.print(id); f.print(",");
  f.print(ax_g, 5); f.print(",");
  f.print(ay_g, 5); f.print(",");
  f.print(az_g, 5); f.print(",");
  f.print(gx_dps, 3); f.print(",");
  f.print(gy_dps, 3); f.print(",");
  f.println(gz_dps, 3);
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

void sendToServer(uint8_t id, const String &datetime, unsigned long epoch_ms, unsigned long ts_millis,
                  float ax_g, float ay_g, float az_g, float gx_dps, float gy_dps, float gz_dps) {
  if (!SEND_TO_SERVER) return;
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected - skip POST");
    return;
  }

  String payload = "{";
  payload += "\"datetime\":\"" + datetime + "\"";
  payload += ",\"epoch_ms\":" + String(epoch_ms);
  payload += ",\"ts_millis\":" + String(ts_millis);
  payload += ",\"mpu_id\":" + String(id);
  payload += ",\"ax\":" + String(ax_g, 5);
  payload += ",\"ay\":" + String(ay_g, 5);
  payload += ",\"az\":" + String(az_g, 5);
  payload += ",\"gx\":" + String(gx_dps, 3);
  payload += ",\"gy\":" + String(gy_dps, 3);
  payload += ",\"gz\":" + String(gz_dps, 3);
  payload += ",\"esp_ip\":\"" + WiFi.localIP().toString() + "\"";
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

// Web server handlers (serve CSV & status)
void handleRoot() {
  String html = "<!doctype html><html><head><meta charset='utf-8'><title>ESP CSV</title></head><body>"
                "<h3>ESP8266 CSV</h3>"
                "<p><a href=\"/data.csv\">Download CSV</a></p>"
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
  server.sendHeader("Content-Disposition", String("attachment; filename=\"") + String(csvPath.substring(1)) + "\"");
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
  Serial.begin(115200);            // <<-- Serial harus paling awal
  delay(100);
  Serial.println("\n=== ESP8266 MPU6050 x2 -> CSV + POST ===");

#if SAVE_TO_LITTLEFS
  if (!LittleFS.begin()) {
    Serial.println("LittleFS.begin() failed!");
  } else {
    Serial.println("LittleFS mounted.");
  }
#endif

  // I2C (use safer pins D3 = SDA, D1 = SCL)
  Wire.begin(D3, D1);

  if (I2C_SCAN_ON_BOOT) i2cScanner();

  Serial.println("Initializing MPU1 (0x68)...");
  mpu1_found = mpu1.begin(0x68);
  Serial.println(mpu1_found ? "MPU1 OK (0x68)" : "MPU1 NOT FOUND (0x68)");

  Serial.println("Initializing MPU2 (0x69)...");
  mpu2_found = mpu2.begin(0x69);
  Serial.println(mpu2_found ? "MPU2 OK (0x69)" : "MPU2 NOT FOUND (0x69)");

  Serial.printf("mpu1_found=%d, mpu2_found=%d\n", (int)mpu1_found, (int)mpu2_found);

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

  // start webserver
  server.on("/", HTTP_GET, handleRoot);
  server.on("/data.csv", HTTP_GET, handleCSV);
  server.on("/status", HTTP_GET, handleStatus);
  server.begin();

  // ---- Boot-time POST test (JALANKAN SETELAH WIFI CONNECT) ----
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("Boot-time POST test to server...");
    String testUrl = "http://" + String(SERVER_HOST) + ":" + String(SERVER_PORT) + String(SERVER_PATH);
    Serial.println("Test URL: " + testUrl);
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
      } else {
        Serial.printf("Boot POST failed (err=%d)\n", r);
      }
      http.end();
    } else {
      Serial.println("Boot test: http.begin FAILED");
    }
  } else {
    Serial.println("Skipping boot POST: WiFi not connected");
  }

  lastSample = millis();

  //Kode Buzzer
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("Buzzer initialized (D7 LOW)");

  // Tambahkan endpoint baru untuk kontrol buzzer
  server.on("/buzzer", HTTP_GET, []() {
    if (!server.hasArg("state")) {
      server.send(400, "application/json", "{\"error\":\"missing state param (use ?state=0 or ?state=1)\"}");
      return;
    }
    String s = server.arg("state");
    if (s == "1") {
      digitalWrite(BUZZER_PIN, HIGH);
      buzzerState = true;
    } else if (s == "0") {
      digitalWrite(BUZZER_PIN, LOW);
      buzzerState = false;
    } else {
      server.send(400, "application/json", "{\"error\":\"invalid state (must be 0 or 1)\"}");
      return;
    }
    String resp = "{\"buzzer_state\":" + String(buzzerState ? "1" : "0") + "}";
    server.send(200, "application/json", resp);
    Serial.printf("Buzzer set to %s\n", buzzerState ? "ON" : "OFF");
  });

}

void loop() {
  server.handleClient();

  unsigned long now = millis();
  if (now - lastSample < SAMPLE_INTERVAL) return;
  lastSample = now;

  unsigned long epoch_ms;
  String datetime = getDateTimeISO(epoch_ms);

  sensors_event_t a, g, temp;

  if (mpu1_found) {
    mpu1.getEvent(&a, &g, &temp);
    float ax_g = a.acceleration.x / 9.80665;
    float ay_g = a.acceleration.y / 9.80665;
    float az_g = a.acceleration.z / 9.80665;
    float gx_dps = g.gyro.x * 57.295779513;
    float gy_dps = g.gyro.y * 57.295779513;
    float gz_dps = g.gyro.z * 57.295779513;
    writeCSVToFile(datetime, epoch_ms, now, 1, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps);
    sendToServer(1, datetime, epoch_ms, now, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps);
  }

  if (mpu2_found) {
    mpu2.getEvent(&a, &g, &temp);
    float ax_g = a.acceleration.x / 9.80665;
    float ay_g = a.acceleration.y / 9.80665;
    float az_g = a.acceleration.z / 9.80665;
    float gx_dps = g.gyro.x * 57.295779513;
    float gy_dps = g.gyro.y * 57.295779513;
    float gz_dps = g.gyro.z * 57.295779513;
    writeCSVToFile(datetime, epoch_ms, now, 2, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps);
    sendToServer(2, datetime, epoch_ms, now, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps);
  }
}
